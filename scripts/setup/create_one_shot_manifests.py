import csv
import os
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Dict
from typing import List
from typing import Sequence
from typing import Tuple

import argbind
import numpy as np
import soundfile as sf


DEFAULT_LABELS = [
    "clap",
    "cymbal",
    "hat",
    "kick",
    "snare",
    "tom",
    "rim",
    "tambourine",
    "cowbell",
    "shaker",
    "perc",
    "other",
]
DEFAULT_EXTS = [".wav", ".flac", ".aif", ".aiff", ".mp3", ".ogg", ".m4a"]


@contextmanager
def chdir(path: str):
    origin = Path().absolute()
    try:
        os.chdir(path)
        yield
    finally:
        os.chdir(origin)


_path = Path(__file__).parent.parent.parent
sys.path.insert(0, str(_path))

with chdir(_path):
    from dcomposer.constants import SAMPLE_RATE
    from scripts.setup.packed_audio_utils import pack_audio_rows


def get_info(path: Path) -> Tuple[float, int]:
    info = sf.info(str(path))
    return float(info.duration), int(info.samplerate)


def collect_rows(
    data_dir: Path,
    labels: Sequence[str],
    exts: Sequence[str],
    absolute_paths: bool = True,
) -> Dict[str, List[Dict]]:
    rows_by_label: Dict[str, List[Dict]] = {}
    exts = {e.lower() if e.startswith(".") else f".{e.lower()}" for e in exts}

    for label in labels:
        subdir = data_dir / label
        if not subdir.is_dir():
            rows_by_label[label] = []
            continue

        rows = []
        for path in sorted(subdir.rglob("*")):
            if not path.is_file() or path.suffix.lower() not in exts:
                continue

            try:
                duration, sample_rate = get_info(path)
            except Exception as error:
                raise ValueError(f"Cannot read audio file: {path}") from error

            rel = path if absolute_paths else path.relative_to(data_dir)
            rows.append(
                {
                    "oneshot": str(rel),
                    "label": label,
                    "coarse_label": label,
                    "duration": f"{duration:.8f}",
                    "sample_rate": sample_rate,
                    "filename": path.name,
                    "subdir": str(path.parent.relative_to(data_dir)),
                    "packed_audio_path": "",
                    "packed_audio_offset": "",
                    "packed_audio_num_frames": "",
                    "packed_audio_num_channels": "",
                    "packed_audio_sample_rate": "",
                    "packed_audio_dtype": "",
                }
            )
        rows_by_label[label] = rows

    return rows_by_label


def split_rows(
    rows_by_label: Dict[str, List[Dict]],
    split: Tuple[float, float, float],
    seed: int = 0,
) -> Dict[str, List[Dict]]:
    p_train, p_val, p_test = split
    total = p_train + p_val + p_test
    if min(split) < 0 or not np.isclose(total, 1.0, atol=1e-6):
        raise ValueError(f"Split probabilities must sum to 1.0, got {total}")

    rng = np.random.RandomState(seed)
    out = {"train": [], "val": [], "test": []}

    for _label, rows in rows_by_label.items():
        if not rows:
            continue

        perm = rng.permutation(len(rows))
        rows = [rows[int(i)] for i in perm]
        n_rows = len(rows)

        n_train = int(np.floor(p_train * n_rows))
        n_val = int(np.floor(p_val * n_rows))
        n_test = n_rows - n_train - n_val

        if n_rows >= 3:
            if p_val > 0 and n_val == 0:
                n_val = 1
                n_train = max(1, n_train - 1)
            if p_test > 0 and n_test == 0:
                n_test = 1
                n_train = max(1, n_train - 1)

        out["train"].extend(rows[:n_train])
        out["val"].extend(rows[n_train : n_train + n_val])
        out["test"].extend(rows[n_train + n_val : n_train + n_val + n_test])

    return out


def write_manifests(
    output_dir: Path, splits: Dict[str, List[Dict]], filename_prefix=""
):
    output_dir.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "oneshot",
        "label",
        "coarse_label",
        "duration",
        "sample_rate",
        "filename",
        "subdir",
        "packed_audio_path",
        "packed_audio_offset",
        "packed_audio_num_frames",
        "packed_audio_num_channels",
        "packed_audio_sample_rate",
        "packed_audio_dtype",
    ]

    for split_name, rows in splits.items():
        path = output_dir / f"{filename_prefix}{split_name}.csv"
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow(row)


@argbind.bind(without_prefix=True)
def main(
    data_dir: str = "",
    output_dir: str = "manifests/training",
    filename_prefix: str = "oneshots_",
    labels: list = DEFAULT_LABELS,
    exts: list = DEFAULT_EXTS,
    train: float = 0.90,
    val: float = 0.05,
    test: float = 0.05,
    seed: int = 0,
    relative_paths: bool = False,
    write_packed_audio: bool = False,
    packed_audio_dir: str = None,
    packed_audio_num_channels: int = 2,
    packed_audio_sample_rate: int = SAMPLE_RATE,
    device: str = None,
):
    if not data_dir:
        raise ValueError("Set data_dir to your class-labeled one-shot directory")
    if filename_prefix and Path(filename_prefix).name != filename_prefix:
        raise ValueError("filename_prefix must not contain directories")
    if not set(labels).issubset(DEFAULT_LABELS):
        raise ValueError(f"Use supported coarse labels: {DEFAULT_LABELS}")
    data_dir = Path(data_dir).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()
    if not data_dir.is_dir():
        raise NotADirectoryError(data_dir)
    if any(
        (output_dir / f"{filename_prefix}{split}.csv").exists()
        for split in ("train", "val", "test")
    ):
        raise FileExistsError(f"Refusing to overwrite manifests in {output_dir}")

    rows_by_label = collect_rows(
        data_dir=data_dir,
        labels=labels,
        exts=exts,
        absolute_paths=not relative_paths,
    )
    if not any(rows_by_label.values()):
        raise ValueError(f"No supported audio files in class folders under {data_dir}")
    if write_packed_audio:
        packed_audio_dir = (
            data_dir / "packed"
            if packed_audio_dir in [None, ""]
            else Path(packed_audio_dir).expanduser()
        )
        for label, rows in rows_by_label.items():
            if not rows:
                continue
            packed_path = packed_audio_dir / f"{label}.bin"
            packed_meta = pack_audio_rows(
                rows,
                resolve_path=lambda row: (
                    Path(row["oneshot"])
                    if Path(row["oneshot"]).is_absolute()
                    else (data_dir / row["oneshot"])
                ),
                packed_audio_path=packed_path,
                sample_rate=int(packed_audio_sample_rate),
                num_channels=int(packed_audio_num_channels),
                device=device,
            )
            for row in rows:
                source_path = (
                    Path(row["oneshot"])
                    if Path(row["oneshot"]).is_absolute()
                    else (data_dir / row["oneshot"])
                )
                row.update(packed_meta[str(source_path.resolve())])
    splits = split_rows(
        rows_by_label=rows_by_label,
        split=(train, val, test),
        seed=seed,
    )
    for name, fraction in zip(("train", "val", "test"), (train, val, test)):
        if fraction > 0 and not splits[name]:
            raise ValueError(f"Empty {name} split; provide more samples per class")
    write_manifests(output_dir, splits, filename_prefix=filename_prefix)

    print(f"Created manifests at {output_dir}")
    for split_name in ["train", "val", "test"]:
        counts = {label: 0 for label in labels}
        for row in splits[split_name]:
            counts[row["label"]] += 1
        total = sum(counts.values())
        print(f"\n[{split_name}] n={total}")
        for label, count in counts.items():
            print(f"  {label:12s} {count}")


if __name__ == "__main__":
    args = argbind.parse_args()
    with argbind.scope(args):
        main()
