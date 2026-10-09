import csv
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from contextlib import contextmanager
from pathlib import Path
from typing import Dict
from typing import List

import argbind
import numpy as np
from rich.progress import track


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
    from dcomposer.constants import DATA_DIR
    from dcomposer.constants import MANIFESTS_DIR
    from dcomposer.constants import MIDI_TIME_RES
    from dcomposer.midi import build_text_line_index
    from dcomposer.midi import extract_drum_note_events
    from dcomposer.midi import find_midi_files


def _process_midi(path: str):
    path = Path(path)
    try:
        events = extract_drum_note_events(path)
    except Exception:
        return None
    if events is None or len(events) == 0:
        return None
    return events


def split_rows(rows: List[Dict], split, seed: int):
    p_train, p_val, p_test = split
    if not np.isclose(p_train + p_val + p_test, 1.0, atol=1e-6):
        raise ValueError(f"Split probabilities must sum to 1.0, got {split}")

    rng = np.random.RandomState(seed)
    perm = rng.permutation(len(rows))
    rows = [rows[int(i)] for i in perm]
    n = len(rows)
    n_train = int(np.floor(p_train * n))
    n_val = int(np.floor(p_val * n))
    n_test = n - n_train - n_val
    if n >= 3:
        if n_val == 0:
            n_val = 1
            n_train = max(1, n_train - 1)
        if n_test == 0:
            n_test = 1
            n_train = max(1, n_train - 1)
    return {
        "train": rows[:n_train],
        "val": rows[n_train : n_train + n_val],
        "test": rows[n_train + n_val : n_train + n_val + n_test],
    }


def write_split_manifests(output_dir: Path, rows: List[Dict], split, seed: int):
    fieldnames = [
        "midi_txt",
        "midi_idx_npy",
        "midi_row",
        "source_midi",
        "source_dataset",
        "n_events",
        "duration",
        "duration_steps",
    ]
    splits = split_rows(rows, split=split, seed=seed)
    for split_name, split_rows_ in splits.items():
        with open(output_dir / f"{split_name}.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(split_rows_)


@argbind.bind(without_prefix=True)
def main(
    data_dir: str = str(DATA_DIR / "lmd_full"),
    cache_dir: str = str(DATA_DIR / "lmd_full_preprocessed"),
    output_dir: str = str(MANIFESTS_DIR / "lmd"),
    max_workers: int = 8,
    train: float = 0.90,
    val: float = 0.05,
    test: float = 0.05,
    seed: int = 0,
):
    data_dir = Path(data_dir).expanduser()
    cache_dir = Path(cache_dir).expanduser()
    output_dir = Path(output_dir).expanduser()
    cache_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    midi_paths = find_midi_files(data_dir)
    events_txt = cache_dir / "events.txt"
    idx_npy = cache_dir / "events.idx.npy"
    manifest_rows = []

    with open(events_txt, "w") as f_events, ProcessPoolExecutor(
        max_workers=max_workers
    ) as ex:
        processed = ex.map(_process_midi, [str(path) for path in midi_paths])
        line_idx = 0
        for path, events in track(
            zip(midi_paths, processed),
            total=len(midi_paths),
            description="Preprocessing Lakh MIDI",
        ):
            if events is None:
                continue
            f_events.write(" ".join(map(str, events.reshape(-1).tolist())) + "\n")
            duration_steps = int(events[-1, 0]) + 1
            manifest_rows.append(
                {
                    "midi_txt": str(events_txt),
                    "midi_idx_npy": str(idx_npy),
                    "midi_row": line_idx,
                    "source_midi": path.relative_to(data_dir).as_posix(),
                    "source_dataset": "lmd_full",
                    "n_events": int(events.shape[0]),
                    "duration": duration_steps / float(MIDI_TIME_RES),
                    "duration_steps": duration_steps,
                }
            )
            line_idx += 1

    build_text_line_index(events_txt, idx_npy, overwrite=True)
    write_split_manifests(
        output_dir, manifest_rows, split=(train, val, test), seed=seed
    )
    print(f"Created Lakh MIDI cache at {cache_dir} and manifests at {output_dir}")


if __name__ == "__main__":
    args = argbind.parse_args()
    with argbind.scope(args):
        main()
