import csv
import os
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Dict
from typing import List
from typing import Sequence

import argbind
import numpy as np
import soundfile as sf
import torch
from audiotools import AudioSignal
from rich.progress import track


DEFAULT_EXTS = [".wav"]
CONVERTIBLE_EXTS = [".aif", ".aiff", ".flac", ".mp3", ".ogg", ".m4a"]


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
    from dcomposer.constants import COARSE_MIDI_NOTE_TO_COARSE_LABEL
    from dcomposer.constants import DATA_DIR
    from dcomposer.constants import FINE_MIDI_NOTE_TO_FINE_LABEL
    from dcomposer.constants import MANIFESTS_DIR
    from dcomposer.constants import SAMPLE_RATE
    from dcomposer import dsp
    from scripts.setup.packed_audio_utils import pack_audio_rows


ORGANIC_FINE_MAP = {
    "African and Eastern Percussion": "perc",
    "Punches Hits Discoblasts": "other",
    "Claps": "clap",
    "Rides": "ride",
    "Crashes": "crash",
    "Snares": "snare",
    "Cymbals": "cymbal",
    "Tabla": "perc",
    "Electronic Hits": "other",
    "Timbales": "timbale",
    "Gongs and Super Crashes": "cymbal",
    "Toms": "tom",
    "Hats": "hat",
    "Voice": "other",
    "Kicks": "kick",
    "Weird and Interesting Hits": "other",
    "Melodic Stabs and Hits": "other",
    "Western and Latin Percussion": "perc",
}

FINE_LABEL_TO_FINE_MIDI_NOTE = {
    label: note for note, label in FINE_MIDI_NOTE_TO_FINE_LABEL.items()
}
COARSE_LABEL_TO_COARSE_MIDI_NOTE = {
    label: note for note, label in COARSE_MIDI_NOTE_TO_COARSE_LABEL.items()
}
FINE_TO_COARSE_LABEL = {
    "kick": "kick",
    "snare": "snare",
    "clap": "clap",
    "snap": "clap",
    "rim": "rim",
    "hat": "hat",
    "hat_closed": "hat",
    "hat_open": "hat",
    "hat_pedal": "hat",
    "cymbal": "cymbal",
    "crash": "cymbal",
    "ride": "cymbal",
    "splash": "cymbal",
    "tom": "tom",
    "tom_low": "tom",
    "tom_mid": "tom",
    "tom_high": "tom",
    "shaker": "shaker",
    "tambourine": "tambourine",
    "cowbell": "cowbell",
    "conga": "perc",
    "bongo": "perc",
    "timbale": "perc",
    "clave": "perc",
    "woodblock": "perc",
    "bell": "perc",
    "perc": "perc",
    "other": "other",
}

OUTPUT_FIELDS = [
    "oneshot",
    "label",
    "fine_label",
    "coarse_label",
    "midi_fine_note",
    "midi_coarse_note",
    "source_dataset",
    "source_family",
    "kit_name",
    "midi_note",
    "midi_note_source",
    "velocity_layer",
    "channel_count",
    "sample_rate",
    "duration",
    "packed_audio_path",
    "packed_audio_offset",
    "packed_audio_num_frames",
    "packed_audio_num_channels",
    "packed_audio_sample_rate",
    "packed_audio_dtype",
]


def fine_label_from_organic_class(label: str) -> str:
    return ORGANIC_FINE_MAP.get((label or "").strip(), "other")


def coarse_label_from_fine(label: str) -> str:
    return FINE_TO_COARSE_LABEL.get(label, "other")


def midi_fine_note_from_fine(label: str) -> int:
    if label == "snap":
        return 39
    return int(FINE_LABEL_TO_FINE_MIDI_NOTE.get(label, -1))


def midi_coarse_note_from_fine(label: str) -> int:
    return int(COARSE_LABEL_TO_COARSE_MIDI_NOTE[coarse_label_from_fine(label)])


def get_info(path: Path):
    info = sf.info(str(path))
    return float(info.duration), int(info.samplerate), int(info.channels)


def resolve_device(device: str = None) -> torch.device:
    if device in [None, ""]:
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def write_canonical_wav(audio, sample_rate: int, wav_path: Path, device: torch.device):
    x = torch.from_numpy(audio.T)[None, :, :]
    sig = AudioSignal(x, sample_rate=int(sample_rate))
    sig = sig.to(device)
    sig = dsp.resample(sig, SAMPLE_RATE, inplace=False)
    y = sig.audio_data.detach().cpu().squeeze(0).transpose(0, 1).numpy()
    sf.write(str(wav_path), y, SAMPLE_RATE, subtype="PCM_16")


def convert_audio_to_wav(data_dir: Path, device: torch.device):
    failures = []
    candidates = [
        path
        for path in sorted(data_dir.rglob("*"))
        if path.is_file() and path.suffix.lower() in CONVERTIBLE_EXTS
    ]
    for path in track(candidates, description="Converting Organic audio to wav"):
        wav_path = path.with_suffix(".wav")
        if wav_path.exists():
            path.unlink()
            continue
        try:
            audio, sample_rate = sf.read(str(path), always_2d=True, dtype="float32")
            write_canonical_wav(audio, sample_rate, wav_path, device=device)
        except Exception:
            failures.append(path)
            continue
        path.unlink()
    return failures


def resample_existing_wavs(data_dir: Path, device: torch.device):
    wavs = [path for path in sorted(data_dir.rglob("*.wav")) if path.is_file()]
    for path in track(wavs, description="Resampling Organic wavs"):
        info = sf.info(str(path))
        if int(info.samplerate) == SAMPLE_RATE:
            continue
        audio, sample_rate = sf.read(str(path), always_2d=True, dtype="float32")
        write_canonical_wav(audio, sample_rate, path, device=device)


def collect_rows(
    data_dir: Path,
    exts: Sequence[str],
    relative_paths: bool,
    label_col: str,
) -> List[Dict]:
    exts = {e.lower() if e.startswith(".") else f".{e.lower()}" for e in exts}
    rows = []

    class_dirs = sorted(p for p in data_dir.iterdir() if p.is_dir())
    for class_dir in track(class_dirs, description="Scanning Organic one-shots"):
        raw_class = class_dir.name
        fine_label = fine_label_from_organic_class(raw_class)
        coarse_label = coarse_label_from_fine(fine_label)
        label = fine_label if label_col == "fine_label" else coarse_label

        for path in sorted(class_dir.rglob("*")):
            if not path.is_file() or path.suffix.lower() not in exts:
                continue
            try:
                duration, sample_rate, channels = get_info(path)
            except Exception:
                continue
            rel = path.relative_to(data_dir) if relative_paths else path.resolve()
            rows.append(
                {
                    "oneshot": str(rel),
                    "label": label,
                    "fine_label": fine_label,
                    "coarse_label": coarse_label,
                    "midi_fine_note": midi_fine_note_from_fine(fine_label),
                    "midi_coarse_note": midi_coarse_note_from_fine(fine_label),
                    "source_dataset": "organic_one_shots",
                    "source_family": "organic_one_shots",
                    "kit_name": "",
                    "midi_note": "",
                    "midi_note_source": "",
                    "velocity_layer": "",
                    "channel_count": channels,
                    "sample_rate": sample_rate,
                    "duration": f"{duration:.8f}",
                    "packed_audio_path": "",
                    "packed_audio_offset": "",
                    "packed_audio_num_frames": "",
                    "packed_audio_num_channels": "",
                    "packed_audio_sample_rate": "",
                    "packed_audio_dtype": "",
                }
            )
    return rows


def split_rows(rows: List[Dict], split, seed: int) -> Dict[str, List[Dict]]:
    p_train, p_val, p_test = split
    if not np.isclose(p_train + p_val + p_test, 1.0, atol=1e-6):
        raise ValueError(f"Split probabilities must sum to 1.0, got {split}")

    rng = np.random.RandomState(seed)
    grouped: Dict[str, List[Dict]] = {}
    for row in rows:
        grouped.setdefault(row["label"], []).append(row)

    out = {"train": [], "val": [], "test": []}
    for label_rows in grouped.values():
        perm = rng.permutation(len(label_rows))
        label_rows = [label_rows[int(i)] for i in perm]
        n = len(label_rows)
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
        out["train"].extend(label_rows[:n_train])
        out["val"].extend(label_rows[n_train : n_train + n_val])
        out["test"].extend(label_rows[n_train + n_val : n_train + n_val + n_test])
    return out


def write_manifests(output_dir: Path, splits: Dict[str, List[Dict]]):
    output_dir.mkdir(parents=True, exist_ok=True)
    for split_name, rows in splits.items():
        with open(output_dir / f"{split_name}.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=OUTPUT_FIELDS)
            writer.writeheader()
            writer.writerows(rows)


@argbind.bind(without_prefix=True)
def main(
    data_dir: str = str(DATA_DIR / "organic_oneshots"),
    output_dir: str = str(MANIFESTS_DIR / "organic_one_shots"),
    exts: list = DEFAULT_EXTS,
    label_col: str = "fine_label",
    train: float = 0.90,
    val: float = 0.05,
    test: float = 0.05,
    seed: int = 0,
    relative_paths: bool = False,
    device: str = None,
    write_packed_audio: bool = False,
    packed_audio_dir: str = None,
    packed_audio_num_channels: int = 2,
    packed_audio_sample_rate: int = SAMPLE_RATE,
):
    if label_col not in {"fine_label", "coarse_label"}:
        raise ValueError(
            f"label_col must be 'fine_label' or 'coarse_label', got {label_col}"
        )

    data_dir = Path(data_dir).expanduser()
    output_dir = Path(output_dir).expanduser()
    device = resolve_device(device)
    packed_audio_dir = (
        data_dir / "packed"
        if packed_audio_dir in [None, ""]
        else Path(packed_audio_dir).expanduser()
    )

    failures = convert_audio_to_wav(data_dir, device=device)
    resample_existing_wavs(data_dir, device=device)

    rows = collect_rows(
        data_dir=data_dir,
        exts=exts,
        relative_paths=relative_paths,
        label_col=label_col,
    )
    if write_packed_audio and rows:
        packed_path = packed_audio_dir / "organic_one_shots.bin"
        packed_meta = pack_audio_rows(
            rows,
            resolve_path=lambda row: (
                Path(row["oneshot"])
                if Path(row["oneshot"]).is_absolute()
                else data_dir / row["oneshot"]
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
    splits = split_rows(rows, split=(train, val, test), seed=seed)
    write_manifests(output_dir, splits)


if __name__ == "__main__":
    args = argbind.parse_args()
    with argbind.scope(args):
        main()
