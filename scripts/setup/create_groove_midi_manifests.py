import csv
import importlib.util
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from contextlib import contextmanager
from pathlib import Path
from typing import Dict
from typing import List

import argbind
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


def _process_midi(path: str):
    path = Path(path)
    try:
        events = extract_drum_note_events(path)
    except Exception:
        return None
    if events is None or len(events) == 0:
        return None
    return events


def _cache_line_count(txt_path: Path) -> int:
    with open(txt_path, "rb") as f:
        return sum(1 for line in f if line.strip())


def _build_manifest_rows_from_cache(
    source_rows: List[Dict],
    events_txt: Path,
    idx_npy: Path,
) -> Dict[str, List[Dict]]:
    manifest_rows = {"train": [], "val": [], "test": []}

    with open(events_txt) as f:
        for line_idx, (row, line) in enumerate(zip(source_rows, f)):
            vals = [int(x) for x in line.strip().split()]
            if not vals:
                continue

            n_events = len(vals) // 3
            duration_steps = int(vals[-3]) + 1
            split_name = (row.get("split") or "").strip().lower()
            if split_name == "validation":
                split_name = "val"
            if split_name not in manifest_rows:
                raise ValueError(
                    f"Unexpected Groove MIDI split label {row.get('split')!r} for {row['__midi_rel__']}"
                )

            manifest_rows[split_name].append(
                {
                    "midi_txt": str(events_txt),
                    "midi_idx_npy": str(idx_npy),
                    "midi_row": line_idx,
                    "source_midi": row["__midi_rel__"],
                    "source_dataset": "groove_midi",
                    "n_events": n_events,
                    "duration": duration_steps / float(MIDI_TIME_RES),
                    "duration_steps": duration_steps,
                }
            )

    return manifest_rows


def write_split_manifests(output_dir: Path, split_rows: Dict[str, List[Dict]]):
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
    for split_name, rows in split_rows.items():
        with open(output_dir / f"{split_name}.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)


@argbind.bind(without_prefix=True)
def main(
    data_dir: str = str(DATA_DIR / "groove_midi"),
    cache_dir: str = str(DATA_DIR / "groove_midi_preprocessed"),
    output_dir: str = str(MANIFESTS_DIR / "groove_midi"),
    max_workers: int = 8,
):
    data_dir = Path(data_dir).expanduser()
    cache_dir = Path(cache_dir).expanduser()
    output_dir = Path(output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    groove_dir = data_dir / "groove"
    info_path = groove_dir / "info.csv"
    if not info_path.exists():
        raise FileNotFoundError(f"Expected {info_path}")

    source_rows = []
    with open(info_path, newline="") as f:
        for row in csv.DictReader(f):
            midi_rel = row.get("midi_filename", "").strip()
            if not midi_rel:
                continue
            midi_path = groove_dir / midi_rel
            if not midi_path.is_file():
                continue
            row["__midi_rel__"] = midi_rel
            row["__midi_path__"] = str(midi_path)
            source_rows.append(row)

    existing_cache_dirs = []
    legacy_cache_dir = DATA_DIR / "groove_midi_preprocessed"
    for candidate in [legacy_cache_dir, cache_dir]:
        candidate = Path(candidate).expanduser()
        events_txt = candidate / "events.txt"
        idx_npy = candidate / "events.idx.npy"
        if events_txt.exists() and idx_npy.exists():
            if _cache_line_count(events_txt) == len(source_rows):
                existing_cache_dirs.append(candidate)

    if existing_cache_dirs:
        chosen_cache = existing_cache_dirs[0]
        events_txt = chosen_cache / "events.txt"
        idx_npy = chosen_cache / "events.idx.npy"
        manifest_rows = _build_manifest_rows_from_cache(
            source_rows=source_rows,
            events_txt=events_txt,
            idx_npy=idx_npy,
        )
        write_split_manifests(output_dir, manifest_rows)
        print(
            "Groove MIDI split counts:",
            {name: len(rows) for name, rows in manifest_rows.items()},
        )
        print(
            f"Reused Groove MIDI cache at {chosen_cache} and wrote manifests to {output_dir}"
        )
        return

    if importlib.util.find_spec("mido") is None:
        raise ModuleNotFoundError(
            "mido is required to preprocess Groove MIDI from source when no existing cache is available"
        )

    cache_dir.mkdir(parents=True, exist_ok=True)
    events_txt = cache_dir / "events.txt"
    idx_npy = cache_dir / "events.idx.npy"
    manifest_rows = {"train": [], "val": [], "test": []}

    with open(events_txt, "w") as f_events, ProcessPoolExecutor(
        max_workers=max_workers
    ) as ex:
        processed = ex.map(_process_midi, [row["__midi_path__"] for row in source_rows])
        line_idx = 0
        for row, events in track(
            zip(source_rows, processed),
            total=len(source_rows),
            description="Preprocessing Groove MIDI",
        ):
            if events is None:
                continue
            f_events.write(" ".join(map(str, events.reshape(-1).tolist())) + "\n")
            duration_steps = int(events[-1, 0]) + 1
            split_name = (row.get("split") or "").strip().lower()
            if split_name == "validation":
                split_name = "val"
            if split_name not in manifest_rows:
                raise ValueError(
                    f"Unexpected Groove MIDI split label {row.get('split')!r} for {row['__midi_rel__']}"
                )

            manifest_rows[split_name].append(
                {
                    "midi_txt": str(events_txt),
                    "midi_idx_npy": str(idx_npy),
                    "midi_row": line_idx,
                    "source_midi": row["__midi_rel__"],
                    "source_dataset": "groove_midi",
                    "n_events": int(events.shape[0]),
                    "duration": duration_steps / float(MIDI_TIME_RES),
                    "duration_steps": duration_steps,
                }
            )
            line_idx += 1

    build_text_line_index(events_txt, idx_npy, overwrite=True)
    write_split_manifests(output_dir, manifest_rows)
    print(
        "Groove MIDI split counts:",
        {name: len(rows) for name, rows in manifest_rows.items()},
    )
    print(f"Created Groove MIDI cache at {cache_dir} and manifests at {output_dir}")


if __name__ == "__main__":
    args = argbind.parse_args()
    with argbind.scope(args):
        main()
