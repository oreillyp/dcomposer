"""Cache split MIDI scores and check their drum classes against sample manifests."""
import csv
import os
import sys
import tempfile
from pathlib import Path

import argbind

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from dcomposer.constants import (
    COARSE_MIDI_NOTE_TO_COARSE_LABEL,
    FINE_MIDI_NOTE_TO_COARSE_MIDI_NOTE,
)
from dcomposer.midi import (
    build_text_line_index,
    extract_drum_note_events,
    find_midi_files,
)


@argbind.bind(without_prefix=True)
def run(input_dir: str = "", output_dir: str = "manifests/training"):
    """Input contains train/ and val/ MIDI folders; output already contains WAV CSVs."""
    if not input_dir:
        raise ValueError("Set input_dir to your split MIDI directory")
    root, output = Path(input_dir).resolve(), Path(output_dir).resolve()
    targets = [
        output / f"midi_{split}{suffix}"
        for split in ("train", "val")
        for suffix in (".txt", ".idx.npy", ".csv")
    ]
    if any(p.exists() for p in targets):
        raise FileExistsError("Refusing to overwrite existing MIDI manifests/cache")
    all_rows = {}
    for split in ("train", "val"):
        with (output / f"oneshots_{split}.csv").open(newline="") as stream:
            samples = list(csv.DictReader(stream))
        if not samples:
            raise ValueError(f"Empty {split} one-shot manifest")
        labels = set()
        for sample in samples:
            path = Path(sample["oneshot"])
            label = sample["coarse_label"]
            if not path.is_absolute() or not path.is_file():
                raise ValueError(f"Use absolute paths to existing audio files: {path}")
            if label not in COARSE_MIDI_NOTE_TO_COARSE_LABEL.values():
                raise ValueError(f"Unknown coarse_label: {label}")
            labels.add(label)
        files = find_midi_files(root / split)
        if not files:
            raise ValueError(f"No MIDI files under {root / split}")
        rows = []
        for path in files:
            events = extract_drum_note_events(path)
            if not len(events):
                raise ValueError(f"No recognized channel-10 drum notes: {path}")
            required = {
                COARSE_MIDI_NOTE_TO_COARSE_LABEL[
                    FINE_MIDI_NOTE_TO_COARSE_MIDI_NOTE[int(note)]
                ]
                for note in events[:, 1]
            }
            if required - labels:
                raise ValueError(
                    f"Missing {split} samples for {sorted(required - labels)}: {path}"
                )
            rows.append((path, events))
        all_rows[split] = rows
    # Preflight both splits before writing; publish only complete event caches.
    with tempfile.TemporaryDirectory(dir=output) as tmp:
        tmp = Path(tmp)
        for split, rows in all_rows.items():
            text, index, manifest = (
                tmp / f"midi_{split}{suffix}" for suffix in (".txt", ".idx.npy", ".csv")
            )
            with text.open("w") as stream, manifest.open("w", newline="") as csv_stream:
                writer = csv.DictWriter(
                    csv_stream,
                    fieldnames=[
                        "midi_txt",
                        "midi_idx_npy",
                        "midi_row",
                        "source_midi",
                        "source_dataset",
                        "n_events",
                        "duration_steps",
                    ],
                )
                writer.writeheader()
                for i, (path, events) in enumerate(rows):
                    stream.write(" ".join(map(str, events.reshape(-1).tolist())) + "\n")
                    writer.writerow(
                        {
                            "midi_txt": str(output / text.name),
                            "midi_idx_npy": str(output / index.name),
                            "midi_row": i,
                            "source_midi": str(path),
                            "source_dataset": "local",
                            "n_events": len(events),
                            "duration_steps": int(events[-1, 0]) + 1,
                        }
                    )
            build_text_line_index(text, index)
            print(f"Prepared {split}: {len(rows)} MIDI files", flush=True)
        for target in targets:
            os.link(tmp / target.name, target)


if __name__ == "__main__":
    with argbind.scope(argbind.parse_args()):
        run()
