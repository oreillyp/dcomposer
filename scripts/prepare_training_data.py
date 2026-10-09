"""Prepare samples and cached MIDI for the supplied training configurations."""
import csv
import shutil
import subprocess
import sys
from pathlib import Path

import argbind

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dcomposer.constants import (
    COARSE_MIDI_NOTE_TO_COARSE_LABEL,
    FINE_MIDI_NOTE_TO_COARSE_MIDI_NOTE,
)
from dcomposer.midi import DrumEventTextIndex


def prepare(script, *args):
    print(f"Preparing: {script}", flush=True)
    command = ["bash"] if script.endswith(".sh") else [sys.executable]
    subprocess.run(
        [*command, str(ROOT / "scripts" / script), *map(str, args)],
        cwd=ROOT,
        check=True,
    )


def select_scores(source, target, samples):
    """Keep whole scores whose drum classes are available in this sample split."""
    with samples.open(newline="") as stream:
        labels = {row["coarse_label"] for row in csv.DictReader(stream)}
    caches, rows, total = {}, [], 0
    try:
        with source.open(newline="") as stream:
            reader = csv.DictReader(stream)
            fields = reader.fieldnames
            for row in reader:
                total += 1
                key = (row["midi_txt"], row["midi_idx_npy"])
                if key not in caches:
                    caches[key] = DrumEventTextIndex(*key)
                events = caches[key].read_row(int(row["midi_row"]))
                required = {
                    COARSE_MIDI_NOTE_TO_COARSE_LABEL[
                        FINE_MIDI_NOTE_TO_COARSE_MIDI_NOTE[int(note)]
                    ]
                    for note in events[:, 1]
                }
                if len(events) and required <= labels:
                    rows.append(row)
    finally:
        for cache in caches.values():
            cache.close()
    if not rows:
        raise ValueError(f"No scores in {source} match the available sample classes")
    with target.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(
        f"{target.name}: kept {len(rows)}/{total} scores with available drums",
        flush=True,
    )


@argbind.bind(without_prefix=True)
def prepare_data(
    oneshots: str = "",
    midi: str = "",
    download: bool = False,
    output: str = "manifests/training",
):
    """Prepare samples, Lakh/local MIDI, GMD, and noise/reverb assets for training.

    Use --download for Organic/Lakh, or pass --oneshots and --midi for local data.
    Both modes pack samples and prepare Groove MIDI and noise/reverb assets.
    """
    if (download and (oneshots or midi)) or (not download and not (oneshots and midi)):
        raise ValueError("Use --download OR both --oneshots and --midi")
    output = Path(output).expanduser().resolve()
    if any(output.glob("*.csv")) or any(output.glob("midi_*")):
        raise FileExistsError(f"Refusing to overwrite prepared data in {output}")
    if not download:
        oneshots, midi = (
            Path(oneshots).expanduser().resolve(),
            Path(midi).expanduser().resolve(),
        )
        for folder in (oneshots, midi / "train", midi / "val"):
            if not folder.is_dir():
                raise FileNotFoundError(folder)
    output.mkdir(parents=True, exist_ok=True)
    if download:
        prepare("download/download_organic_one_shots.sh")
        prepare(
            "setup/create_organic_one_shots_manifests.py",
            "--label_col",
            "coarse_label",
            "--write_packed_audio",
        )
        prepare("download/download_lmd.sh")
        prepare("setup/create_lmd_manifests.py")
        for split in ("train", "val", "test"):
            shutil.copy2(
                ROOT / f"manifests/organic_one_shots/{split}.csv",
                output / f"oneshots_{split}.csv",
            )
        for split in ("train", "val"):
            select_scores(
                ROOT / f"manifests/lmd/{split}.csv",
                output / f"midi_{split}.csv",
                output / f"oneshots_{split}.csv",
            )
    else:
        prepare(
            "setup/create_one_shot_manifests.py",
            "--data_dir",
            oneshots,
            "--output_dir",
            output,
            "--write_packed_audio",
        )
        prepare(
            "setup/create_midi_manifests.py",
            "--input_dir",
            midi,
            "--output_dir",
            output,
        )
    prepare("download/download_groove_midi.sh")
    prepare("setup/create_groove_midi_manifests.py")
    for split in ("train", "val"):
        select_scores(
            ROOT / f"manifests/groove_midi/{split}.csv",
            output / f"groove_{split}.csv",
            output / f"oneshots_{split}.csv",
        )
    prepare("download/download_extra_augs.sh")
    prepare("setup/create_extra_aug_manifests.py")
    print(
        f"Ready: {output}. Use conf/doc.yml to train DOC.",
        flush=True,
    )


if __name__ == "__main__":
    with argbind.scope(argbind.parse_args()):
        prepare_data()
