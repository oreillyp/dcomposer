"""Rescore preserved codec reconstructions; no inference or input normalization."""
import csv
import importlib.metadata
import json
import sys
from pathlib import Path

import argbind
import torch
from audiotools import AudioSignal

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.evaluation import WINDOWS, load_audio, spectral_scores

CODECS = ("dac", "codicodec", "autoencoder", "melodyflow")
DETAILS = (
    ("DAC", "44.1k", "No", "7.7", "2322"),
    ("CoDiCodec", "44.1k", "Yes", "2.3", "512"),
    ("DOC", "48k", "Yes", "0.1", "32"),
    ("MelodyFlow", "48k", "Yes", "--", "--"),
)


def pairs(root):
    files = sorted((root / "inputs").glob("*.wav"))
    expected = {f"{i:05d}.wav" for i in range(2000)}
    if {p.name for p in files} != expected:
        raise ValueError("Expected exactly the 2,000 preserved codec inputs")
    for codec in CODECS:
        actual = {p.name for p in (root / "reconstructions" / codec).glob("*.wav")}
        if actual != expected:
            raise ValueError(f"Incomplete or unexpected reconstruction set: {codec}")
    return files


def table(summary, metric):
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\begin{tabular}{lccrrr}",
        r"\toprule",
        r"Codec & SR & Stereo & kbps & Tokens & MSL $\downarrow$ \\",
        r"\midrule",
    ]
    for codec, details in zip(CODECS, DETAILS):
        lines.append(
            " & ".join((*details, f"{summary[codec][metric + '_raw']:.3f}")) + r" \\"
        )
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            "\\caption{Codec reconstruction on 2,000 preserved one-shots. MSL uses "
            + ("MSE" if metric == "mse" else "L1")
            + " on STFT magnitudes, summed over six resolutions; 44.1 kHz mono scoring, without loudness matching.}",
            r"\label{tab:codecs-" + metric + "}",
            r"\end{table}",
        ]
    )
    return "\n".join(lines) + "\n"


@argbind.bind(without_prefix=True)
def run(root: str = "", output: str = "", device: str = "cuda", batch_size: int = 32):
    if not root or not output or batch_size < 1:
        raise ValueError("Pass --root, --output, and a positive --batch_size")
    root, output = (
        Path(root).expanduser().resolve(),
        Path(output).expanduser().resolve(),
    )
    if output.exists():
        raise FileExistsError(output)
    print("Preflighting all 8,000 codec pairs...", flush=True)
    files = pairs(root)
    output.mkdir(parents=True)
    metadata = dict(
        root=str(root),
        device=device,
        batch_size=batch_size,
        complete=False,
        samples=2000,
        duration=3.0,
        metric_sample_rate=44100,
        windows=WINDOWS,
        log_weight=0,
        normalization="none; mono conversion only",
        versions={
            p: importlib.metadata.version(p) for p in ("torch", "descript-audiotools")
        },
    )
    (output / "config.json").write_text(json.dumps(metadata, indent=2))
    summary = {}
    with (output / "per_sample_metrics.csv").open("w", newline="") as stream:
        keys = ("mse_raw", "mse_rms_matched", "l1_raw", "l1_rms_matched")
        writer = csv.DictWriter(stream, fieldnames=["sample_idx", "codec", *keys])
        writer.writeheader()
        for codec in CODECS:
            totals = dict.fromkeys(keys, 0.0)
            for start in range(0, len(files), batch_size):
                batch = files[start : start + batch_size]
                ref = AudioSignal.batch([load_audio(p, 3 * 44100) for p in batch]).to(
                    device
                )
                pred = AudioSignal.batch(
                    [
                        load_audio(root / "reconstructions" / codec / p.name, 3 * 44100)
                        for p in batch
                    ]
                ).to(device)
                scores = spectral_scores(ref, pred)
                for i, path in enumerate(batch):
                    row = {k: scores[k][i] for k in keys}
                    writer.writerow(dict(sample_idx=int(path.stem), codec=codec, **row))
                    for k in keys:
                        totals[k] += row[k]
                stream.flush()
                print(f"{codec}: {start + len(batch)}/{len(files)}", flush=True)
            summary[codec] = dict(
                count=len(files), **{k: v / len(files) for k, v in totals.items()}
            )
            (output / "summary.json").write_text(
                json.dumps(summary, indent=2, allow_nan=False)
            )
    for metric in ("mse", "l1"):
        (output / f"table2_{metric}.tex").write_text(table(summary, metric))
    metadata["complete"] = True
    (output / "config.json").write_text(json.dumps(metadata, indent=2))
    print(f"Wrote codec evaluation to {output}", flush=True)


if __name__ == "__main__":
    with argbind.scope(argbind.parse_args()):
        run()
