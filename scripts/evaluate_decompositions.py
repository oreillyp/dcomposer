"""Score preserved drum decompositions without loading or running any models."""
import csv
import hashlib
import importlib.metadata
import json
import math
import sys
from collections import Counter
from pathlib import Path

import argbind
import mido
import soundfile as sf
import torch
from audiotools import AudioSignal

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from dcomposer.constants import COARSE_MIDI_NOTE_TO_COARSE_LABEL
from dcomposer.constants import FINE_MIDI_NOTE_TO_COARSE_MIDI_NOTE
from dcomposer.constants import RAW_MIDI_NOTE_TO_FINE_MIDI_NOTE
from scripts.evaluation import WINDOWS, load_audio, spectral_scores

CLASSES = ("kick", "snare", "tom", "hat", "cymbal")
TRANSCRIPTION_DATASETS = ("e_gmd", "mdb_drums", "synthetic")
ROUNDTRIP_DATASETS = (*TRANSCRIPTION_DATASETS, "fsl10k")
EXPECTED_COUNTS = {"e_gmd": 1000, "mdb_drums": 1000, "synthetic": 2000, "fsl10k": 500}
METHODS = {
    "coarse": "final_normm16_p095_fullamp_ae5",
    "fine": "final_normm16_p095_fullamp_ae5_fine",
    "idm": "idm",
    "adtof": "adtof",
    "dose": "dose",
}
LABELS = {
    "KD": "kick",
    "BD": "kick",
    "SD": "snare",
    "TT": "tom",
    "HH": "hat",
    "CY": "cymbal",
    "hihat": "hat",
    "rim": "snare",
    "clap": "snare",
}
IGNORED = {"OT", "tambourine", "cowbell", "shaker", "perc", "other"}


def label_class(label):
    label = LABELS.get(label, label)
    if label in CLASSES:
        return label
    if label in IGNORED:
        return None
    raise ValueError(f"Unknown drum label: {label!r}")


def note_class(note):
    fine = RAW_MIDI_NOTE_TO_FINE_MIDI_NOTE.get(note)
    if fine is None:
        raise ValueError(f"Unsupported reference MIDI pitch: {note}")
    coarse = FINE_MIDI_NOTE_TO_COARSE_MIDI_NOTE[fine]
    return label_class(COARSE_MIDI_NOTE_TO_COARSE_LABEL[coarse])


def transcript(path):
    groups = {c: [] for c in CLASSES}
    excluded = 0
    if path.suffix == ".json":
        payload = json.loads(path.read_text())
        raw = payload.get("coarse", payload.get("5class"))
        if not isinstance(raw, dict):
            raise ValueError(f"Missing coarse/5class transcript in {path}")
        mapped = {label_class(k) for k in raw}
        if not set(CLASSES) <= mapped or any(
            not isinstance(v, list) for v in raw.values()
        ):
            raise ValueError(f"Incomplete or malformed transcript groups in {path}")
        events = [
            (label_class(k), float(t)) for k, values in raw.items() for t in values
        ]
    elif path.suffix == ".mid":
        events, time = [], 0.0
        for msg in mido.MidiFile(path):
            time += msg.time
            if msg.type == "note_on" and msg.velocity > 0:
                events.append((note_class(msg.note), time))
    else:
        events = [
            (label_class(label), float(time))
            for time, label in (
                line.split() for line in path.read_text().splitlines() if line.strip()
            )
        ]
    for label, time in events:
        if not math.isfinite(time) or time < 0:
            raise ValueError(f"Invalid onset {time} in {path}")
        if label is None:
            excluded += 1
        else:
            groups[label].append(time)
    return {k: sorted(v) for k, v in groups.items()}, excluded


def event_counts(ref, pred, tolerance=0.05):
    # Earliest feasible one-to-one matching maximizes matches on sorted times.
    i = j = tp = 0
    while i < len(ref) and j < len(pred):
        delta = pred[j] - ref[i]
        if abs(delta) <= tolerance + 1e-10:
            tp += 1
            i += 1
            j += 1
        elif delta < -tolerance:
            j += 1
        else:
            i += 1
    return tp, len(pred) - tp, len(ref) - tp


def f1(tp, fp, fn):
    return 2 * tp / (2 * tp + fp + fn) if tp or fp or fn else 0.0


def select(paths, seed, key):
    paths = sorted(paths)
    if not paths:
        raise ValueError(f"No candidates for {key}")
    digest = hashlib.sha256(f"{seed}:{key}".encode()).digest()
    return paths[int.from_bytes(digest, "big") % len(paths)]


def shot_groups(directory):
    groups = {c: [] for c in CLASSES}
    for path in sorted(directory.glob("*.wav")):
        if "__coarse" in path.stem:
            label = path.stem.split("__coarse", 1)[1].split("_", 1)[1]
        else:
            label = path.stem.split("_")[1]
        label = label_class(label)
        if label:
            groups[label].append(path)
    return groups


def make_plan(inputs, predictions, methods, tasks, seed, limit):
    plan = {task: [] for task in tasks}
    selected = set(ROUNDTRIP_DATASETS if "roundtrip" in tasks else ())
    if "transcription" in tasks:
        selected.update(TRANSCRIPTION_DATASETS)
    if "oneshot" in tasks:
        selected.add("synthetic")
    audio = {}
    for dataset in sorted(selected):
        files = sorted((inputs / dataset / "audio").glob("*.wav"))
        if not files:
            raise FileNotFoundError(f"No inputs for {dataset} in {inputs}")
        if not limit and len(files) != EXPECTED_COUNTS[dataset]:
            raise ValueError(
                f"Expected {EXPECTED_COUNTS[dataset]} {dataset} inputs, found {len(files)}; use --limit for smoke tests only"
            )
        audio[dataset] = files[:limit] if limit else files
    for method in methods:
        run = predictions / METHODS[method]
        if not run.is_dir():
            raise FileNotFoundError(run)
        if "transcription" in tasks and method != "dose":
            for dataset in TRANSCRIPTION_DATASETS:
                for ref_audio in audio[dataset]:
                    stem = ref_audio.stem
                    suffix = {
                        "mdb_drums": ".txt",
                        "e_gmd": ".mid",
                        "synthetic": ".json",
                    }[dataset]
                    ref = inputs / dataset / "transcript" / f"{stem}{suffix}"
                    pred = (
                        run / "transcripts" / dataset / "audio" / f"{stem}.json"
                        if method in ("coarse", "fine")
                        else run / "transcription/preds" / dataset / f"{stem}.json"
                    )
                    if not ref.is_file() or not pred.is_file():
                        raise FileNotFoundError(
                            f"Missing transcript pair: {ref}, {pred}"
                        )
                    plan["transcription"].append(
                        dict(
                            method=method,
                            dataset=dataset,
                            mixture_id=stem,
                            reference=ref,
                            prediction=pred,
                        )
                    )
        if "roundtrip" in tasks and method not in ("dose", "adtof"):
            for dataset in ROUNDTRIP_DATASETS:
                for ref in audio[dataset]:
                    pred = (
                        run
                        / "roundtrip"
                        / dataset
                        / (ref.name if method == "idm" else f"audio/{ref.name}")
                    )
                    if not pred.is_file():
                        raise FileNotFoundError(pred)
                    info = sf.info(ref)
                    if (
                        abs(sf.info(pred).duration - info.duration)
                        > 1 / info.samplerate
                    ):
                        raise ValueError(f"Round-trip duration mismatch: {pred}")
                    plan["roundtrip"].append(
                        dict(
                            method=method,
                            dataset=dataset,
                            mixture_id=ref.stem,
                            reference=ref,
                            prediction=pred,
                            samples=round(info.duration * 44100),
                            missing=False,
                        )
                    )
        if "oneshot" in tasks and method != "adtof":
            root = (
                run
                / ("oneshots" if method in ("coarse", "fine") else "oneshot")
                / "synthetic"
            )
            classes = ("kick", "snare", "hat") if method == "dose" else CLASSES
            if not root.is_dir():
                raise FileNotFoundError(
                    f"Missing one-shot root (not an extraction miss): {root}"
                )
            if method in ("dose", "idm"):
                for label in classes:
                    folder = root / ("hihat" if label == "hat" else label)
                    if not folder.is_dir():
                        raise FileNotFoundError(
                            f"Missing baseline class directory: {folder}"
                        )
            for ref_audio in audio["synthetic"]:
                stem = ref_audio.stem
                gt_dir = inputs / "synthetic/oneshot" / stem
                if not gt_dir.is_dir():
                    raise FileNotFoundError(gt_dir)
                refs = shot_groups(gt_dir)
                if method in ("coarse", "fine"):
                    folder = root / "audio" / stem
                    if not folder.is_dir():
                        raise FileNotFoundError(
                            f"Missing exported mixture directory: {folder}"
                        )
                    preds = shot_groups(folder)
                else:
                    preds = {
                        c: [root / ("hihat" if c == "hat" else c) / f"{stem}.wav"]
                        for c in classes
                    }
                for label in classes:
                    if not refs[label]:
                        continue
                    ref = select(refs[label], seed, f"reference/{stem}/{label}")
                    candidates = [p for p in preds[label] if p.is_file()]
                    pred = (
                        select(candidates, seed, f"prediction/{method}/{stem}/{label}")
                        if candidates
                        else None
                    )
                    plan["oneshot"].append(
                        dict(
                            method=method,
                            dataset="synthetic",
                            mixture_id=stem,
                            drum_class=label,
                            reference=ref,
                            prediction=pred,
                            samples=3 * 44100,
                            missing=pred is None,
                            reference_variants=len(
                                {p.stem.split("_")[1] for p in refs[label]}
                            ),
                        )
                    )
    return plan


def evaluate_transcripts(rows):
    results = []
    for idx, row in enumerate(rows):
        ref, excluded_ref = transcript(row["reference"])
        pred, excluded_pred = transcript(row["prediction"])
        for label in CLASSES:
            tp, fp, fn = event_counts(ref[label], pred[label])
            results.append(
                {
                    **row,
                    "drum_class": label,
                    "tp": tp,
                    "fp": fp,
                    "fn": fn,
                    "f1": f1(tp, fp, fn),
                    "reference_present": bool(ref[label]),
                    "excluded_reference_events": excluded_ref,
                    "excluded_prediction_events": excluded_pred,
                }
            )
        if (idx + 1) % 100 == 0 or idx + 1 == len(rows):
            print(f"Transcription: {idx + 1}/{len(rows)}", flush=True)
    return results


def evaluate_audio(rows, task, batch_size, device, writer, stream):
    results = []
    for start in range(0, len(rows), batch_size):
        batch = rows[start : start + batch_size]
        # All paper inputs have the same duration. Never pad unequal-length pairs into a shared reduction.
        if len({r["samples"] for r in batch}) != 1:
            raise ValueError("Unequal durations in metric batch")
        refs = AudioSignal.batch(
            [
                load_audio(r["reference"], r["samples"], task == "roundtrip")
                for r in batch
            ]
        ).to(device)
        preds = AudioSignal.batch(
            [load_audio(r["prediction"], r["samples"]) for r in batch]
        ).to(device)
        scores = spectral_scores(refs, preds)
        for i, row in enumerate(batch):
            result = {**row, **{key: values[i] for key, values in scores.items()}}
            writer.writerow(result)
            results.append(result)
        stream.flush()
        print(f"{task}: {len(results)}/{len(rows)}", flush=True)
    return results


def summarize(rows, task):
    groups = {}
    for row in rows:
        key = "/".join(
            str(row[k])
            for k in ("method", "dataset")
            + (() if task == "roundtrip" else ("drum_class",))
        )
        groups.setdefault(key, []).append(row)
    result = {}
    for key, group in groups.items():
        entry = {"count": len(group)}
        if task == "transcription":
            tp, fp, fn = (sum(r[k] for r in group) for k in ("tp", "fp", "fn"))
            entry.update(
                tp=tp,
                fp=fp,
                fn=fn,
                pooled_f1=f1(tp, fp, fn),
                mean_clip_f1=sum(r["f1"] for r in group) / len(group),
            )
        else:
            entry.update(
                {
                    k: sum(r[k] for r in group) / len(group)
                    for k in ("mse_raw", "mse_rms_matched", "l1_raw", "l1_rms_matched")
                }
            )
            entry["missing_predictions"] = sum(r["missing"] for r in group)
            if task == "oneshot":
                entry["mixed_reference_classes"] = sum(
                    r["reference_variants"] > 1 for r in group
                )
        result[key] = entry
    return result


def table(summary, methods, metric):
    columns = [
        "Model",
        "F1 E-GMD",
        "F1 MDB",
        "F1 Synthetic",
        "One-shot Synthetic",
        *[f"MSL {d}" for d in ROUNDTRIP_DATASETS],
    ]
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for method in methods:
        values = [method]
        for dataset in TRANSCRIPTION_DATASETS:
            vals = [
                summary.get("transcription", {})
                .get(f"{method}/{dataset}/{c}", {})
                .get("mean_clip_f1")
                for c in CLASSES
            ]
            values.append("/".join("-" if v is None else f"{v:.3f}" for v in vals))
        vals = [
            summary.get("oneshot", {})
            .get(f"{method}/synthetic/{c}", {})
            .get(f"{metric}_raw")
            for c in CLASSES
        ]
        values.append("/".join("-" if v is None else f"{v:.3f}" for v in vals))
        for dataset in ROUNDTRIP_DATASETS:
            value = (
                summary.get("roundtrip", {})
                .get(f"{method}/{dataset}", {})
                .get(f"{metric}_raw")
            )
            values.append("-" if value is None else f"{value:.3f}")
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines) + "\n"


def latex_table(summary, methods, metric):
    names = {
        "coarse": "DComposer",
        "fine": "DComposer (fine)",
        "idm": "IDM",
        "adtof": "ADTOF",
        "dose": "DOSE",
    }
    rows = table(summary, methods, metric).splitlines()[2:]
    lines = [
        r"\begin{table*}[t]",
        r"\centering\scriptsize",
        r"\begin{tabular}{lcccccccc}",
        r"\toprule",
        r" & \multicolumn{3}{c}{Transcription F1 $\uparrow$} & Extraction MSL $\downarrow$ & \multicolumn{4}{c}{Round-trip MSL $\downarrow$} \\",
        r"\cmidrule(lr){2-4}\cmidrule(lr){6-9}",
        r"Model & E-GMD & MDB & Synthetic & Synthetic & E-GMD & MDB & Synthetic & FSL \\",
        r"\midrule",
    ]
    for row in rows:
        cells = [cell.strip() for cell in row.strip("|").split("|")]
        cells[0] = names[cells[0]]
        lines.append(" & ".join(cells) + r" \\")
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            "\\caption{Corrected evaluation. Classwise entries follow KD/SD/TT/HH/CY. F1 is mean per-clip F1 at 50 ms. Extraction scores only reference-present classes, penalizing genuine missing predictions as silence. MSL uses "
            + ("MSE" if metric == "mse" else "L1")
            + " on STFT magnitudes, summed over six resolutions, without loudness matching.}",
            r"\label{tab:decomposition-" + metric + "}",
            r"\end{table*}",
        ]
    )
    return "\n".join(lines) + "\n"


@argbind.bind(without_prefix=True)
def run(
    inputs: str = "",
    predictions: str = "",
    output: str = "",
    methods: str = "coarse,idm,adtof,dose",
    tasks: str = "transcription,oneshot,roundtrip",
    seed: int = 0,
    device: str = "cpu",
    batch_size: int = 16,
    limit: int = 0,
):
    """Use --limit only for smoke tests; full scoring uses every preserved paper input."""
    if not inputs or not predictions or not output:
        raise ValueError("Pass --inputs, --predictions, and --output")
    methods, tasks = methods.split(","), tasks.split(",")
    if len(set(methods)) != len(methods) or not set(methods) <= set(METHODS):
        raise ValueError(f"Unknown or repeated methods: {methods}")
    if len(set(tasks)) != len(tasks) or not set(tasks) <= {
        "transcription",
        "oneshot",
        "roundtrip",
    }:
        raise ValueError(f"Unknown or repeated tasks: {tasks}")
    if batch_size < 1 or limit < 0:
        raise ValueError("batch_size must be positive and limit nonnegative")
    if device == "cpu":
        torch.set_num_threads(min(torch.get_num_threads(), 4))
    output = Path(output).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite evaluation: {output}")
    print("Preflighting input and prediction paths...", flush=True)
    plan = make_plan(
        Path(inputs).expanduser().resolve(),
        Path(predictions).expanduser().resolve(),
        methods,
        tasks,
        seed,
        limit,
    )
    if not any(plan.values()):
        raise ValueError("No supported method/task pairs selected")
    output.mkdir(parents=True)
    metadata = dict(
        inputs=str(Path(inputs).resolve()),
        predictions=str(Path(predictions).resolve()),
        methods=methods,
        tasks=tasks,
        seed=seed,
        device=device,
        batch_size=batch_size,
        limit=limit,
        partial=bool(limit),
        complete=False,
        classes=CLASSES,
        tolerance_ms=50,
        metric_sample_rate=44100,
        expected_counts=EXPECTED_COUNTS,
        planned_pairs={k: len(v) for k, v in plan.items()},
        windows=WINDOWS,
        magnitude_loss="mse",
        log_weight=0,
        normalization="reference -16 dB RMS with peak protection, before mono conversion",
        missing_policy="silence for genuine missing predictions of reference-present classes; invalid roots fail",
        transcription_aggregation="mean clip F1, empty/empty=0; pooled F1 also reported",
        versions={
            p: importlib.metadata.version(p)
            for p in ("torch", "descript-audiotools", "mido")
        },
    )
    (output / "config.json").write_text(json.dumps(metadata, indent=2))
    summary = {}
    for task, rows in plan.items():
        if not rows:
            continue
        path = output / f"{task}.csv"
        print(f"Starting {task}: {len(rows)} pairs", flush=True)
        if task == "transcription":
            results = evaluate_transcripts(rows)
            with path.open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(results[0]))
                writer.writeheader()
                writer.writerows(results)
        else:
            with path.open("w", newline="") as stream:
                writer = csv.DictWriter(
                    stream,
                    fieldnames=[
                        *rows[0],
                        "mse_raw",
                        "mse_rms_matched",
                        "l1_raw",
                        "l1_rms_matched",
                    ],
                )
                writer.writeheader()
                results = evaluate_audio(rows, task, batch_size, device, writer, stream)
        summary[task] = summarize(results, task)
        (output / "summary.json").write_text(
            json.dumps(summary, indent=2, allow_nan=False)
        )
    for metric in ("mse", "l1"):
        prefix = "SMOKE TEST ONLY: limited input subset.\n\n" if limit else ""
        (output / f"table3_{metric}.md").write_text(
            prefix + table(summary, methods, metric)
        )
        (output / f"table3_{metric}.tex").write_text(
            ("% SMOKE TEST ONLY\n" if limit else "")
            + latex_table(summary, methods, metric)
        )
    metadata["complete"] = True
    (output / "config.json").write_text(json.dumps(metadata, indent=2))
    print(f"Wrote evaluation to {output}", flush=True)


if __name__ == "__main__":
    args = argbind.parse_args()
    with argbind.scope(args):
        run()
