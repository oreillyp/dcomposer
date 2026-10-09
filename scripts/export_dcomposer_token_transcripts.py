#!/usr/bin/env python3
import argparse
import json
import struct
import sys
import time
from pathlib import Path
from typing import Dict
from typing import List
from typing import Optional

import argbind
import torch

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))

from scripts import train_dcomposer
from dcomposer.constants import COARSE_MIDI_NOTE_TO_COARSE_LABEL
from dcomposer.constants import FINE_MIDI_NOTE_TO_COARSE_MIDI_NOTE
from dcomposer.constants import FINE_MIDI_NOTE_TO_FINE_LABEL
from dcomposer.util import load_config
from dcomposer.retrieval import parse_events
from dcomposer.windows import window_events, event_row
from dcomposer.constants import ACOUSTIC_TOK_OFFSET


def log(message: str):
    print(message, file=sys.stderr, flush=True)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input_dir", type=str, required=True)
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--dcomposer_run_dir", type=str, required=True)
    p.add_argument("--max_files", type=int, default=None)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--note_duration_s", type=float, default=0.03)
    p.add_argument("--midi_tempo_bpm", type=float, default=120.0)
    p.add_argument("--ticks_per_beat", type=int, default=480)
    return p.parse_args()


def discover_token_files(root: Path, max_files: Optional[int] = None) -> List[Path]:
    files = [
        p
        for p in sorted(root.rglob("*.pt"))
        if p.is_file() and p.name not in {"_run_metadata.pt", "_run_metadata.json"}
    ]
    if max_files is not None:
        files = files[: int(max_files)]
    return files


def load_token_payload(path: Path) -> Dict:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or not (
        "tokens" in payload or "windows" in payload
    ):
        raise ValueError(f"Invalid token payload in {path}")
    return payload


def trim_tokens(payload: Dict) -> torch.Tensor:
    tokens = payload["tokens"]
    if not isinstance(tokens, torch.Tensor):
        tokens = torch.as_tensor(tokens)
    length = int(payload.get("length", tokens.shape[0]))
    return tokens[:length].to(dtype=torch.long, device="cpu")


def get_relative_output_key(token_path: Path, input_dir: Path, payload: Dict) -> Path:
    rel_str = payload.get("relative_path", None)
    if rel_str not in [None, ""]:
        return Path(rel_str)
    return token_path.relative_to(input_dir)


def fine_to_coarse(fine_note: int) -> int:
    return int(FINE_MIDI_NOTE_TO_COARSE_MIDI_NOTE.get(int(fine_note), int(fine_note)))


def empty_coarse_dict() -> Dict[str, List[float]]:
    labels = list(dict.fromkeys(COARSE_MIDI_NOTE_TO_COARSE_LABEL.values()))
    return {label: [] for label in labels}


def empty_fine_dict() -> Dict[str, List[float]]:
    labels = list(dict.fromkeys(FINE_MIDI_NOTE_TO_FINE_LABEL.values()))
    return {label: [] for label in labels}


def build_transcript_payload(
    *,
    token_path: Path,
    rel_key: Path,
    payload: Dict,
    parsed_row: Dict,
    hierarchy_level: str,
) -> Dict:
    coarse = empty_coarse_dict()
    fine = empty_fine_dict()
    events = []
    for event_idx, (fine_note, onset_sec) in enumerate(
        zip(parsed_row["notes"], parsed_row["onsets_sec"])
    ):
        fine_note = int(fine_note)
        coarse_note = fine_to_coarse(fine_note)
        fine_label = FINE_MIDI_NOTE_TO_FINE_LABEL.get(fine_note, "unknown")
        coarse_label = COARSE_MIDI_NOTE_TO_COARSE_LABEL.get(coarse_note, "unknown")
        onset_sec = float(onset_sec)
        if coarse_label in coarse:
            coarse[coarse_label].append(onset_sec)
        if fine_label in fine:
            fine[fine_label].append(onset_sec)
        events.append(
            {
                "event_idx": int(event_idx),
                "onset_sec": onset_sec,
                "fine_note": fine_note,
                "fine_label": fine_label,
                "coarse_note": coarse_note,
                "coarse_label": coarse_label,
            }
        )
    for key in coarse:
        coarse[key].sort()
    for key in fine:
        fine[key].sort()
    events.sort(key=lambda x: (x["onset_sec"], x["event_idx"]))
    return {
        "source_token_path": str(token_path),
        "source_relative_path": str(rel_key),
        "source_audio_path": payload.get("source_path", None),
        "hierarchy_level": hierarchy_level,
        "fine_labels_are_canonicalized_from_predicted_notes": bool(
            hierarchy_level == "coarse"
        ),
        "coarse": coarse,
        "fine": fine,
        "events": events,
    }


def encode_var_len(value: int) -> bytes:
    if value < 0:
        raise ValueError(f"VLQ value must be >= 0, got {value}")
    out = [value & 0x7F]
    value >>= 7
    while value:
        out.append(0x80 | (value & 0x7F))
        value >>= 7
    return bytes(reversed(out))


def make_midi_bytes(
    *,
    parsed_row: Dict,
    note_duration_s: float,
    tempo_bpm: float,
    ticks_per_beat: int,
) -> bytes:
    if note_duration_s <= 0:
        raise ValueError(f"`note_duration_s` must be positive; got {note_duration_s}")
    if tempo_bpm <= 0:
        raise ValueError(f"`midi_tempo_bpm` must be positive; got {tempo_bpm}")
    if ticks_per_beat <= 0:
        raise ValueError(f"`ticks_per_beat` must be positive; got {ticks_per_beat}")

    tempo_us_per_beat = int(round(60_000_000.0 / float(tempo_bpm)))
    ticks_per_second = float(ticks_per_beat) * (1_000_000.0 / float(tempo_us_per_beat))
    note_dur_ticks = max(1, int(round(float(note_duration_s) * ticks_per_second)))

    events = []
    for fine_note, onset_sec in zip(parsed_row["notes"], parsed_row["onsets_sec"]):
        note = int(fine_note)
        onset_tick = max(0, int(round(float(onset_sec) * ticks_per_second)))
        off_tick = onset_tick + note_dur_ticks
        velocity = 100
        events.append((onset_tick, 0, bytes([0x99, note, velocity])))
        events.append((off_tick, 1, bytes([0x89, note, 0])))
    events.sort(key=lambda x: (x[0], x[1]))

    track = bytearray()
    track.extend(encode_var_len(0))
    track.extend(b"\xFF\x51\x03")
    track.extend(struct.pack(">I", tempo_us_per_beat)[1:])
    track.extend(encode_var_len(0))
    track.extend(b"\xC9\x00")

    prev_tick = 0
    for abs_tick, _priority, msg in events:
        delta = int(abs_tick) - int(prev_tick)
        track.extend(encode_var_len(delta))
        track.extend(msg)
        prev_tick = int(abs_tick)

    track.extend(encode_var_len(0))
    track.extend(b"\xFF\x2F\x00")

    header = bytearray()
    header.extend(b"MThd")
    header.extend(struct.pack(">IHHH", 6, 0, 1, int(ticks_per_beat)))
    chunk = bytearray()
    chunk.extend(b"MTrk")
    chunk.extend(struct.pack(">I", len(track)))
    chunk.extend(track)
    return bytes(header + chunk)


def write_outputs(
    *,
    out_json_path: Path,
    out_mid_path: Path,
    transcript_payload: Dict,
    midi_bytes: bytes,
):
    out_json_path.parent.mkdir(parents=True, exist_ok=True)
    out_mid_path.parent.mkdir(parents=True, exist_ok=True)
    out_json_path.write_text(json.dumps(transcript_payload, indent=2))
    out_mid_path.write_bytes(midi_bytes)


def main():
    cli = parse_args()
    input_dir = Path(cli.input_dir).expanduser().resolve()
    output_dir = Path(cli.output_dir).expanduser().resolve()
    dcomposer_run_dir = Path(cli.dcomposer_run_dir).expanduser().resolve()
    if not input_dir.is_dir():
        raise NotADirectoryError(f"Input directory not found: {input_dir}")
    if output_dir.exists() and any(output_dir.iterdir()) and not cli.overwrite:
        raise FileExistsError(
            f"Refusing to mix transcripts into existing outputs: {output_dir}"
        )
    if not dcomposer_run_dir.exists():
        raise FileNotFoundError(f"DComposer model not found: {dcomposer_run_dir}")

    run_args = load_config(train_dcomposer._config_path(dcomposer_run_dir))
    hierarchy_level = str(run_args.get("OneShotMidiDataset.hierarchy_level", "coarse"))
    if hierarchy_level not in {"coarse", "fine"}:
        raise ValueError(f"Unexpected hierarchy level: {hierarchy_level}")
    n_acoustic_tokens = int(run_args["DComposer.n_acoustic_tokens"])

    files = discover_token_files(input_dir, max_files=cli.max_files)
    if not files:
        raise FileNotFoundError(f"No token files found in {input_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "dcomposer_run_dir": str(dcomposer_run_dir),
        "dcomposer_config_path": str(train_dcomposer._config_path(dcomposer_run_dir)),
        "hierarchy_level": hierarchy_level,
        "n_acoustic_tokens": n_acoustic_tokens,
        "note_duration_s": float(cli.note_duration_s),
        "midi_tempo_bpm": float(cli.midi_tempo_bpm),
        "ticks_per_beat": int(cli.ticks_per_beat),
        "n_files": int(len(files)),
    }
    (output_dir / "_transcript_metadata.json").write_text(
        json.dumps(metadata, indent=2)
    )

    log(f"Exporting transcripts for {len(files)} token files -> {output_dir}")
    t_start = time.perf_counter()
    processed = 0
    total_events = 0
    for token_path in files:
        payload = load_token_payload(token_path)
        rel_key = get_relative_output_key(token_path, input_dir, payload)
        n_vocab = int(run_args["DComposer.n_vocab"]) - ACOUSTIC_TOK_OFFSET
        if "windows" in payload:
            parsed_row = event_row(window_events(payload, n_acoustic_tokens, n_vocab))
        else:
            tokens = trim_tokens(payload).unsqueeze(0)
            parse_events(tokens[0], n_acoustic_tokens, n_vocab)
            parsed_row = train_dcomposer.parse_dcomposer_tokens(
                tokens,
                n_acoustic_tokens=n_acoustic_tokens,
            )[0]
        transcript_payload = build_transcript_payload(
            token_path=token_path,
            rel_key=rel_key,
            payload=payload,
            parsed_row=parsed_row,
            hierarchy_level=hierarchy_level,
        )
        midi_bytes = make_midi_bytes(
            parsed_row=parsed_row,
            note_duration_s=float(cli.note_duration_s),
            tempo_bpm=float(cli.midi_tempo_bpm),
            ticks_per_beat=int(cli.ticks_per_beat),
        )
        out_json = (output_dir / rel_key).with_suffix(".json")
        out_mid = (output_dir / rel_key).with_suffix(".mid")
        write_outputs(
            out_json_path=out_json,
            out_mid_path=out_mid,
            transcript_payload=transcript_payload,
            midi_bytes=midi_bytes,
        )
        processed += 1
        total_events += len(parsed_row["notes"])
        if processed % 50 == 0 or processed == len(files):
            elapsed = time.perf_counter() - t_start
            files_per_second = processed / max(elapsed, 1e-8)
            remaining = len(files) - processed
            eta_seconds = remaining / max(files_per_second, 1e-8)
            log(
                f"Processed {processed}/{len(files)} token files "
                f"(eta={eta_seconds:.1f}s)"
            )

    total_seconds = time.perf_counter() - t_start
    log("Done.")
    log("Timing summary: " f"total={total_seconds:.2f}s")
    log(
        "Throughput summary: "
        f"files={processed}, "
        f"events={total_events}, "
        f"files_per_second={processed / max(total_seconds, 1e-8):.3f}, "
        f"seconds_per_file={total_seconds / max(processed, 1):.3f}"
    )


if __name__ == "__main__":
    main()
