#!/usr/bin/env python3
import argparse
import csv
import json
import math
import random
import shutil
import struct
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict
from typing import Iterable
from typing import List
from typing import Optional
from typing import Sequence
from typing import Tuple

import argbind
import numpy as np
import soundfile as sf
import torch
from audiotools import AudioSignal

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))

from scripts import train_dcomposer
from dcomposer.constants import COARSE_MIDI_NOTE_TO_COARSE_LABEL
from dcomposer.constants import FINE_MIDI_NOTE_TO_COARSE_MIDI_NOTE
from dcomposer.constants import FINE_MIDI_NOTE_TO_FINE_LABEL
from dcomposer.constants import MIDI_TIME_RES
from dcomposer.data.one_shot_midi_dataset import OneShotMidiDataset
from dcomposer.data.one_shot_midi_dataset import build_one_shot_midi_batch
from dcomposer.util import load_excerpt
from dcomposer.util import load_config

AUDIO_EXTS = {".wav", ".aif", ".aiff", ".flac", ".mp3", ".ogg", ".m4a"}


def log(message: str):
    print(message, file=sys.stderr, flush=True)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--output_dir", type=str, default="eval/inputs")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--salience_loudness_cutoff", type=float, default=-40.0)
    p.add_argument("--salience_num_tries", type=int, default=8)
    p.add_argument("--midi_tempo_bpm", type=float, default=120.0)
    p.add_argument("--midi_note_duration_s", type=float, default=0.03)
    p.add_argument("--ticks_per_beat", type=int, default=480)
    p.add_argument("--sample_rate", type=int, default=44100)
    p.add_argument("--clip_duration_s", type=float, default=4.0)
    p.add_argument("--small_count", type=int, default=5)
    p.add_argument("--n_mdb_drums", type=int, default=None)
    p.add_argument("--n_e_gmd", type=int, default=None)
    p.add_argument("--n_fsl10k", type=int, default=None)
    p.add_argument("--n_synthetic", type=int, default=None)
    p.add_argument("--progress_interval", type=int, default=50)
    p.add_argument("--max_synthetic_kit_tries", type=int, default=256)
    p.add_argument("--max_synthetic_midi_tries", type=int, default=64)
    p.add_argument(
        "--dcomposer_config",
        type=str,
        default="conf/dcomposer.yml",
    )
    return p.parse_args()


def ensure_empty_or_create(path: Path, overwrite: bool):
    if path.exists():
        if overwrite:
            shutil.rmtree(path)
        elif any(path.iterdir()):
            raise FileExistsError(
                f"{path} exists and is non-empty; pass --overwrite to replace"
            )
    path.mkdir(parents=True, exist_ok=True)


def write_signal(sig: AudioSignal, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    out = sig.cpu()
    if out.audio_data.dtype not in (
        torch.float32,
        torch.float64,
        torch.int16,
        torch.int32,
    ):
        out.audio_data = out.audio_data.to(torch.float32)
    out.write(path)


def fit_signal(
    sig: AudioSignal, target_sr: int, target_channels: int, target_duration_s: float
) -> AudioSignal:
    target_len = int(round(float(target_sr) * float(target_duration_s)))
    if sig.sample_rate != int(target_sr):
        from dcomposer import dsp

        sig = dsp.resample(sig, int(target_sr), inplace=False)
    if sig.num_channels != int(target_channels):
        if int(target_channels) == 1:
            sig = sig.to_mono()
        elif sig.num_channels == 1:
            sig.audio_data = sig.audio_data.repeat(1, int(target_channels), 1)
        else:
            sig.audio_data = sig.audio_data.mean(dim=1, keepdim=True).repeat(
                1, int(target_channels), 1
            )
    x = sig.audio_data
    if x.shape[-1] < target_len:
        x = torch.nn.functional.pad(x, (0, target_len - x.shape[-1]))
    elif x.shape[-1] > target_len:
        x = x[..., :target_len]
    out = AudioSignal(x, sample_rate=int(target_sr))
    out.metadata = dict(getattr(sig, "metadata", {}) or {})
    return out


def empty_coarse_dict() -> Dict[str, List[float]]:
    labels = list(dict.fromkeys(COARSE_MIDI_NOTE_TO_COARSE_LABEL.values()))
    return {label: [] for label in labels}


def empty_fine_dict() -> Dict[str, List[float]]:
    labels = list(dict.fromkeys(FINE_MIDI_NOTE_TO_FINE_LABEL.values()))
    return {label: [] for label in labels}


def canonical_fine_note_for_coarse(coarse_note: int) -> int:
    mapping = {
        36: 36,
        37: 37,
        38: 38,
        39: 39,
        42: 42,
        43: 43,
        49: 49,
        54: 54,
        56: 56,
        67: 67,
        70: 70,
        71: 71,
    }
    return int(mapping[int(coarse_note)])


def encode_var_len(value: int) -> bytes:
    if value < 0:
        raise ValueError(value)
    out = [value & 0x7F]
    value >>= 7
    while value:
        out.append(0x80 | (value & 0x7F))
        value >>= 7
    return bytes(reversed(out))


def make_midi_bytes(
    events: Sequence[Dict],
    note_duration_s: float,
    tempo_bpm: float,
    ticks_per_beat: int,
) -> bytes:
    tempo_us_per_beat = int(round(60_000_000.0 / float(tempo_bpm)))
    ticks_per_second = float(ticks_per_beat) * (1_000_000.0 / float(tempo_us_per_beat))
    note_dur_ticks = max(1, int(round(float(note_duration_s) * ticks_per_second)))
    midi_events = []
    for event in events:
        note = int(event["fine_note"])
        onset_tick = max(0, int(round(float(event["onset_sec"]) * ticks_per_second)))
        off_tick = onset_tick + note_dur_ticks
        midi_events.append((onset_tick, 0, bytes([0x99, note, 100])))
        midi_events.append((off_tick, 1, bytes([0x89, note, 0])))
    midi_events.sort(key=lambda x: (x[0], x[1]))
    track = bytearray()
    track.extend(encode_var_len(0))
    track.extend(b"\xFF\x51\x03")
    track.extend(struct.pack(">I", tempo_us_per_beat)[1:])
    track.extend(encode_var_len(0))
    track.extend(b"\xC9\x00")
    prev_tick = 0
    for abs_tick, _priority, msg in midi_events:
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


def make_note_midi_bytes(
    note_events: Sequence[Dict],
    note_duration_s: float,
    tempo_bpm: float,
    ticks_per_beat: int,
) -> bytes:
    tempo_us_per_beat = int(round(60_000_000.0 / float(tempo_bpm)))
    ticks_per_second = float(ticks_per_beat) * (1_000_000.0 / float(tempo_us_per_beat))
    note_dur_ticks = max(1, int(round(float(note_duration_s) * ticks_per_second)))
    midi_events = []
    for event in note_events:
        note = int(event["note"])
        velocity = int(event.get("velocity", 100))
        onset_tick = max(0, int(round(float(event["onset_sec"]) * ticks_per_second)))
        off_tick = onset_tick + note_dur_ticks
        midi_events.append((onset_tick, 0, bytes([0x99, note, velocity])))
        midi_events.append((off_tick, 1, bytes([0x89, note, 0])))
    midi_events.sort(key=lambda x: (x[0], x[1]))
    track = bytearray()
    track.extend(encode_var_len(0))
    track.extend(b"\xFF\x51\x03")
    track.extend(struct.pack(">I", tempo_us_per_beat)[1:])
    track.extend(encode_var_len(0))
    track.extend(b"\xC9\x00")
    prev_tick = 0
    for abs_tick, _priority, msg in midi_events:
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


def build_transcript_payload(events: Sequence[Dict], extra: Dict) -> Dict:
    coarse = empty_coarse_dict()
    fine = empty_fine_dict()
    out_events = []
    for i, event in enumerate(
        sorted(events, key=lambda x: (x["onset_sec"], x["fine_note"]))
    ):
        onset = float(event["onset_sec"])
        fine_note = int(event["fine_note"])
        fine_label = str(event["fine_label"])
        coarse_note = int(event["coarse_note"])
        coarse_label = str(event["coarse_label"])
        coarse[coarse_label].append(onset)
        fine[fine_label].append(onset)
        out_events.append(
            {
                "event_idx": int(i),
                "onset_sec": onset,
                "fine_note": fine_note,
                "fine_label": fine_label,
                "coarse_note": coarse_note,
                "coarse_label": coarse_label,
                **({"velocity": int(event["velocity"])} if "velocity" in event else {}),
            }
        )
    for k in coarse:
        coarse[k].sort()
    for k in fine:
        fine[k].sort()
    return {**extra, "coarse": coarse, "fine": fine, "events": out_events}


def write_transcript(
    base_path: Path,
    payload: Dict,
    midi_tempo_bpm: float,
    midi_note_duration_s: float,
    ticks_per_beat: int,
):
    base_path.parent.mkdir(parents=True, exist_ok=True)
    base_path.with_suffix(".json").write_text(json.dumps(payload, indent=2))
    base_path.with_suffix(".mid").write_bytes(
        make_midi_bytes(
            payload["events"], midi_note_duration_s, midi_tempo_bpm, ticks_per_beat
        )
    )


def choose_dense_window(
    event_onsets_sec: np.ndarray,
    clip_duration_s: float,
    total_duration_s: float,
    rng: random.Random,
) -> float:
    max_start = max(0.0, float(total_duration_s) - float(clip_duration_s))
    if len(event_onsets_sec) == 0 or max_start <= 0.0:
        return 0.0
    candidates = []
    for onset in event_onsets_sec.tolist():
        start = min(max(onset - clip_duration_s / 2.0, 0.0), max_start)
        end = start + clip_duration_s
        count = int(np.sum((event_onsets_sec >= start) & (event_onsets_sec < end)))
        candidates.append((count, start))
    candidates.sort(key=lambda x: (-x[0], x[1]))
    top = candidates[: min(16, len(candidates))]
    return float(top[rng.randrange(len(top))][1])


def crop_events(
    events: Sequence[Tuple[float, int, int]], start_sec: float, duration_s: float
) -> List[Dict]:
    end = float(start_sec) + float(duration_s)
    cropped = []
    for onset_sec, fine_note, velocity in events:
        if float(start_sec) <= float(onset_sec) < end:
            fine_note = int(fine_note)
            coarse_note = int(
                FINE_MIDI_NOTE_TO_COARSE_MIDI_NOTE.get(fine_note, fine_note)
            )
            cropped.append(
                {
                    "onset_sec": float(onset_sec) - float(start_sec),
                    "fine_note": fine_note,
                    "fine_label": FINE_MIDI_NOTE_TO_FINE_LABEL[fine_note],
                    "coarse_note": coarse_note,
                    "coarse_label": COARSE_MIDI_NOTE_TO_COARSE_LABEL[coarse_note],
                    "velocity": int(velocity),
                }
            )
    cropped.sort(key=lambda x: (x["onset_sec"], x["fine_note"]))
    return cropped


def load_audio_window(
    path: Path, *, offset: float, duration_s: float, sample_rate: int, num_channels: int
) -> AudioSignal:
    sig, _ = load_excerpt(
        str(path),
        duration=float(duration_s),
        sample_rate=int(sample_rate),
        state=np.random.RandomState(0),
        offset=float(offset),
        from_start=False,
        loudness_cutoff=None,
        num_tries=0,
        num_channels=int(num_channels),
    )
    return fit_signal(
        sig,
        target_sr=sample_rate,
        target_channels=num_channels,
        target_duration_s=duration_s,
    )


def salient_audio_excerpt(
    path: Path,
    *,
    duration_s: float,
    sample_rate: int,
    num_channels: int,
    rng_seed: int,
    loudness_cutoff: float,
    num_tries: int,
) -> Tuple[AudioSignal, float]:
    sig, off = load_excerpt(
        str(path),
        duration=float(duration_s),
        sample_rate=int(sample_rate),
        state=np.random.RandomState(int(rng_seed) & 0x7FFFFFFF),
        from_start=False,
        loudness_cutoff=float(loudness_cutoff),
        num_tries=int(num_tries),
        num_channels=int(num_channels),
    )
    sig = fit_signal(
        sig,
        target_sr=sample_rate,
        target_channels=num_channels,
        target_duration_s=duration_s,
    )
    return sig, float(off)


def discover_audio_files(root: Path) -> List[Path]:
    return sorted(
        p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in AUDIO_EXTS
    )


def read_audio_paths_from_manifests(
    csv_paths: Sequence[Path], field: str
) -> List[Path]:
    out: List[Path] = []
    seen = set()
    for csv_path in csv_paths:
        with csv_path.open() as f:
            reader = csv.DictReader(f)
            for row in reader:
                value = row.get(field, "")
                if not value:
                    continue
                path = Path(value).expanduser()
                if path.is_file():
                    key = str(path.resolve())
                    if key not in seen:
                        out.append(path)
                        seen.add(key)
    return sorted(out)


def read_mdb_lines(txt_path: Path) -> List[Tuple[float, str]]:
    events = []
    with txt_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            onset_s, label = line.split()[:2]
            events.append((float(onset_s), str(label)))
    return events


def write_mdb_excerpt_txt(path: Path, events: Sequence[Tuple[float, str]]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for onset_sec, label in events:
            f.write(f"{float(onset_sec):.6f}\t{label}\n")


def build_mdb_records(root: Path) -> List[Dict]:
    audio_dir = root / "audio" / "drum_only"
    txt_dir = root / "annotations" / "class"
    records = []
    for wav_path in sorted(audio_dir.glob("*.wav")):
        base = wav_path.stem
        if base.endswith("_Drum"):
            base = base[: -len("_Drum")]
        txt_path = txt_dir / f"{base}_class.txt"
        if not txt_path.is_file():
            continue
        info = sf.info(str(wav_path))
        records.append(
            {
                "audio_path": wav_path,
                "txt_path": txt_path,
                "duration_s": float(info.duration),
            }
        )
    return records


def build_egmd_records(root: Path) -> List[Dict]:
    rows = []
    csv_path = root / "e-gmd-v1.0.0.csv"
    with csv_path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("split") != "test":
                continue
            audio_path = root / row["audio_filename"]
            midi_path = root / row["midi_filename"]
            if audio_path.is_file() and midi_path.is_file():
                rows.append(
                    {"audio_path": audio_path, "midi_path": midi_path, "row": row}
                )
    return rows


def read_vlq(buf: bytes, i: int) -> Tuple[int, int]:
    value = 0
    while True:
        b = buf[i]
        i += 1
        value = (value << 7) | (b & 0x7F)
        if not (b & 0x80):
            break
    return value, i


def parse_percussion_note_ons_from_midi(path: Path) -> List[Dict]:
    data = path.read_bytes()
    if data[:4] != b"MThd":
        raise ValueError(f"Invalid MIDI header in {path}")
    header_len = struct.unpack(">I", data[4:8])[0]
    fmt, ntrks, division = struct.unpack(">HHH", data[8:14])
    if division & 0x8000:
        raise ValueError("SMPTE time division not supported.")
    ticks_per_beat = int(division)
    offset = 8 + header_len
    tempo_events: List[Tuple[int, int]] = []
    note_events: List[Tuple[int, int, int]] = []
    for _ in range(ntrks):
        if data[offset : offset + 4] != b"MTrk":
            raise ValueError(f"Invalid MIDI track chunk in {path}")
        track_len = struct.unpack(">I", data[offset + 4 : offset + 8])[0]
        track = data[offset + 8 : offset + 8 + track_len]
        offset += 8 + track_len
        i = 0
        abs_tick = 0
        running_status = None
        while i < len(track):
            delta, i = read_vlq(track, i)
            abs_tick += delta
            status = track[i]
            if status < 0x80:
                if running_status is None:
                    raise ValueError(f"Missing running status in {path}")
                status = running_status
            else:
                i += 1
                if status < 0xF0:
                    running_status = status
            if status == 0xFF:
                meta_type = track[i]
                i += 1
                meta_len, i = read_vlq(track, i)
                meta_data = track[i : i + meta_len]
                i += meta_len
                if meta_type == 0x51 and meta_len == 3:
                    tempo_events.append((abs_tick, int.from_bytes(meta_data, "big")))
                continue
            if status in (0xF0, 0xF7):
                syx_len, i = read_vlq(track, i)
                i += syx_len
                continue
            op = status & 0xF0
            ch = status & 0x0F
            if op in (0xC0, 0xD0):
                i += 1
                continue
            data1 = track[i]
            data2 = track[i + 1]
            i += 2
            if op == 0x90 and ch == 9 and data2 > 0:
                note_events.append((abs_tick, int(data1), int(data2)))
    if not tempo_events:
        tempo_events = [(0, 500000)]
    tempo_events = sorted(tempo_events, key=lambda x: x[0])
    if tempo_events[0][0] != 0:
        tempo_events = [(0, 500000)] + tempo_events

    segments = []
    cur_sec = 0.0
    for i, (tick, tempo) in enumerate(tempo_events):
        next_tick = tempo_events[i + 1][0] if i + 1 < len(tempo_events) else None
        segments.append((tick, next_tick, tempo, cur_sec))
        if next_tick is not None:
            cur_sec += ((next_tick - tick) / float(ticks_per_beat)) * (
                tempo / 1_000_000.0
            )

    def tick_to_sec(tick: int) -> float:
        for start_tick, next_tick, tempo, start_sec in segments:
            if next_tick is None or tick < next_tick:
                return float(
                    start_sec
                    + ((tick - start_tick) / float(ticks_per_beat))
                    * (tempo / 1_000_000.0)
                )
        return 0.0

    return [
        {"onset_sec": tick_to_sec(tick), "note": note, "velocity": vel}
        for tick, note, vel in sorted(note_events, key=lambda x: x[0])
    ]


def sample_without_replacement(
    records: Sequence[Dict], n: int, rng: random.Random
) -> List[Dict]:
    if len(records) < n:
        raise ValueError(f"Requested {n} records from only {len(records)} available")
    idxs = list(range(len(records)))
    rng.shuffle(idxs)
    return [records[i] for i in idxs[:n]]


def sample_with_replacement(
    records: Sequence[Dict], n: int, rng: random.Random
) -> List[Dict]:
    if len(records) == 0 and n > 0:
        raise ValueError("Cannot sample from an empty record set.")
    return [records[rng.randrange(len(records))] for _ in range(n)]


def resolve_count(explicit: Optional[int], fallback: int, name: str) -> int:
    value = fallback if explicit is None else int(explicit)
    if value < 0:
        raise ValueError(f"`{name}` must be >= 0, got {value}")
    return value


def should_log_progress(i: int, total: int, interval: int) -> bool:
    if total <= 0:
        return False
    if i == 0 or i + 1 == total:
        return True
    return interval > 0 and ((i + 1) % interval == 0)


def create_transcription_dataset(
    dataset_name: str,
    records: Sequence[Dict],
    output_dir: Path,
    n_examples: int,
    rng: random.Random,
    clip_duration_s: float,
    sample_rate: int,
    num_channels: int,
    midi_tempo_bpm: float,
    midi_note_duration_s: float,
    ticks_per_beat: int,
    source_kind: str,
    progress_interval: int,
    reuse_source_files: bool = False,
):
    dataset_dir = output_dir / dataset_name
    audio_dir = dataset_dir / "audio"
    transcript_dir = dataset_dir / "transcript"
    audio_dir.mkdir(parents=True, exist_ok=True)
    transcript_dir.mkdir(parents=True, exist_ok=True)
    log(f"[{dataset_name}] creating {n_examples} excerpt(s)")
    chosen = (
        sample_with_replacement(records, n_examples, rng)
        if reuse_source_files
        else sample_without_replacement(records, n_examples, rng)
    )
    manifest = []
    for i, record in enumerate(chosen):
        if should_log_progress(i, n_examples, progress_interval):
            log(f"[{dataset_name}] {i + 1}/{n_examples}")
        stem = f"{i:05d}"
        if source_kind == "mdb":
            events = read_mdb_lines(record["txt_path"])
            total_duration = float(record["duration_s"])
        elif source_kind == "egmd":
            events = parse_percussion_note_ons_from_midi(record["midi_path"])
            total_duration = float(record["row"]["duration"])
        else:
            raise ValueError(source_kind)
        onset_arr = np.asarray(
            [e[0] if source_kind == "mdb" else e["onset_sec"] for e in events],
            dtype=float,
        )
        start = choose_dense_window(onset_arr, clip_duration_s, total_duration, rng)
        if len(events) > 0:
            # Fallback to a guaranteed event window.
            preview_count = int(
                np.sum((onset_arr >= start) & (onset_arr < start + clip_duration_s))
            )
            if preview_count == 0:
                onset0 = float(
                    events[0][0] if source_kind == "mdb" else events[0]["onset_sec"]
                )
                start = min(
                    max(onset0 - clip_duration_s / 2.0, 0.0),
                    max(0.0, total_duration - clip_duration_s),
                )
        sig = load_audio_window(
            record["audio_path"],
            offset=start,
            duration_s=clip_duration_s,
            sample_rate=sample_rate,
            num_channels=num_channels,
        )
        audio_path = audio_dir / f"{stem}.wav"
        write_signal(sig, audio_path)
        if source_kind == "mdb":
            clipped_events = [
                (float(onset_sec) - float(start), str(label))
                for onset_sec, label in events
                if float(start)
                <= float(onset_sec)
                < float(start) + float(clip_duration_s)
            ]
            out_path = (transcript_dir / stem).with_suffix(".txt")
            write_mdb_excerpt_txt(out_path, clipped_events)
            manifest.append(
                {
                    "stem": stem,
                    "audio_path": str(audio_path),
                    "transcript_txt": str(out_path),
                    "source_audio_path": str(record["audio_path"]),
                    "source_transcript_path": str(record["txt_path"]),
                    "excerpt_offset_sec": float(start),
                    "n_events": len(clipped_events),
                }
            )
        else:
            clipped_events = [
                {
                    "onset_sec": float(event["onset_sec"]) - float(start),
                    "note": int(event["note"]),
                    "velocity": int(event["velocity"]),
                }
                for event in events
                if float(start)
                <= float(event["onset_sec"])
                < float(start) + float(clip_duration_s)
            ]
            out_path = (transcript_dir / stem).with_suffix(".mid")
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_bytes(
                make_note_midi_bytes(
                    clipped_events, midi_note_duration_s, midi_tempo_bpm, ticks_per_beat
                )
            )
            manifest.append(
                {
                    "stem": stem,
                    "audio_path": str(audio_path),
                    "transcript_mid": str(out_path),
                    "source_audio_path": str(record["audio_path"]),
                    "source_transcript_path": str(record["midi_path"]),
                    "excerpt_offset_sec": float(start),
                    "n_events": len(clipped_events),
                }
            )
    (dataset_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def create_audio_only_dataset(
    dataset_name: str,
    audio_paths: Sequence[Path],
    output_dir: Path,
    n_examples: int,
    rng: random.Random,
    clip_duration_s: float,
    sample_rate: int,
    num_channels: int,
    loudness_cutoff: float,
    salience_num_tries: int,
    progress_interval: int,
    reuse_source_files: bool = False,
):
    dataset_dir = output_dir / dataset_name
    audio_dir = dataset_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    log(f"[{dataset_name}] creating {n_examples} excerpt(s)")
    source_records = [{"audio_path": p} for p in audio_paths]
    chosen = (
        sample_with_replacement(source_records, n_examples, rng)
        if reuse_source_files
        else sample_without_replacement(source_records, n_examples, rng)
    )
    manifest = []
    for i, record in enumerate(chosen):
        if should_log_progress(i, n_examples, progress_interval):
            log(f"[{dataset_name}] {i + 1}/{n_examples}")
        sig, start = salient_audio_excerpt(
            record["audio_path"],
            duration_s=clip_duration_s,
            sample_rate=sample_rate,
            num_channels=num_channels,
            rng_seed=rng.randrange(2**31 - 1),
            loudness_cutoff=loudness_cutoff,
            num_tries=salience_num_tries,
        )
        stem = f"{i:05d}"
        audio_path = audio_dir / f"{stem}.wav"
        write_signal(sig, audio_path)
        manifest.append(
            {
                "stem": stem,
                "audio_path": str(audio_path),
                "source_audio_path": str(record["audio_path"]),
                "excerpt_offset_sec": float(start),
            }
        )
    (dataset_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def load_dcomposer_args(cfg_path: Path) -> Dict:
    return load_config(cfg_path)


def build_synthetic_mix_transform(run_args: Dict):
    with argbind.scope(run_args, "train_mix"):
        return train_dcomposer.build_transform(
            prob=1.0, names=["ParametricEqualizer", "Compressor"]
        )


def build_synthetic_builder_cfg(run_args: Dict) -> Dict:
    with argbind.scope(run_args, "train"):
        cfg = train_dcomposer.build_builder_config()
    cfg = dict(cfg)
    cfg["p_speed"] = 0.0
    cfg["p_timing_jitter"] = 0.0
    cfg["p_velocity_jitter"] = 0.0
    cfg["p_time_shift"] = 0.0
    cfg["max_duration"] = float(
        run_args.get("OneShotMidiDataset.max_midi_duration", 4.0)
    )
    return cfg


def build_synthetic_dataset(run_args: Dict) -> OneShotMidiDataset:
    cfg = {
        "midi_sources": run_args["val/OneShotMidiDataset.midi_sources"],
        "midi_source_weights": run_args["val/OneShotMidiDataset.midi_source_weights"],
        "oneshot_sources": run_args["val/OneShotMidiDataset.oneshot_sources"],
        "oneshot_source_weights": run_args[
            "val/OneShotMidiDataset.oneshot_source_weights"
        ],
        "sample_rate": int(run_args["OneShotMidiDataset.sample_rate"]),
        "duration": float(run_args["OneShotMidiDataset.duration"]),
        "max_midi_events": int(run_args["OneShotMidiDataset.max_midi_events"]),
        "max_midi_duration": float(run_args["OneShotMidiDataset.max_midi_duration"]),
        "n_examples": 1000000,
        "num_channels": int(run_args["OneShotMidiDataset.num_channels"]),
        "n_variations": 1,
        "hierarchy_level": str(run_args["OneShotMidiDataset.hierarchy_level"]),
        "midi_relative_path": run_args.get(
            "OneShotMidiDataset.midi_relative_path", "."
        ),
        "oneshot_relative_path": run_args.get(
            "OneShotMidiDataset.oneshot_relative_path", "."
        ),
        "oneshot_with_replacement": True,
        "audio_backend": str(
            run_args.get("OneShotMidiDataset.audio_backend", "packed")
        ),
        "midi_max_tries": int(run_args.get("OneShotMidiDataset.midi_max_tries", 4)),
        "shuffle_state": int(run_args.get("seed", 0)),
        "p_same_kit": 1.0,
        "same_kit_max_tries": int(
            run_args.get("OneShotMidiDataset.same_kit_max_tries", 2)
        ),
        "same_kit_min_coarse_classes": int(
            run_args.get("OneShotMidiDataset.same_kit_min_coarse_classes", 3)
        ),
        "from_start": True,
        "loudness_cutoff": None,
        "salience_num_tries": int(
            run_args.get("OneShotMidiDataset.salience_num_tries", 4) or 4
        ),
    }
    return OneShotMidiDataset(**cfg)


def build_kit_pool(
    dataset: OneShotMidiDataset, rng: random.Random
) -> List[Tuple[int, str]]:
    pool = []
    for source_idx, kits in enumerate(dataset._oneshot_kits):
        counts = dataset._oneshot_kit_coarse_counts[source_idx]
        for kit_name in sorted(kits):
            if not kit_name:
                continue
            if counts.get(kit_name, 0) >= int(dataset.same_kit_min_coarse_classes):
                pool.append((int(source_idx), str(kit_name)))
    if not pool:
        pool = [
            (index, "")
            for index, groups in enumerate(dataset._oneshot_rows_by_group)
            if any(groups.values())
        ]
    if not pool:
        raise RuntimeError("No usable samples found for synthetic evaluation data")
    rng.shuffle(pool)
    return pool


def sample_fixed_kit_candidates(
    dataset: OneShotMidiDataset,
    midi_excerpt: Dict,
    item_idx: int,
    base_seed: int,
    source_idx: int,
    kit_name: str,
):
    requested_labels = dataset._requested_group_labels(midi_excerpt)
    groups = {}
    chosen_rows = {}
    for i, group_label in enumerate(dataset.group_labels):
        group_state = np.random.RandomState(
            (
                int(dataset.shuffle_state)
                + int(item_idx)
                + 9973 * (i + 1)
                + int(base_seed)
            )
            & 0x7FFFFFFF
        )
        if group_label not in requested_labels:
            rows = []
            sampled_rows = []
        else:
            groups_by_label = (
                dataset._oneshot_rows_by_kit_group[source_idx][kit_name]
                if kit_name
                else dataset._oneshot_rows_by_group[source_idx]
            )
            rows = groups_by_label.get(group_label, [])
            sampled_rows = dataset._sample_rows(rows, group_state) if len(rows) else []
        load_rows = sampled_rows if sampled_rows else []
        groups[group_label] = dataset._load_packed_group_signal(
            load_rows, group_state, i
        )
        chosen_rows[group_label] = sampled_rows[0] if sampled_rows else None
    return {
        "same_kit": bool(kit_name),
        "kit_name": kit_name,
        "groups": groups,
    }, chosen_rows


def create_synthetic_dataset(
    output_dir: Path,
    n_examples: int,
    rng: random.Random,
    run_args: Dict,
    midi_tempo_bpm: float,
    midi_note_duration_s: float,
    ticks_per_beat: int,
    max_synthetic_kit_tries: int,
    max_synthetic_midi_tries: int,
    progress_interval: int,
):
    dataset_dir = output_dir / "synthetic"
    audio_dir = dataset_dir / "audio"
    transcript_dir = dataset_dir / "transcript"
    oneshot_dir = dataset_dir / "oneshot"
    audio_dir.mkdir(parents=True, exist_ok=True)
    transcript_dir.mkdir(parents=True, exist_ok=True)
    oneshot_dir.mkdir(parents=True, exist_ok=True)

    dataset = build_synthetic_dataset(run_args)
    builder_cfg = build_synthetic_builder_cfg(run_args)
    mix_transform = build_synthetic_mix_transform(run_args)
    kit_pool = build_kit_pool(dataset, rng)
    kit_cursor = 0
    manifest = []
    log(f"[synthetic] creating {n_examples} excerpt(s)")

    def next_kit_with_coverage(midi_excerpt: Dict):
        nonlocal kit_cursor, kit_pool
        requested = dataset._requested_group_labels(midi_excerpt)
        for _ in range(min(max_synthetic_kit_tries, max(1, len(kit_pool)))):
            source_idx, kit_name = kit_pool[kit_cursor]
            kit_cursor = (kit_cursor + 1) % len(kit_pool)
            if kit_cursor == 0:
                rng.shuffle(kit_pool)
            groups = (
                dataset._oneshot_rows_by_kit_group[source_idx][kit_name]
                if kit_name
                else dataset._oneshot_rows_by_group[source_idx]
            )
            if requested and all(groups.get(label) for label in requested):
                return source_idx, kit_name
        return None

    dataset_len = len(dataset)
    for ex_idx in range(n_examples):
        if should_log_progress(ex_idx, n_examples, progress_interval):
            log(f"[synthetic] {ex_idx + 1}/{n_examples}")
        selected = None
        for attempt in range(max_synthetic_midi_tries):
            item_idx = rng.randrange(dataset_len)
            state = np.random.RandomState(
                (int(dataset.shuffle_state) + int(item_idx) + attempt * 1013)
                & 0x7FFFFFFF
            )
            midi_excerpt = dataset._load_midi_excerpt(idx=item_idx, state=state)
            kit_choice = next_kit_with_coverage(midi_excerpt)
            if kit_choice is None:
                continue
            source_idx, kit_name = kit_choice
            oneshots, chosen_rows = sample_fixed_kit_candidates(
                dataset,
                midi_excerpt,
                item_idx,
                ex_idx * 1009 + attempt,
                source_idx,
                kit_name,
            )
            selected = (
                item_idx,
                midi_excerpt,
                oneshots,
                chosen_rows,
                source_idx,
                kit_name,
            )
            break
        if selected is None:
            raise RuntimeError(
                f"Failed to sample synthetic example {ex_idx} with usable MIDI and sample coverage."
            )

        item_idx, midi_excerpt, oneshots, chosen_rows, source_idx, kit_name = selected
        item = {
            "hierarchy_level": dataset.hierarchy_level,
            "midi": midi_excerpt,
            "oneshots": oneshots,
            "idx": item_idx,
        }
        batch = dataset.collate([item])
        batch = build_one_shot_midi_batch(
            batch,
            oneshot_transform=None,
            mix_transform=mix_transform,
            state=[item_idx],
            **builder_cfg,
        )
        mix_sig = fit_signal(
            batch["signal"],
            target_sr=int(run_args["OneShotMidiDataset.sample_rate"]),
            target_channels=int(run_args["OneShotMidiDataset.num_channels"]),
            target_duration_s=float(run_args["OneShotMidiDataset.max_midi_duration"]),
        )
        stem = f"{ex_idx:05d}"
        audio_path = audio_dir / f"{stem}.wav"
        write_signal(mix_sig, audio_path)

        notes = batch["arrangement"]["notes"][0, 0].tolist()
        onsets = batch["arrangement"]["onsets_sec"][0, 0].tolist()
        velocities = batch["arrangement"]["velocities"][0, 0].tolist()
        target_oneshots = batch["targets"]["oneshots"]
        oneshot_num_samples = int(batch["targets"]["oneshot_num_samples"][0].item())
        shot_subdir = oneshot_dir / stem
        shot_subdir.mkdir(parents=True, exist_ok=True)

        events = []
        for seq_idx, (note, onset_sec, vel) in enumerate(
            zip(notes, onsets, velocities)
        ):
            coarse_note = int(note)
            coarse_label = COARSE_MIDI_NOTE_TO_COARSE_LABEL[coarse_note]
            chosen = chosen_rows.get(coarse_label)
            if chosen is None:
                fine_note = canonical_fine_note_for_coarse(coarse_note)
                fine_label = FINE_MIDI_NOTE_TO_FINE_LABEL[fine_note]
            else:
                fine_note = int(chosen.get("midi_fine_note", coarse_note))
                fine_label = str(chosen.get("fine_label", ""))
                if (
                    fine_note < 0
                    or fine_note not in FINE_MIDI_NOTE_TO_FINE_LABEL
                    or fine_label not in FINE_MIDI_NOTE_TO_FINE_LABEL.values()
                ):
                    fine_note = canonical_fine_note_for_coarse(coarse_note)
                    fine_label = FINE_MIDI_NOTE_TO_FINE_LABEL[fine_note]
            events.append(
                {
                    "onset_sec": float(onset_sec),
                    "fine_note": fine_note,
                    "fine_label": fine_label,
                    "coarse_note": coarse_note,
                    "coarse_label": coarse_label,
                    "velocity": int(vel),
                }
            )
            shot_name = f"{seq_idx:04d}_{coarse_label}_{coarse_note}_{fine_label}_{fine_note}.wav"
            start_sample = seq_idx * oneshot_num_samples
            end_sample = start_sample + oneshot_num_samples
            clip = AudioSignal(
                target_oneshots.audio_data[:, :, start_sample:end_sample].cpu(),
                sample_rate=target_oneshots.sample_rate,
            )
            clip.metadata = {}
            write_signal(clip, shot_subdir / shot_name)

        transcript_payload = build_transcript_payload(
            events,
            {
                "dataset": "synthetic",
                "source_midi_dataset": midi_excerpt.get("source_dataset"),
                "source_midi": midi_excerpt.get("source_midi"),
                "source_oneshot_manifest": str(dataset.oneshot_sources[source_idx]),
                "source_kit_name": kit_name,
                "duration_s": float(run_args["OneShotMidiDataset.max_midi_duration"]),
            },
        )
        write_transcript(
            transcript_dir / stem,
            transcript_payload,
            midi_tempo_bpm,
            midi_note_duration_s,
            ticks_per_beat,
        )
        manifest.append(
            {
                "stem": stem,
                "audio_path": str(audio_path),
                "transcript_json": str((transcript_dir / stem).with_suffix(".json")),
                "transcript_mid": str((transcript_dir / stem).with_suffix(".mid")),
                "oneshot_dir": str(shot_subdir),
                "source_midi": midi_excerpt.get("source_midi"),
                "source_kit_name": kit_name,
                "n_events": len(events),
            }
        )

    (dataset_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def main():
    cli = parse_args()
    rng = random.Random(int(cli.seed))
    output_dir = Path(cli.output_dir).expanduser().resolve()
    ensure_empty_or_create(output_dir, overwrite=cli.overwrite)
    n_mdb_drums = resolve_count(cli.n_mdb_drums, cli.small_count, "n_mdb_drums")
    n_e_gmd = resolve_count(cli.n_e_gmd, cli.small_count, "n_e_gmd")
    n_fsl10k = resolve_count(cli.n_fsl10k, cli.small_count, "n_fsl10k")
    n_synthetic = resolve_count(cli.n_synthetic, cli.small_count, "n_synthetic")

    t0 = time.time()
    log("Building evaluation input set")

    mdb_manifest = create_transcription_dataset(
        "mdb_drums",
        build_mdb_records(PROJECT_DIR / "data" / "MDBDrums" / "MDB Drums"),
        output_dir,
        n_mdb_drums,
        rng,
        cli.clip_duration_s,
        cli.sample_rate,
        2,
        cli.midi_tempo_bpm,
        cli.midi_note_duration_s,
        cli.ticks_per_beat,
        "mdb",
        cli.progress_interval,
        reuse_source_files=True,
    )

    egmd_manifest = create_transcription_dataset(
        "e_gmd",
        build_egmd_records(PROJECT_DIR / "data" / "e-gmd-v1.0.0"),
        output_dir,
        n_e_gmd,
        rng,
        cli.clip_duration_s,
        cli.sample_rate,
        1,
        cli.midi_tempo_bpm,
        cli.midi_note_duration_s,
        cli.ticks_per_beat,
        "egmd",
        cli.progress_interval,
        reuse_source_files=True,
    )

    fsl_audio = read_audio_paths_from_manifests(
        [
            PROJECT_DIR / "manifests" / "fsl" / "train.csv",
            PROJECT_DIR / "manifests" / "fsl" / "val.csv",
            PROJECT_DIR / "manifests" / "fsl" / "test.csv",
        ],
        field="oneshot",
    )
    fsl_manifest = create_audio_only_dataset(
        "fsl10k",
        fsl_audio,
        output_dir,
        n_fsl10k,
        rng,
        cli.clip_duration_s,
        cli.sample_rate,
        1,
        cli.salience_loudness_cutoff,
        cli.salience_num_tries,
        cli.progress_interval,
    )

    run_args = load_dcomposer_args(Path(cli.dcomposer_config))
    synthetic_manifest = create_synthetic_dataset(
        output_dir,
        n_synthetic,
        rng,
        run_args,
        cli.midi_tempo_bpm,
        cli.midi_note_duration_s,
        cli.ticks_per_beat,
        cli.max_synthetic_kit_tries,
        cli.max_synthetic_midi_tries,
        cli.progress_interval,
    )

    meta = {
        "output_dir": str(output_dir),
        "seed": int(cli.seed),
        "clip_duration_s": float(cli.clip_duration_s),
        "sample_rate": int(cli.sample_rate),
        "small_count_fallback": int(cli.small_count),
        "counts": {
            "mdb_drums": n_mdb_drums,
            "e_gmd": n_e_gmd,
            "fsl10k": n_fsl10k,
            "synthetic": n_synthetic,
        },
        "datasets": {
            "mdb_drums": {"n_examples": len(mdb_manifest)},
            "e_gmd": {"n_examples": len(egmd_manifest)},
            "fsl10k": {"n_examples": len(fsl_manifest)},
            "synthetic": {"n_examples": len(synthetic_manifest)},
        },
        "elapsed_seconds": float(time.time() - t0),
    }
    (output_dir / "_metadata.json").write_text(json.dumps(meta, indent=2))
    log(f"Wrote evaluation inputs to {output_dir}")
    log(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
