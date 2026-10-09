import csv
import json
import mmap
from collections import defaultdict
from pathlib import Path
from typing import Dict
from typing import Iterable
from typing import List
from typing import Optional
from typing import Sequence
from typing import Tuple
from typing import Union

import numpy as np
import torch

from .constants import FINE_MIDI_NOTE_TO_COARSE_MIDI_NOTE
from .constants import MIDI_TIME_RES
from .constants import RAW_MIDI_NOTE_TO_FINE_MIDI_NOTE


MIDI_EVENT_DIM = 3  # onset_step, fine_note, velocity


def find_midi_files(root: Union[str, Path]) -> List[Path]:
    root = Path(root)
    paths = []
    for ext in ("*.mid", "*.MID", "*.midi", "*.MIDI"):
        paths.extend(root.rglob(ext))
    return sorted(paths)


def extract_drum_note_events(
    path: Union[str, Path],
    time_res: int = MIDI_TIME_RES,
) -> np.ndarray:
    import mido

    mid = mido.MidiFile(str(path))
    time_sec = 0.0
    events: List[Tuple[int, int, int]] = []

    for msg in mid:
        time_sec += float(msg.time)
        if msg.type != "note_on" or msg.velocity <= 0:
            continue
        if not hasattr(msg, "channel") or int(msg.channel) != 9:
            continue

        fine_note = RAW_MIDI_NOTE_TO_FINE_MIDI_NOTE.get(int(msg.note))
        if fine_note is None:
            continue

        onset_step = int(round(time_res * time_sec))
        velocity = int(msg.velocity)
        events.append((onset_step, fine_note, velocity))

    if not events:
        return np.empty((0, MIDI_EVENT_DIM), dtype=np.int32)

    return np.asarray(events, dtype=np.int32)


def crop_drum_event_excerpt(
    events: np.ndarray,
    max_events: Optional[int] = None,
    max_duration_steps: Optional[int] = None,
    state: Optional[np.random.RandomState] = None,
) -> np.ndarray:
    if events.ndim != 2 or events.shape[-1] != MIDI_EVENT_DIM:
        raise ValueError(
            f"Expected events with shape (n, {MIDI_EVENT_DIM}), got {events.shape}"
        )
    if len(events) == 0:
        return events

    state = np.random.RandomState() if state is None else state

    if max_events is None or len(events) <= max_events:
        excerpt = events.copy()
    else:
        start = int(state.randint(len(events) - int(max_events) + 1))
        excerpt = events[start : start + int(max_events)].copy()

    if len(excerpt) == 0:
        return excerpt

    excerpt[:, 0] -= int(excerpt[0, 0])

    if max_duration_steps is not None:
        excerpt = excerpt[excerpt[:, 0] < int(max_duration_steps)]

    return excerpt


def fine_to_coarse_notes(fine_notes: np.ndarray) -> np.ndarray:
    lut = np.full(128, -1, dtype=np.int64)
    for fine_note, coarse_note in FINE_MIDI_NOTE_TO_COARSE_MIDI_NOTE.items():
        lut[int(fine_note)] = int(coarse_note)
    fine_notes = np.asarray(fine_notes, dtype=np.int64)
    return lut[fine_notes]


def build_text_line_index(
    txt_path: Union[str, Path],
    out_idx_path: Union[str, Path],
    overwrite: bool = False,
) -> int:
    txt_path = Path(txt_path)
    out_idx_path = Path(out_idx_path)
    out_idx_path.parent.mkdir(parents=True, exist_ok=True)

    if out_idx_path.exists() and not overwrite:
        idx = np.load(out_idx_path, mmap_mode="r")
        return int(idx.shape[0])

    offsets = []
    off = 0
    with open(txt_path, "rb") as f:
        for line in f:
            if line.strip():
                offsets.append(off)
            off += len(line)

    idx = np.asarray(offsets, dtype=np.int64)
    np.save(out_idx_path, idx)
    return int(idx.shape[0])


class DrumEventTextIndex:
    """
    Lightweight worker-safe reader for line-based drum MIDI event text files.

    Each non-empty line stores integer triples:
      onset_step fine_note velocity ...
    """

    def __init__(self, txt_path: Union[str, Path], idx_path: Union[str, Path]):
        self.txt_path = Path(txt_path)
        self.idx_path = Path(idx_path)
        self._mm = None
        self._f = None
        self._idx = None

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_mm"] = None
        state["_f"] = None
        state["_idx"] = None
        return state

    def _ensure_open(self):
        if self._mm is None:
            self._f = open(self.txt_path, "rb", buffering=0)
            self._mm = mmap.mmap(self._f.fileno(), 0, access=mmap.ACCESS_READ)
            self._idx = np.load(self.idx_path, mmap_mode="r")

    def __len__(self) -> int:
        self._ensure_open()
        return int(self._idx.shape[0])

    def read_row(self, idx: int) -> np.ndarray:
        self._ensure_open()
        off = int(self._idx[idx])
        self._mm.seek(off)
        line = self._mm.readline().decode("ascii").strip()
        if not line:
            return np.empty((0, MIDI_EVENT_DIM), dtype=np.int32)
        arr = np.fromstring(line, sep=" ", dtype=np.int32)
        arr = arr[: (arr.size // MIDI_EVENT_DIM) * MIDI_EVENT_DIM]
        return arr.reshape(-1, MIDI_EVENT_DIM)

    def close(self):
        try:
            if self._mm is not None:
                self._mm.close()
            if self._f is not None:
                self._f.close()
        finally:
            self._mm = None
            self._f = None
            self._idx = None

    def __del__(self):
        self.close()
