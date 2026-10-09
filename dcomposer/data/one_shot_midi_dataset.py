import csv
import hashlib
from pathlib import Path
from typing import Callable
from typing import Dict
from typing import List
from typing import Optional
from typing import Sequence
from typing import Union

import numpy as np
import torch
import torch.nn.functional as F
from audiotools import AudioSignal
from audiotools.core.util import random_state
from audiotools.core.util import sample_from_dist
from torch.utils.data import Dataset

from ..constants import ACOUSTIC_TOK_OFFSET
from ..constants import BOS_TOK
from ..constants import COARSE_MIDI_NOTE_TO_COARSE_LABEL
from ..constants import DEFAULT_COARSE_LOUDNESS_RANGES
from ..constants import EOS_TOK
from ..constants import FINE_MIDI_NOTE_TO_FINE_LABEL
from ..constants import MAX_ONSET_TOK
from ..constants import MIDI_TIME_RES
from ..constants import MIDI_TO_TOK
from ..constants import ONSET_TOK_OFFSET
from ..constants import PAD_TOK
from ..midi import crop_drum_event_excerpt
from ..midi import DrumEventTextIndex
from ..midi import fine_to_coarse_notes
from ..transforms.base import _energy
from ..util import collate
from ..util import load_excerpt
from ..util import normalize_source_weights

COARSE_NOTES = list(COARSE_MIDI_NOTE_TO_COARSE_LABEL.keys())
COARSE_LABELS = [COARSE_MIDI_NOTE_TO_COARSE_LABEL[n] for n in COARSE_NOTES]
COARSE_LABEL_TO_NOTE = {
    label: note for note, label in COARSE_MIDI_NOTE_TO_COARSE_LABEL.items()
}
FINE_NOTES = list(FINE_MIDI_NOTE_TO_FINE_LABEL.keys())
FINE_LABELS = [FINE_MIDI_NOTE_TO_FINE_LABEL[n] for n in FINE_NOTES]
FINE_LABEL_TO_NOTE = {
    label: note for note, label in FINE_MIDI_NOTE_TO_FINE_LABEL.items()
}


def _read_midi_manifest(
    sources: Union[str, Path, Sequence[Union[str, Path]]],
    relative_path: Optional[Union[str, Path]] = None,
) -> tuple[List[List[Dict]], List[bool]]:
    rel = None if relative_path in [None, ""] else Path(relative_path).expanduser()
    csv_paths = [sources] if isinstance(sources, (str, Path)) else list(sources)

    per_source_rows = []
    kept_mask = []
    for cpath in csv_paths:
        cpath = Path(cpath).expanduser()
        rows = []
        with open(cpath, "r") as f:
            for row in csv.DictReader(f):
                txt = Path((row.get("midi_txt") or "").strip()).expanduser()
                idx = Path((row.get("midi_idx_npy") or "").strip()).expanduser()
                if not txt.is_absolute():
                    if rel is None:
                        raise ValueError(
                            "MIDI manifest contains relative cache paths but `midi_relative_path` was not provided."
                        )
                    txt = (rel / txt).expanduser()
                if not idx.is_absolute():
                    if rel is None:
                        raise ValueError(
                            "MIDI manifest contains relative cache paths but `midi_relative_path` was not provided."
                        )
                    idx = (rel / idx).expanduser()
                if not txt.is_file() or not idx.is_file():
                    continue

                rows.append(
                    {
                        "__manifest__": str(cpath),
                        "midi_txt": str(txt),
                        "midi_idx_npy": str(idx),
                        "midi_row": int(row["midi_row"]),
                        "source_midi": row.get("source_midi", ""),
                        "source_dataset": row.get("source_dataset", ""),
                        "n_events": int(row.get("n_events", 0) or 0),
                        "duration": float(row.get("duration", 0.0) or 0.0),
                        "duration_steps": int(row.get("duration_steps", 0) or 0),
                        "meta": {
                            k: v
                            for k, v in row.items()
                            if k
                            not in {
                                "midi_txt",
                                "midi_idx_npy",
                                "midi_row",
                                "source_midi",
                                "source_dataset",
                                "n_events",
                                "duration",
                                "duration_steps",
                            }
                        },
                    }
                )
        if rows:
            per_source_rows.append(rows)
            kept_mask.append(True)
        else:
            kept_mask.append(False)

    return per_source_rows, kept_mask


def _read_oneshot_manifest(
    sources: Union[str, Path, Sequence[Union[str, Path]]],
    relative_path: Optional[Union[str, Path]] = None,
) -> tuple[List[List[Dict]], List[bool]]:
    rel = None if relative_path in [None, ""] else Path(relative_path).expanduser()
    csv_paths = [sources] if isinstance(sources, (str, Path)) else list(sources)

    per_source_rows = []
    kept_mask = []
    for cpath in csv_paths:
        cpath = Path(cpath).expanduser()
        rows = []
        with open(cpath, "r") as f:
            for row in csv.DictReader(f):
                raw = (row.get("oneshot") or "").strip()
                coarse_label = (row.get("coarse_label") or "").strip()
                if (
                    not raw
                    or not coarse_label
                    or coarse_label not in COARSE_LABEL_TO_NOTE
                ):
                    continue

                packed_audio_path = (row.get("packed_audio_path") or "").strip()
                if packed_audio_path:
                    packed_path = Path(packed_audio_path).expanduser()
                    if not packed_path.is_absolute():
                        if rel is None:
                            raise ValueError(
                                "One-shot manifest contains relative packed audio paths but `oneshot_relative_path` was not provided."
                            )
                        packed_path = (rel / packed_path).expanduser()
                    packed_audio_path = str(packed_path)

                path = Path(raw).expanduser()
                if not path.is_absolute():
                    if rel is None:
                        raise ValueError(
                            "One-shot manifest contains relative paths but `oneshot_relative_path` was not provided."
                        )
                    resolved = (rel / path).expanduser()
                    if resolved.is_file():
                        path = resolved
                    else:
                        alt = (rel / "audio" / path).expanduser()
                        path = alt if alt.is_file() else resolved
                if not path.is_file() and not packed_audio_path:
                    continue

                rows.append(
                    {
                        "__manifest__": str(cpath),
                        "path": str(path),
                        "packed_audio_path": packed_audio_path,
                        "packed_audio_offset": int(
                            row.get("packed_audio_offset", 0) or 0
                        ),
                        "packed_audio_num_frames": int(
                            row.get("packed_audio_num_frames", 0) or 0
                        ),
                        "packed_audio_num_channels": int(
                            row.get("packed_audio_num_channels", 0) or 0
                        ),
                        "packed_audio_sample_rate": int(
                            row.get("packed_audio_sample_rate", 0) or 0
                        ),
                        "packed_audio_dtype": (
                            row.get("packed_audio_dtype") or "float32"
                        ).strip(),
                        "fine_label": (row.get("fine_label") or "").strip(),
                        "coarse_label": coarse_label,
                        "midi_fine_note": int(row.get("midi_fine_note", -1) or -1),
                        "midi_coarse_note": int(row.get("midi_coarse_note", -1) or -1),
                        "kit_name": (row.get("kit_name") or "").strip(),
                        "source_dataset": (row.get("source_dataset") or "").strip(),
                        "source_family": (row.get("source_family") or "").strip(),
                        "meta": {
                            k: v
                            for k, v in row.items()
                            if k
                            not in {
                                "oneshot",
                                "label",
                                "fine_label",
                                "coarse_label",
                                "midi_fine_note",
                                "midi_coarse_note",
                                "kit_name",
                                "source_dataset",
                                "source_family",
                            }
                        },
                    }
                )

        if rows:
            per_source_rows.append(rows)
            kept_mask.append(True)
        else:
            kept_mask.append(False)

    return per_source_rows, kept_mask


class OneShotMidiDataset(Dataset):
    """
    Joint MIDI / one-shot dataset.

    Design intent
    -------------
    - Primary sampling axis is MIDI excerpts.
    - `__getitem__()` loads one MIDI excerpt plus a fixed-key candidate set of
      one-shots keyed either by coarse groups (`hierarchy_level="coarse"`) or
      exact fine label (`hierarchy_level="fine"`).
    - In coarse mode, the returned MIDI event note stream is coarseized to
      match the one-shot group resolution.
    - In fine mode, the returned MIDI event note stream remains at fine-note
      resolution and one-shot candidates are requested by exact fine note.
    - Multiple variations for a group are packed into the channel dimension of
      a single AudioSignal:
        `(1, num_channels * n_variations, n_samples)`
    - Mapping / fallback / dropping policy is deferred to an external
      batch-builder.
    """

    def __init__(
        self,
        midi_sources: Union[str, Path, Sequence[Union[str, Path]]] = None,
        oneshot_sources: Union[str, Path, Sequence[Union[str, Path]]] = None,
        midi_source_weights: Optional[Sequence[float]] = None,
        oneshot_source_weights: Optional[Sequence[float]] = None,
        sample_rate: int = None,
        duration: float = None,
        max_midi_events: Optional[int] = None,
        max_midi_duration: Optional[float] = None,
        n_examples: int = 1000,
        num_channels: int = 1,
        n_variations: int = 1,
        hierarchy_level: str = "coarse",
        midi_relative_path: Optional[Union[str, Path]] = None,
        oneshot_relative_path: Optional[Union[str, Path]] = None,
        oneshot_with_replacement: bool = True,
        audio_backend: str = "file",
        midi_max_tries: int = 4,
        shuffle_state: int = 0,
        p_same_kit: float = 0.0,
        same_kit_max_tries: int = 4,
        same_kit_min_coarse_classes: int = 3,
        from_start: bool = True,
        loudness_cutoff: Optional[float] = None,
        salience_num_tries: int = 4,
    ):
        super().__init__()

        self.midi_sources = midi_sources
        self.oneshot_sources = oneshot_sources
        self.midi_source_weights = midi_source_weights
        self.oneshot_source_weights = oneshot_source_weights
        self.sample_rate = int(sample_rate) if sample_rate is not None else None
        self.duration = float(duration) if duration is not None else None
        self.max_midi_events = None if max_midi_events is None else int(max_midi_events)
        self.max_midi_duration = (
            None if max_midi_duration is None else float(max_midi_duration)
        )
        self.length = int(n_examples)
        self.num_channels = int(num_channels)
        self.n_variations = int(n_variations)
        self.hierarchy_level = str(hierarchy_level)
        self.midi_relative_path = (
            None if midi_relative_path in [None, ""] else str(midi_relative_path)
        )
        self.oneshot_relative_path = (
            None if oneshot_relative_path in [None, ""] else str(oneshot_relative_path)
        )
        self.oneshot_with_replacement = bool(oneshot_with_replacement)
        self.audio_backend = str(audio_backend)
        if self.audio_backend not in {"file", "packed"}:
            raise ValueError(
                f"`audio_backend` must be 'file' or 'packed', got {audio_backend!r}."
            )
        self.midi_max_tries = int(midi_max_tries)
        self.shuffle_state = int(shuffle_state)
        self.p_same_kit = float(p_same_kit)
        self.same_kit_max_tries = int(same_kit_max_tries)
        self.same_kit_min_coarse_classes = int(same_kit_min_coarse_classes)
        self.from_start = bool(from_start)
        self.loudness_cutoff = loudness_cutoff
        self.salience_num_tries = int(salience_num_tries)

        if self.n_variations <= 0:
            raise ValueError("`n_variations` must be positive.")
        if self.midi_max_tries <= 0:
            raise ValueError("`midi_max_tries` must be positive.")
        if self.hierarchy_level not in {"coarse", "fine"}:
            raise ValueError("`hierarchy_level` must be 'coarse' or 'fine'.")
        if not (0.0 <= self.p_same_kit <= 1.0):
            raise ValueError("`p_same_kit` must be in [0, 1].")
        if self.same_kit_max_tries < 0:
            raise ValueError("`same_kit_max_tries` must be >= 0.")
        if self.same_kit_min_coarse_classes < 0:
            raise ValueError("`same_kit_min_coarse_classes` must be >= 0.")
        if self.midi_sources is None:
            raise ValueError("`midi_sources` must be provided.")
        if self.sample_rate is None or self.duration is None:
            raise ValueError("`sample_rate` and `duration` must be provided.")

        self.midi_rows, kept_mask = _read_midi_manifest(
            self.midi_sources,
            relative_path=self.midi_relative_path,
        )
        if len(self.midi_rows) == 0:
            raise RuntimeError(
                "OneShotMidiDataset: no valid MIDI rows found in manifest(s)."
            )

        midi_csv_paths = (
            [self.midi_sources]
            if isinstance(self.midi_sources, (str, Path))
            else list(self.midi_sources)
        )
        weights = normalize_source_weights(
            source_weights=self.midi_source_weights,
            n_sources=len(midi_csv_paths),
            kept_mask=kept_mask,
            source_lengths=[len(rows) for rows in self.midi_rows],
        )
        self._midi_source_weights = np.asarray(weights, dtype=float)
        self._midi_indices: Dict[tuple, DrumEventTextIndex] = {}

        self.oneshot_rows = []
        self._oneshot_source_weights = None
        self._oneshot_rows_by_group = []
        self._oneshot_rows_by_kit_group = []
        self._oneshot_kits = []
        self._oneshot_kit_coarse_counts = []
        self._available_group_labels = set()

        if self.hierarchy_level == "coarse":
            self.group_labels = COARSE_LABELS
            self.group_label_to_note = COARSE_LABEL_TO_NOTE
        else:
            self.group_labels = FINE_LABELS
            self.group_label_to_note = FINE_LABEL_TO_NOTE

        if self.oneshot_sources is not None:
            self.oneshot_rows, kept_mask = _read_oneshot_manifest(
                self.oneshot_sources,
                relative_path=self.oneshot_relative_path,
            )
            if len(self.oneshot_rows) == 0:
                raise RuntimeError(
                    "OneShotMidiDataset: no valid one-shot rows found in manifest(s)."
                )
            if self.audio_backend == "packed":
                missing = [
                    row["path"]
                    for rows in self.oneshot_rows
                    for row in rows
                    if not row["packed_audio_path"]
                ]
                if missing:
                    raise RuntimeError(
                        "OneShotMidiDataset: packed audio backend requested but "
                        "one-shot manifest rows are missing packed audio metadata."
                    )

            oneshot_csv_paths = (
                [self.oneshot_sources]
                if isinstance(self.oneshot_sources, (str, Path))
                else list(self.oneshot_sources)
            )
            weights = normalize_source_weights(
                source_weights=self.oneshot_source_weights,
                n_sources=len(oneshot_csv_paths),
                kept_mask=kept_mask,
                source_lengths=[len(rows) for rows in self.oneshot_rows],
            )
            self._oneshot_source_weights = np.asarray(weights, dtype=float)
            self._build_oneshot_indices()

    def _row_group_label(self, row: Dict) -> Optional[str]:
        if self.hierarchy_level == "coarse":
            return row["coarse_label"]

        fine_note = int(row["midi_fine_note"])
        if fine_note < 0 or fine_note not in FINE_MIDI_NOTE_TO_FINE_LABEL:
            return None
        return FINE_MIDI_NOTE_TO_FINE_LABEL[fine_note]

    def _build_oneshot_indices(self):
        self._oneshot_rows_by_group = []
        self._oneshot_rows_by_kit_group = []
        self._oneshot_kits = []
        self._oneshot_kit_coarse_counts = []
        self._available_group_labels = set()

        for rows in self.oneshot_rows:
            by_group = {label: [] for label in self.group_labels}
            by_kit_group: Dict[str, Dict[str, List[Dict]]] = {}
            kits = set()
            coarse_by_kit: Dict[str, set[str]] = {}

            for row in rows:
                label = self._row_group_label(row)
                if label is None:
                    continue

                by_group[label].append(row)
                self._available_group_labels.add(label)
                kit = row["kit_name"]
                if not kit:
                    continue

                kits.add(kit)
                by_kit_group.setdefault(kit, {lab: [] for lab in self.group_labels})
                by_kit_group[kit][label].append(row)
                coarse_by_kit.setdefault(kit, set()).add(row["coarse_label"])

            coarse_counts = {kit: len(labels) for kit, labels in coarse_by_kit.items()}

            self._oneshot_rows_by_group.append(by_group)
            self._oneshot_rows_by_kit_group.append(by_kit_group)
            self._oneshot_kits.append(sorted(kits))
            self._oneshot_kit_coarse_counts.append(coarse_counts)

    def __len__(self):
        return self.length

    def _sample_midi_source(self, state: np.random.RandomState):
        source_idx = int(state.choice(len(self.midi_rows), p=self._midi_source_weights))
        rows = self.midi_rows[source_idx]
        row_idx = int(state.randint(len(rows)))
        return source_idx, row_idx, rows[row_idx]

    def _load_midi_excerpt(self, idx: int, state: np.random.RandomState) -> Dict:
        _source_idx, source_row_idx, row = self._sample_midi_source(state)
        key = (row["midi_txt"], row["midi_idx_npy"])
        if key not in self._midi_indices:
            self._midi_indices[key] = DrumEventTextIndex(*key)

        events = self._midi_indices[key].read_row(int(row["midi_row"]))
        events = self._postprocess_midi_excerpt(events, state=state)

        if len(events) == 0:
            events = np.empty((0, 3), dtype=np.int32)
        else:
            fine_notes = events[:, 1].astype(np.int64, copy=False)
            coarse_notes = fine_to_coarse_notes(fine_notes)
            if np.any(coarse_notes < 0):
                raise ValueError("Encountered fine MIDI notes without coarse mapping.")
            if self.hierarchy_level == "coarse":
                events = events.copy()
                events[:, 1] = coarse_notes.astype(events.dtype, copy=False)

        return {
            "events": torch.from_numpy(
                events.astype(np.int64, copy=False).T[None, ...]
            ),
            "source_midi": row["source_midi"],
            "source_dataset": row["source_dataset"],
            "source_manifest": row["__manifest__"],
            "midi_txt": row["midi_txt"],
            "midi_idx_npy": row["midi_idx_npy"],
            "midi_row": int(row["midi_row"]),
            "source_row": int(source_row_idx),
        }

    def _postprocess_midi_excerpt(
        self,
        events: np.ndarray,
        state: Optional[np.random.RandomState] = None,
    ) -> np.ndarray:
        max_duration_steps = None
        if self.max_midi_duration is not None:
            max_duration_steps = int(round(MIDI_TIME_RES * self.max_midi_duration))
        return crop_drum_event_excerpt(
            events,
            max_events=self.max_midi_events,
            max_duration_steps=max_duration_steps,
            state=state,
        )

    def _sample_same_kit(self, state: np.random.RandomState) -> bool:
        return bool(state.rand() < self.p_same_kit)

    def _requested_group_labels(self, midi_excerpt: Dict) -> set[str]:
        events = midi_excerpt["events"]
        if events.shape[-1] == 0:
            return set()
        label_map = (
            COARSE_MIDI_NOTE_TO_COARSE_LABEL
            if self.hierarchy_level == "coarse"
            else FINE_MIDI_NOTE_TO_FINE_LABEL
        )
        return {
            label_map[int(note)]
            for note in events[0, 1].tolist()
            if int(note) in label_map
        }

    def _empty_signal(self) -> AudioSignal:
        n_samples = int(round(self.sample_rate * self.duration))
        x = torch.zeros(1, self.num_channels * self.n_variations, n_samples)
        sig = AudioSignal(x, sample_rate=self.sample_rate)
        sig.metadata = {}
        return sig

    def _fit_signal_length(self, sig: AudioSignal) -> AudioSignal:
        target = int(round(self.sample_rate * self.duration))
        x = sig.audio_data
        if x.shape[-1] < target:
            x = F.pad(x, (0, target - x.shape[-1]))
        elif x.shape[-1] > target:
            x = x[..., :target]
        out = AudioSignal(x, sample_rate=sig.sample_rate)
        out.metadata = dict(getattr(sig, "metadata", {}) or {})
        return out

    def _sample_rows(
        self, rows: List[Dict], state: np.random.RandomState
    ) -> List[Dict]:
        if len(rows) == 0:
            return []
        if self.oneshot_with_replacement:
            idxs = state.randint(len(rows), size=self.n_variations)
            return [rows[int(i)] for i in idxs]

        if len(rows) >= self.n_variations:
            idxs = state.permutation(len(rows))[: self.n_variations]
            return [rows[int(i)] for i in idxs]

        idxs = list(state.permutation(len(rows)))
        if idxs:
            extra = state.choice(idxs, size=self.n_variations - len(idxs), replace=True)
            idxs.extend(int(i) for i in extra)
        return [rows[int(i)] for i in idxs]

    def _load_packed_group_signal(
        self,
        rows: List[Dict],
        state: np.random.RandomState,
        group_idx: int,
    ) -> Dict:
        group_label = self.group_labels[group_idx]
        group_note = self.group_label_to_note[group_label]
        packed = []
        fine_notes = []
        valid = []

        if len(rows) == 0:
            sig = self._empty_signal()
            if self.hierarchy_level == "coarse":
                fine_notes = torch.full((self.n_variations,), -1, dtype=torch.long)
                coarse_notes = torch.full(
                    (self.n_variations,), group_note, dtype=torch.long
                )
            else:
                fine_notes = torch.full(
                    (self.n_variations,), group_note, dtype=torch.long
                )
                coarse_note = int(fine_to_coarse_notes(np.asarray([group_note]))[0])
                coarse_notes = torch.full(
                    (self.n_variations,), coarse_note, dtype=torch.long
                )
            valid = torch.zeros(self.n_variations, dtype=torch.bool)
            return {
                "signal": sig,
                "midi_fine_note": fine_notes,
                "midi_coarse_note": coarse_notes,
                "valid": valid,
            }

        sampled = self._sample_rows(rows, state)
        for j, row in enumerate(sampled):
            load_state = np.random.RandomState(
                (self.shuffle_state + int(state.randint(2**31 - 1)) + 104729 * j)
                & 0x7FFFFFFF
            )
            sig, offset = load_excerpt(
                row["path"],
                duration=self.duration,
                sample_rate=self.sample_rate,
                state=load_state,
                from_start=self.from_start,
                loudness_cutoff=self.loudness_cutoff,
                num_tries=self.salience_num_tries,
                num_channels=self.num_channels,
                packed_audio_path=(
                    row["packed_audio_path"] if self.audio_backend == "packed" else None
                ),
                packed_audio_offset=row["packed_audio_offset"],
                packed_audio_num_frames=row["packed_audio_num_frames"],
                packed_audio_num_channels=row["packed_audio_num_channels"],
                packed_audio_sample_rate=row["packed_audio_sample_rate"],
                packed_audio_dtype=row["packed_audio_dtype"],
            )
            sig = self._fit_signal_length(sig)
            sig.metadata["path"] = row["path"]
            sig.metadata["offset"] = offset
            sig.metadata["fine_label"] = row["fine_label"]
            sig.metadata["coarse_label"] = row["coarse_label"]
            sig.metadata["kit_name"] = row["kit_name"]
            packed.append(sig.audio_data)
            fine_notes.append(int(row["midi_fine_note"]))
            valid.append(True)

        x = torch.cat(packed, dim=1)
        sig = AudioSignal(x, sample_rate=self.sample_rate)
        sig.metadata = {
            "hierarchy_level": self.hierarchy_level,
            "group_label": group_label,
        }
        fine_notes_t = torch.tensor(fine_notes, dtype=torch.long)
        if self.hierarchy_level == "coarse":
            coarse_notes_t = torch.full(
                (self.n_variations,), group_note, dtype=torch.long
            )
        else:
            coarse_note = int(fine_to_coarse_notes(np.asarray([group_note]))[0])
            coarse_notes_t = torch.full(
                (self.n_variations,), coarse_note, dtype=torch.long
            )
        return {
            "signal": sig,
            "midi_fine_note": fine_notes_t,
            "midi_coarse_note": coarse_notes_t,
            "valid": torch.tensor(valid, dtype=torch.bool),
        }

    def _sample_same_kit_source(self, state: np.random.RandomState) -> tuple[int, str]:
        if self._oneshot_source_weights is None:
            return -1, ""
        source_idx = int(
            state.choice(len(self.oneshot_rows), p=self._oneshot_source_weights)
        )
        kits = [
            kit
            for kit in self._oneshot_kits[source_idx]
            if self._oneshot_kit_coarse_counts[source_idx].get(kit, 0)
            >= self.same_kit_min_coarse_classes
        ]
        if len(kits) == 0:
            return source_idx, ""
        kit_idx = int(state.randint(len(kits)))
        return source_idx, kits[kit_idx]

    def _same_kit_has_requested_coverage(
        self,
        source_idx: int,
        kit_name: str,
        requested_labels: set[str],
    ) -> bool:
        if source_idx < 0 or not kit_name or len(requested_labels) == 0:
            return False
        by_group = self._oneshot_rows_by_kit_group[source_idx].get(kit_name, {})
        return any(len(by_group.get(label, [])) > 0 for label in requested_labels)

    def _sample_free_group_rows(
        self,
        group_label: str,
        state: np.random.RandomState,
    ) -> List[Dict]:
        if self._oneshot_source_weights is None:
            return []
        available = [
            i
            for i, by_group in enumerate(self._oneshot_rows_by_group)
            if len(by_group[group_label]) > 0
        ]
        if len(available) == 0:
            return []

        weights = self._oneshot_source_weights[available]
        weights = weights / weights.sum()
        choice = int(state.choice(len(available), p=weights))
        return self._oneshot_rows_by_group[available[choice]][group_label]

    def _sample_oneshot_candidates(
        self,
        idx: int,
        midi_excerpt: Dict,
        state: np.random.RandomState,
        same_kit: bool,
    ) -> Dict:
        requested_labels = self._requested_group_labels(midi_excerpt)

        if len(self.oneshot_rows) == 0:
            return {
                "same_kit": bool(same_kit),
                "kit_name": "",
                "groups": {
                    label: self._load_packed_group_signal([], state, i)
                    for i, label in enumerate(self.group_labels)
                },
            }

        source_idx, kit_name = (-1, "")
        use_same_kit = False
        if same_kit and len(requested_labels) > 0:
            for _ in range(self.same_kit_max_tries):
                cand_source_idx, cand_kit_name = self._sample_same_kit_source(state)
                if self._same_kit_has_requested_coverage(
                    cand_source_idx,
                    cand_kit_name,
                    requested_labels,
                ):
                    source_idx, kit_name = cand_source_idx, cand_kit_name
                    use_same_kit = True
                    break

        groups = {}
        for i, group_label in enumerate(self.group_labels):
            group_state = np.random.RandomState(
                (self.shuffle_state + int(idx) + 9973 * (i + 1)) & 0x7FFFFFFF
            )
            if group_label not in requested_labels:
                rows = []
            elif use_same_kit:
                rows = self._oneshot_rows_by_kit_group[source_idx].get(
                    kit_name, {lab: [] for lab in self.group_labels}
                )[group_label]
            else:
                rows = self._sample_free_group_rows(group_label, group_state)

            groups[group_label] = self._load_packed_group_signal(
                rows=rows,
                state=group_state,
                group_idx=i,
            )

        return {
            "same_kit": bool(use_same_kit),
            "kit_name": kit_name if use_same_kit else "",
            "groups": groups,
        }

    def __getitem__(self, idx: int):
        state = np.random.RandomState((self.shuffle_state + int(idx)) & 0x7FFFFFFF)
        midi_excerpt = None
        for _ in range(self.midi_max_tries):
            midi_excerpt = self._load_midi_excerpt(idx=idx, state=state)
            if (
                len(self._available_group_labels) == 0
                or len(
                    self._requested_group_labels(midi_excerpt)
                    & self._available_group_labels
                )
                > 0
            ):
                break
        same_kit = self._sample_same_kit(state)
        oneshot_candidates = self._sample_oneshot_candidates(
            idx=idx,
            midi_excerpt=midi_excerpt,
            state=state,
            same_kit=same_kit,
        )

        return {
            "hierarchy_level": self.hierarchy_level,
            "midi": midi_excerpt,
            "oneshots": oneshot_candidates,
            "idx": idx,
        }

    @staticmethod
    def collate(list_of_dicts: Union[list, dict], n_splits: int = None):
        return collate(list_of_dicts, n_splits=n_splits)


@torch.no_grad()
def build_one_shot_midi_batch(
    batch: Dict,
    oneshot_transform: Optional[Callable] = None,
    mix_transform: Optional[Callable] = None,
    velocity_to_gain: Optional[Callable] = None,
    coarse_loudness_ranges: Optional[
        Dict[int, tuple[float, float]]
    ] = DEFAULT_COARSE_LOUDNESS_RANGES,
    p_normalize_loudness: float = 1.0,
    p_speed: float = 0.0,
    speed_ratio: tuple = ("uniform", 0.9, 1.1),
    p_timing_jitter: float = 0.0,
    timing_jitter_std: tuple = ("uniform", 0.0, 0.0),
    p_velocity_jitter: float = 0.0,
    velocity_jitter_std: tuple = ("uniform", 0.0, 0.0),
    p_time_shift: float = 1.0,
    time_shift: tuple = ("uniform", 0.0, 0.25),
    max_duration: Optional[float] = None,
    state: Optional[
        Union[int, Sequence[int], torch.Tensor, np.random.RandomState]
    ] = None,
):
    """
    Prepare a collated `OneShotMidiDataset` batch for later synthesis.

    Current implementation does:
    - normalize per-example RNG
    - drop MIDI events whose one-shot group has no valid variation
    - apply optional speed / timing-jitter / velocity-jitter MIDI augmentation
    - apply an optional random global time shift last
    - optionally trim by `max_duration`
    - optionally apply a one-shot transform by folding packed variations into
      batch, transforming, and unpacking
    - sample one valid variation index per surviving MIDI event
    - synthesize a first-pass audio mixture by placing selected one-shots on a
      sample-time canvas and summing overlaps
    - optionally apply `mix_transform` to the realized mixture

    The main remaining work is target formatting and any refinement of the
    gain/mixing policy.
    """
    batch_size = int(batch["midi"]["events"].shape[0])
    midi_events = batch["midi"]["events"]
    midi_event_lengths = batch["midi"].get("events_lengths")
    if midi_event_lengths is None:
        midi_event_lengths = torch.full(
            (batch_size,), midi_events.shape[-1], dtype=torch.long
        )

    hierarchy_level = batch["hierarchy_level"]
    if isinstance(hierarchy_level, Sequence) and not isinstance(hierarchy_level, str):
        hierarchy_level = hierarchy_level[0]
    hierarchy_level = str(hierarchy_level)
    if hierarchy_level not in {"coarse", "fine"}:
        raise ValueError("Expected `hierarchy_level` to be 'coarse' or 'fine'.")
    if coarse_loudness_ranges is not None:
        bad = sorted(
            set(coarse_loudness_ranges) - set(COARSE_MIDI_NOTE_TO_COARSE_LABEL)
        )
        if bad:
            raise ValueError(
                f"`coarse_loudness_ranges` contains unknown coarse MIDI notes: {bad}"
            )

    for p, name in [
        (p_normalize_loudness, "p_normalize_loudness"),
        (p_speed, "p_speed"),
        (p_timing_jitter, "p_timing_jitter"),
        (p_velocity_jitter, "p_velocity_jitter"),
        (p_time_shift, "p_time_shift"),
    ]:
        if not (0.0 <= p <= 1.0):
            raise ValueError(f"`{name}` must be in [0, 1].")

    if isinstance(state, np.random.RandomState):
        seeds = state.randint(0, 2**31 - 1, size=batch_size).tolist()
    elif isinstance(state, torch.Tensor):
        seeds = state.reshape(-1).tolist()
    elif isinstance(state, Sequence) and not isinstance(state, (str, bytes)):
        seeds = list(state)
    elif state is None:
        seeds = list(range(batch_size))
    else:
        seeds = [int(state)]
    if len(seeds) == 1 and batch_size > 1:
        seeds = seeds * batch_size
    if len(seeds) != batch_size:
        raise ValueError("`state` must provide one seed or one seed per batch item.")

    label_map = (
        COARSE_MIDI_NOTE_TO_COARSE_LABEL
        if hierarchy_level == "coarse"
        else FINE_MIDI_NOTE_TO_FINE_LABEL
    )
    max_duration_sec = None if max_duration is None else float(max_duration)

    groups = batch["oneshots"]["groups"]

    def _stable_seed(*parts) -> int:
        h = hashlib.sha1("|".join(str(p) for p in parts).encode("utf-8")).digest()
        return int.from_bytes(h[:4], "little") & 0x7FFFFFFF

    if velocity_to_gain is None:
        velocity_to_gain = lambda v: v.to(torch.float32) / 127.0

    realized = []
    if max_duration_sec is not None and max_duration_sec < 0:
        raise ValueError("`max_duration` must be non-negative.")
    max_duration_samples = (
        None
        if max_duration is None
        else int(
            round(
                float(max_duration) * groups[next(iter(groups))]["signal"].sample_rate
            )
        )
    )

    for i in range(batch_size):
        seed_i = int(seeds[i])
        state_i = random_state(seed_i)
        item_idx = (
            torch.tensor(int(batch["idx"][i]), dtype=torch.long)
            if "idx" in batch
            else torch.tensor(i, dtype=torch.long)
        )

        n_events = int(midi_event_lengths[i].item())
        events_i = midi_events[i : i + 1, :, :n_events].clone()
        notes_i = events_i[0, 1].to(dtype=torch.long)
        keep_mask = torch.tensor(
            [
                bool(groups[label_map[int(note)]]["valid"][i].any().item())
                if int(note) in label_map
                else False
                for note in notes_i.tolist()
            ],
            dtype=torch.bool,
            device=events_i.device,
        )
        events_i = events_i[..., keep_mask]

        onsets_sec = events_i[0, 0].to(torch.float32) / float(MIDI_TIME_RES)
        notes = events_i[0, 1].to(torch.long)
        velocities = events_i[0, 2].to(torch.float32)

        if events_i.shape[-1] > 0:
            if float(state_i.rand()) < p_speed:
                onsets_sec = onsets_sec * float(sample_from_dist(speed_ratio, state_i))

            if float(state_i.rand()) < p_timing_jitter:
                std_sec = float(sample_from_dist(timing_jitter_std, state_i))
                if std_sec > 0:
                    onsets_sec = onsets_sec + torch.as_tensor(
                        state_i.normal(0.0, std_sec, size=onsets_sec.shape[0]),
                        device=onsets_sec.device,
                        dtype=onsets_sec.dtype,
                    )

            if float(state_i.rand()) < p_velocity_jitter:
                std_vel = float(sample_from_dist(velocity_jitter_std, state_i))
                if std_vel > 0:
                    velocities = velocities + torch.as_tensor(
                        state_i.normal(0.0, std_vel, size=velocities.shape[0]),
                        device=velocities.device,
                        dtype=velocities.dtype,
                    )

            onsets_sec = onsets_sec.clamp_min_(0.0)
            velocities = torch.round(velocities).clamp_(1, 127).to(torch.long)
            order = torch.argsort(notes, stable=True)
            onsets_sec = onsets_sec[order]
            notes = notes[order]
            velocities = velocities[order]
            order = torch.argsort(onsets_sec, stable=True)
            onsets_sec = onsets_sec[order]
            notes = notes[order]
            velocities = velocities[order]

        time_shift_sec = 0.0
        if events_i.shape[-1] > 0 and float(state_i.rand()) < p_time_shift:
            time_shift_sec = float(sample_from_dist(time_shift, state_i))
            if time_shift_sec:
                onsets_sec = onsets_sec + time_shift_sec

        if max_duration_sec is not None and events_i.shape[-1] > 0:
            keep = onsets_sec <= max_duration_sec
            onsets_sec = onsets_sec[keep]
            notes = notes[keep]
            velocities = velocities[keep]

        notes_i = notes.to(dtype=torch.long)
        variation_idx_i = torch.empty(
            notes_i.shape[0], dtype=torch.long, device=events_i.device
        )
        for j, note in enumerate(notes_i.tolist()):
            valid = groups[label_map[int(note)]]["valid"][i]
            choices = torch.nonzero(valid, as_tuple=False).flatten()
            if len(choices) == 0:
                raise RuntimeError("Invalid MIDI event survived pruning.")
            variation_idx_i[j] = int(state_i.choice(choices.cpu().numpy()))

        used_group_labels = {label_map[int(note)] for note in notes.tolist()}

        group_cache = {}
        for group_label in used_group_labels:
            group = groups[group_label]
            signal = group["signal"]
            valid = group["valid"][i : i + 1]
            fine_note = group["midi_fine_note"][i : i + 1]
            coarse_note = group["midi_coarse_note"][i : i + 1]

            x = signal.audio_data[i : i + 1]
            n_variations = int(valid.shape[-1])
            n_channels = x.shape[1] // n_variations
            x = x.reshape(1, n_variations, n_channels, x.shape[-1])
            folded = AudioSignal(
                x.reshape(-1, n_channels, x.shape[-1]),
                sample_rate=signal.sample_rate,
            )
            folded.metadata = dict(getattr(signal, "metadata", {}) or {})
            if oneshot_transform is None:
                transformed = folded
            else:
                tfm_seed = _stable_seed(seed_i, group_label, "oneshot")
                tfm_seeds = [tfm_seed] * n_variations
                kwargs = oneshot_transform.batch_instantiate(
                    tfm_seeds,
                    folded.clone(),
                )
                transformed = oneshot_transform.transform(folded.clone(), **kwargs)
            xt = transformed.audio_data.reshape(
                1,
                n_variations * n_channels,
                transformed.audio_data.shape[-1],
            )
            out_signal = AudioSignal(xt, sample_rate=transformed.sample_rate)
            out_signal.metadata = dict(getattr(transformed, "metadata", {}) or {})
            group_cache[group_label] = {
                "signal": out_signal,
                "valid": valid,
                "midi_fine_note": fine_note,
                "midi_coarse_note": coarse_note,
            }

        if notes.shape[0] == 0:
            first_group = next(iter(group_cache.values()), next(iter(groups.values())))
            sr = first_group["signal"].sample_rate
            base_channels = (
                first_group["signal"].audio_data.shape[1]
                // first_group["valid"].shape[-1]
            )
            audio = torch.zeros(
                1,
                base_channels,
                1,
                dtype=first_group["signal"].audio_data.dtype,
                device=first_group["signal"].audio_data.device,
            )
            sig = AudioSignal(audio, sample_rate=sr)
            sig.metadata = {"hierarchy_level": hierarchy_level}
            target_oneshots = AudioSignal(audio.clone(), sample_rate=sr)
            if mix_transform is not None:
                kwargs = mix_transform.batch_instantiate(
                    [_stable_seed(seed_i, "mix")], sig.clone()
                )
                sig = mix_transform.transform(sig.clone(), **kwargs)
            realized.append(
                {
                    "signal": sig,
                    "arrangement": {
                        "notes": notes[None, None, :],
                        "onsets_sec": onsets_sec[None, None, :],
                        "velocities": velocities[None, None, :],
                        "variation_idx": variation_idx_i[None, None, :],
                        "time_shift_sec": torch.tensor(
                            time_shift_sec, dtype=torch.float32
                        ),
                    },
                    "targets": {
                        "oneshots": target_oneshots,
                        "oneshot_num_samples": torch.tensor(0, dtype=torch.long),
                    },
                    "idx": item_idx,
                }
            )
            continue

        velocities = velocities.to(dtype=torch.long)
        do_loudness_norm = (
            coarse_loudness_ranges is not None
            and float(state_i.rand()) < p_normalize_loudness
        )

        sr = group_cache[next(iter(group_cache))]["signal"].sample_rate
        onset_samples = torch.round(onsets_sec * float(sr)).to(torch.long)

        loudness_targets = {}
        tail_ends = []
        selected = []
        for j, note in enumerate(notes.tolist()):
            group_label = label_map[int(note)]
            group = group_cache[group_label]
            signal = group["signal"].audio_data
            n_variations = int(group["valid"].shape[-1])
            n_channels = signal.shape[1] // n_variations
            v = int(variation_idx_i[j].item())
            clip = signal[:, v * n_channels : (v + 1) * n_channels]

            gain = velocity_to_gain(velocities[j : j + 1])
            gain = torch.as_tensor(gain, device=clip.device, dtype=clip.dtype).reshape(
                -1
            )[0]
            coarse_note = int(group["midi_coarse_note"][0, 0].item())
            if do_loudness_norm and coarse_note in coarse_loudness_ranges:
                if group_label not in loudness_targets:
                    lo, hi = coarse_loudness_ranges[coarse_note]
                    loudness_targets[group_label] = float(
                        state_i.uniform(float(lo), float(hi))
                    )
                cur_db = 10.0 * torch.log10(_energy(clip)).reshape(-1)[0]
                gain_db = torch.as_tensor(
                    loudness_targets[group_label],
                    device=clip.device,
                    dtype=clip.dtype,
                ) - cur_db.to(dtype=clip.dtype)
                clip = clip * torch.pow(
                    torch.as_tensor(10.0, device=clip.device, dtype=clip.dtype),
                    gain_db / 20.0,
                )

            start = int(onset_samples[j].item())
            end = start + int(clip.shape[-1])
            tail_ends.append(end)
            selected.append((start, clip * gain))

        total_len = max(tail_ends) if tail_ends else 1
        if max_duration_samples is not None:
            total_len = min(total_len, max_duration_samples)
        total_len = max(1, int(total_len))

        mix = torch.zeros(
            1,
            selected[0][1].shape[1],
            total_len,
            dtype=selected[0][1].dtype,
            device=selected[0][1].device,
        )
        for start, clip in selected:
            if start >= total_len:
                continue
            clip = clip[..., : max(0, total_len - start)]
            if clip.shape[-1] == 0:
                continue
            mix[..., start : start + clip.shape[-1]] += clip

        sig = AudioSignal(mix, sample_rate=sr)
        sig.metadata = {"hierarchy_level": hierarchy_level}
        target_batch = AudioSignal(
            torch.cat([clip for _, clip in selected], dim=0),
            sample_rate=sr,
        )
        oneshot_num_samples = int(target_batch.signal_length)
        if mix_transform is not None:
            mix_seed = _stable_seed(seed_i, "mix")
            mix_ref = sig.clone()
            kwargs = mix_transform.batch_instantiate([mix_seed], mix_ref.clone())
            sig = mix_transform.transform(sig.clone(), **kwargs)
            target_kwargs = mix_transform.batch_instantiate(
                [mix_seed] * target_batch.batch_size,
                mix_ref.clone(),
            )
            target_batch = mix_transform.transform(
                target_batch.clone(), **target_kwargs
            )
        if not torch.isfinite(sig.audio_data).all():
            raise RuntimeError("build_one_shot_midi_batch: non-finite audio detected")
        if not torch.isfinite(target_batch.audio_data).all():
            raise RuntimeError(
                "build_one_shot_midi_batch: non-finite target audio detected"
            )
        peak = sig.audio_data.abs().amax()
        if torch.isfinite(peak) and peak.item() > 1.0:
            scale = peak.reciprocal()
            sig.audio_data = sig.audio_data * scale
            target_batch.audio_data = target_batch.audio_data * scale

        target_oneshots = AudioSignal(
            target_batch.audio_data.permute(1, 0, 2).reshape(
                1,
                target_batch.num_channels,
                target_batch.batch_size * target_batch.signal_length,
            ),
            sample_rate=sr,
        )

        realized.append(
            {
                "signal": sig,
                "arrangement": {
                    "notes": notes[None, None, :],
                    "onsets_sec": onsets_sec[None, None, :],
                    "velocities": velocities[None, None, :],
                    "variation_idx": variation_idx_i[None, None, :],
                    "time_shift_sec": torch.tensor(time_shift_sec, dtype=torch.float32),
                },
                "targets": {
                    "oneshots": target_oneshots,
                    "oneshot_num_samples": torch.tensor(
                        oneshot_num_samples, dtype=torch.long
                    ),
                },
                "idx": item_idx,
            }
        )

    return collate(realized)


@torch.no_grad()
def format_dcomposer_targets(
    batch: Dict,
    tokenizer,
    autoencoder=None,
    tokenizer_batch_size: int = 128,
    scale: bool = True,
    output_device=None,
) -> Dict:
    arrangement = batch["arrangement"]
    target_data = batch["targets"]

    notes = arrangement["notes"]
    onsets_sec = arrangement["onsets_sec"]
    event_lengths = arrangement["notes_lengths"]
    oneshots = target_data["oneshots"]
    oneshot_num_samples = target_data["oneshot_num_samples"]

    batch_size = notes.shape[0]
    tokenizer_batch_size = max(1, int(tokenizer_batch_size))
    output_device = (
        notes.device if output_device is None else torch.device(output_device)
    )

    flat_oneshots = []
    flat_meta = []
    for i in range(batch_size):
        n_events = int(event_lengths[i].item())
        n_samples = int(oneshot_num_samples[i].item())
        if n_events <= 0 or n_samples <= 0:
            continue
        audio_i = oneshots.audio_data[i : i + 1, :, : n_events * n_samples]
        for j in range(n_events):
            start = j * n_samples
            stop = start + n_samples
            flat_oneshots.append(
                AudioSignal(
                    audio_i[:, :, start:stop].clone(),
                    sample_rate=oneshots.sample_rate,
                )
            )
            flat_meta.append((i, j))

    codes_by_row = [[] for _ in range(batch_size)]
    if flat_oneshots:
        for start in range(0, len(flat_oneshots), tokenizer_batch_size):
            chunk = flat_oneshots[start : start + tokenizer_batch_size]
            chunk_sig = AudioSignal.batch(chunk, pad_signals=True)
            if autoencoder is not None:
                seq = tokenizer.encode(chunk_sig.clone(), scale=scale, no_grad=True)
                latents = seq.tokens.detach()
                if latents.ndim != 3:
                    raise ValueError(
                        f"Tokenizer latents must have ndim 3, got {latents.ndim}."
                    )
                max_len = int(autoencoder.max_len)
                if latents.shape[-1] > max_len:
                    raise ValueError(
                        "format_dcomposer_targets: tokenizer latent length exceeds "
                        f"autoencoder.max_len ({latents.shape[-1]} > {max_len})."
                    )
                latents = F.pad(
                    latents[..., :max_len], (0, max(0, max_len - latents.shape[-1]))
                )
                summary = autoencoder.encode(latents)
                _latents_prequant, _latents_quant, codes = autoencoder.quantize(summary)
            else:
                encoded = (
                    tokenizer.encode(chunk_sig)
                    if hasattr(tokenizer, "encode")
                    else tokenizer(chunk_sig)
                )
                codes = encoded.tokens if hasattr(encoded, "tokens") else encoded
                if not isinstance(codes, torch.Tensor):
                    codes = torch.as_tensor(codes)
                if codes.ndim == 3:
                    codes = codes.reshape(codes.shape[0], -1)
                elif codes.ndim != 2:
                    raise ValueError(
                        f"Tokenizer codes must have ndim 2 or 3, got {codes.ndim}."
                    )
            codes = codes.to(device=output_device, dtype=torch.long)
            for local_idx in range(codes.shape[0]):
                row_idx, _ = flat_meta[start + local_idx]
                codes_by_row[row_idx].append(codes[local_idx])

    seqs = []
    lengths = []
    for i in range(batch_size):
        n_events = int(event_lengths[i].item())
        toks = [BOS_TOK]
        if n_events > 0:
            notes_i = notes[i, 0, :n_events].to(device=output_device, dtype=torch.long)
            onsets_i = onsets_sec[i, 0, :n_events].to(
                device=output_device, dtype=torch.float32
            )
            if len(codes_by_row[i]) != n_events:
                raise RuntimeError(
                    "format_dcomposer_targets: one-shot/token count does not match event count."
                )
            onset_toks = torch.round(onsets_i * float(MIDI_TIME_RES)).to(torch.long)
            if torch.any(onset_toks < 0) or torch.any(onset_toks > MAX_ONSET_TOK):
                raise ValueError(
                    "format_dcomposer_targets: onset token exceeds configured range."
                )
            for j in range(n_events):
                note = int(notes_i[j].item())
                if note not in MIDI_TO_TOK:
                    raise ValueError(
                        f"format_dcomposer_targets: note {note} is not in MIDI token vocabulary."
                    )
                toks.append(int(MIDI_TO_TOK[note]))
                toks.append(int(ONSET_TOK_OFFSET + int(onset_toks[j].item())))
                toks.extend((codes_by_row[i][j] + ACOUSTIC_TOK_OFFSET).tolist())
        toks.append(EOS_TOK)
        seq = torch.tensor(toks, dtype=torch.long, device=output_device)
        seqs.append(seq)
        lengths.append(int(seq.numel()))

    max_len = max(lengths) if lengths else 1
    out = torch.full(
        (batch_size, max_len),
        PAD_TOK,
        dtype=torch.long,
        device=output_device,
    )
    for i, seq in enumerate(seqs):
        out[i, : seq.numel()] = seq

    return {
        "tokens": out,
        "lengths": torch.tensor(lengths, dtype=torch.long, device=output_device),
    }
