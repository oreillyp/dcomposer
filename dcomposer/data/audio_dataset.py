from pathlib import Path
from typing import Optional
from typing import Sequence
from typing import Union

import numpy as np
from audiotools.core.util import random_state
from torch.utils.data import Dataset

from ..util import collate
from ..util import load_excerpt
from ..util import normalize_source_weights
from .oneshot_dataset import _read_oneshot_manifest


class AudioDataset(Dataset):
    """
    Flat audio dataset backed by the same one-shot manifest format used by
    `OneShotDataset`, but sampling rows directly rather than one example per
    class label.
    """

    def __init__(
        self,
        sources: Union[str, Path, Sequence[Union[str, Path]]] = None,
        source_weights: Sequence[float] = None,
        sample_rate: int = None,
        duration: float = None,
        n_examples: int = 1000,
        num_channels: int = 1,
        relative_path: Optional[Union[str, Path]] = None,
        with_replacement: bool = True,
        shuffle_state: int = 0,
        from_start: bool = True,
        loudness_cutoff: Optional[float] = None,
        salience_num_tries: int = 4,
        path_col: str = "oneshot",
        label_col: str = "label",
        audio_backend: str = "file",
    ):
        super().__init__()

        self.sources = sources
        self.source_weights = source_weights
        self.sample_rate = int(sample_rate)
        self.duration = float(duration)
        self.num_channels = int(num_channels)
        self.length = int(n_examples)
        self.relative_path = None if relative_path in [None, ""] else str(relative_path)
        self.with_replacement = bool(with_replacement)
        self.shuffle_state = int(shuffle_state)
        self.from_start = bool(from_start)
        self.loudness_cutoff = loudness_cutoff
        self.salience_num_tries = int(salience_num_tries)
        self.path_col = str(path_col)
        self.label_col = str(label_col)
        self.audio_backend = str(audio_backend)
        if self.audio_backend not in {"file", "packed"}:
            raise ValueError(
                f"`audio_backend` must be 'file' or 'packed', got {audio_backend!r}."
            )

        source_list = [sources] if isinstance(sources, (str, Path)) else list(sources)
        per_source_rows = [
            _read_oneshot_manifest(
                sources=source,
                path_col=path_col,
                label_col=label_col,
                relative_path=relative_path,
            )
            for source in source_list
        ]
        kept_mask = [len(rows) > 0 for rows in per_source_rows]
        self.source_rows = [rows for rows in per_source_rows if len(rows) > 0]
        self.rows = [row for rows in self.source_rows for row in rows]
        if self.audio_backend == "packed":
            missing = [row["path"] for row in self.rows if not row["packed_audio_path"]]
            if missing:
                raise RuntimeError(
                    "AudioDataset: packed audio backend requested but manifest rows "
                    "are missing packed audio metadata."
                )
        if len(self.rows) == 0:
            raise RuntimeError("AudioDataset: no valid rows found in manifest(s).")
        self._weights = normalize_source_weights(
            source_weights=source_weights,
            n_sources=len(source_list),
            kept_mask=kept_mask,
            source_lengths=[len(rows) for rows in per_source_rows],
        )
        lengths = [len(rows) for rows in self.source_rows]
        self._source_offsets = np.cumsum([0] + lengths[:-1])

    def __len__(self):
        return self.length

    def _pick_row(self, idx: int, state):
        if self.with_replacement:
            source_idx = int(state.choice(len(self.source_rows), p=self._weights))
            item_idx = int(state.randint(len(self.source_rows[source_idx])))
            row_idx = int(self._source_offsets[source_idx] + item_idx)
            return row_idx, self.source_rows[source_idx][item_idx]
        row_idx = int(idx) % len(self.rows)
        return row_idx, self.rows[row_idx]

    def __getitem__(self, idx: int):
        state = random_state((self.shuffle_state + int(idx)) & 0x7FFFFFFF)
        row_idx, row = self._pick_row(idx, state)

        sig, offset = load_excerpt(
            row["path"],
            duration=self.duration,
            sample_rate=self.sample_rate,
            state=state,
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

        sig.metadata["path"] = row["path"]
        sig.metadata["offset"] = offset
        sig.metadata["source_row"] = row_idx
        sig.metadata["label"] = row["label"]
        for k, v in row["meta"].items():
            sig.metadata[k] = v

        return {
            "signal": sig,
            "label": row["label"],
            "path": row["path"],
            "idx": idx,
        }

    @staticmethod
    def collate(list_of_dicts, n_splits=None):
        return collate(list_of_dicts, n_splits=n_splits)
