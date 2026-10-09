import csv
from pathlib import Path
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
from torch.utils.data import Dataset

from ..util import collate
from ..util import load_excerpt

################################################################################
# Quick one-shot dataset for notebook experiments
################################################################################


def _read_oneshot_manifest(
    sources: Union[str, Path, Sequence[Union[str, Path]]],
    path_col: str = "oneshot",
    label_col: str = "label",
    relative_path: Optional[Union[str, Path]] = None,
) -> List[Dict]:
    rel = None if relative_path in [None, ""] else Path(relative_path).expanduser()
    csv_paths = [sources] if isinstance(sources, (str, Path)) else list(sources)

    rows = []
    for cpath in csv_paths:
        cpath = Path(cpath).expanduser()
        with open(cpath, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                raw = (row.get(path_col) or "").strip()
                label = (row.get(label_col) or "").strip()
                if not raw or not label:
                    continue

                meta = {k: v for k, v in row.items() if k not in [path_col, label_col]}
                packed_audio_path = (row.get("packed_audio_path") or "").strip()
                if packed_audio_path:
                    packed_path = Path(packed_audio_path).expanduser()
                    if not packed_path.is_absolute():
                        if rel is None:
                            raise ValueError(
                                "Manifest contains relative packed audio paths but `relative_path` was not provided."
                            )
                        packed_path = (rel / packed_path).expanduser()
                    packed_audio_path = str(packed_path)

                path = Path(raw).expanduser()
                if not path.is_absolute():
                    if rel is None:
                        raise ValueError(
                            "Manifest contains relative paths but `relative_path` was not provided."
                        )
                    path = (rel / path).expanduser()
                if not path.is_file() and not packed_audio_path:
                    continue
                rows.append(
                    {
                        "__manifest__": str(cpath),
                        "path": str(path),
                        "label": label,
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
                        "meta": meta,
                    }
                )

    return rows


class OneShotDataset(Dataset):
    """
    Load one drum one-shot per class label on every `__getitem__()` call.

    The label column is configurable via `label_col`, so the same manifest can
    expose multiple taxonomy levels such as `fine_label` and `coarse_label`.

    This yields a nested item of the form:
      {
        "oneshots": {
          "<label>": {
            "signal": AudioSignal,
            "label": str,
            "label_idx": LongTensor[()],
            "path": str,
          },
          ...
        },
        "idx": int,
      }
    """

    def __init__(
        self,
        sources: Union[str, Path, Sequence[Union[str, Path]]] = None,
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
        allowed_labels: Optional[Sequence[str]] = None,
        audio_backend: str = "file",
    ):
        super().__init__()

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
        self.audio_backend = str(audio_backend)
        if self.audio_backend not in {"file", "packed"}:
            raise ValueError(
                f"`audio_backend` must be 'file' or 'packed', got {audio_backend!r}."
            )

        rows = _read_oneshot_manifest(
            sources=sources,
            path_col=path_col,
            label_col=label_col,
            relative_path=relative_path,
        )
        if self.audio_backend == "packed":
            missing = [row["path"] for row in rows if not row["packed_audio_path"]]
            if missing:
                raise RuntimeError(
                    "OneShotDataset: packed audio backend requested but manifest rows "
                    "are missing packed audio metadata."
                )
        if allowed_labels is not None:
            allowed = set(allowed_labels)
            rows = [row for row in rows if row["label"] in allowed]

        if len(rows) == 0:
            raise RuntimeError("OneShotDataset: no valid rows found in manifest(s).")

        self.labels = sorted({row["label"] for row in rows})
        self.label_to_idx = {label: i for i, label in enumerate(self.labels)}
        self.rows_by_label = {
            label: [row for row in rows if row["label"] == label]
            for label in self.labels
        }
        self.label_counts = {
            label: len(self.rows_by_label[label]) for label in self.labels
        }

        empty = [
            label for label, items in self.rows_by_label.items() if len(items) == 0
        ]
        if empty:
            raise RuntimeError(f"OneShotDataset: labels with no rows: {empty}")

    def __len__(self):
        return self.length

    def _pick_row(self, idx: int, label: str):
        rows = self.rows_by_label[label]
        if self.with_replacement:
            state = random_state(
                (self.shuffle_state + int(idx) + 9973 * self.label_to_idx[label])
                & 0x7FFFFFFF
            )
            row_idx = int(state.randint(len(rows)))
        else:
            row_idx = int(idx) % len(rows)
        return row_idx, rows[row_idx]

    def __getitem__(self, idx: int):
        item = {"oneshots": {}, "idx": idx}

        for label in self.labels:
            row_idx, row = self._pick_row(idx, label=label)
            state = random_state(
                (self.shuffle_state + int(idx) + 9973 * self.label_to_idx[label])
                & 0x7FFFFFFF
            )

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

            item["oneshots"][label] = {
                "signal": sig,
                "label": row["label"],
                "label_idx": torch.tensor(self.label_to_idx[label], dtype=torch.long),
                "path": row["path"],
            }

        return item

    @staticmethod
    def collate(list_of_dicts: Union[list, dict], n_splits: int = None):
        return collate(list_of_dicts, n_splits=n_splits)


def apply_oneshot_mixup(
    *args,
    **kwargs,
):
    raise NotImplementedError(
        "`apply_oneshot_mixup()` has been superseded by "
        "`build_oneshot_training_batch()`, which constructs a fixed-size flat "
        "batch from the class-structured candidate pool."
    )


def _normalize_class_weights(
    labels: Sequence[str],
    class_weights: Optional[Dict[str, float]] = None,
) -> Dict[str, float]:
    if len(labels) == 0:
        raise ValueError("No labels available for batch construction.")

    if class_weights is None:
        weights = {label: 1.0 for label in labels}
    else:
        weights = {label: float(class_weights.get(label, 0.0)) for label in labels}

    total = sum(max(0.0, w) for w in weights.values())
    if total <= 0.0:
        raise ValueError(
            "Class weights must assign positive mass to at least one available label."
        )

    return {label: max(0.0, weights[label]) / total for label in labels}


def _extract_candidate_pool(batch: Dict) -> Dict[str, List[Dict]]:
    pool: Dict[str, List[Dict]] = {}
    batch_idx = batch.get("idx", None)

    for label, item in batch["oneshots"].items():
        signal = item["signal"]
        lengths = item.get("signal_lengths", None)
        label_idx = item["label_idx"]
        paths = item.get("path", None)
        n_batch = signal.audio_data.shape[0]

        if lengths is None:
            lengths = torch.full(
                (n_batch,),
                signal.audio_data.shape[-1],
                dtype=torch.long,
                device=signal.audio_data.device,
            )

        pool[label] = []
        for i in range(n_batch):
            length = int(lengths[i].item())
            audio = signal.audio_data[i : i + 1, :, :length].clone()
            sig_i = AudioSignal(audio, sample_rate=signal.sample_rate)
            sig_i.metadata = dict(getattr(signal, "metadata", {}) or {})
            sig_i.metadata["label"] = label
            if paths is not None:
                sig_i.metadata["path"] = paths[i]

            pool[label].append(
                {
                    "signal": sig_i,
                    "length": length,
                    "label": label,
                    "label_idx": int(label_idx[i].item()),
                    "path": None if paths is None else paths[i],
                    "idx": int(i if batch_idx is None else batch_idx[i]),
                }
            )

    return pool


def _mix_pair(
    a: Dict,
    b: Dict,
    lam: float,
) -> Dict:
    target_len = max(int(a["length"]), int(b["length"]))
    xa = a["signal"].audio_data
    xb = b["signal"].audio_data

    if xa.shape[-1] < target_len:
        xa = F.pad(xa, (0, target_len - xa.shape[-1]))
    if xb.shape[-1] < target_len:
        xb = F.pad(xb, (0, target_len - xb.shape[-1]))

    x = lam * xa + (1.0 - lam) * xb
    m_src = (xa[..., : int(a["length"])] ** 2).mean().clamp_min(1e-12)
    m_mix = (x[..., :target_len] ** 2).mean().clamp_min(1e-12)
    x = x * torch.sqrt(m_src / m_mix)
    sig = AudioSignal(x, sample_rate=a["signal"].sample_rate)
    sig.metadata = dict(getattr(a["signal"], "metadata", {}) or {})
    sig.ensure_max_of_audio()

    return {
        "signal": sig,
        "length": target_len,
        "label": a["label"],
        "label_idx": a["label_idx"],
        "mixed": True,
        "mix_weight": float(lam),
        "mix_src_label_idx": a["label_idx"],
        "mix_dst_label_idx": b["label_idx"],
        "mix_src_label": a["label"],
        "mix_dst_label": b["label"],
        "path": a["path"],
        "mix_path": b["path"],
        "idx": a["idx"],
    }


def build_oneshot_batch(
    batch: Dict,
    out_batch_size: Optional[int] = None,
    class_weights: Optional[Dict[str, float]] = None,
    p_mixup_intra_class: float = 0.0,
    p_mixup_inter_class: float = 0.0,
    alpha: float = 1.0,
    state: Optional[np.random.RandomState] = None,
):
    """
    Construct a fixed-size flat training batch from a class-structured batch.

    Parameters
    ----------
    batch : dict
        Output of `OneShotDataset.collate(...)`.
    out_batch_size : Optional[int]
        Number of final training examples to emit. Defaults to the per-class
        batch size of the input candidate pool.
    class_weights : Optional[Dict[str, float]]
        Relative probability of drawing each class as the base class for an
        output example. Labels missing from the dict receive weight 0.0.
    p_mixup_intra_class : float
        Probability of mixing a sample with another sample from the same class.
    p_mixup_inter_class : float
        Probability of mixing a sample with a sample from a different class.
    alpha : float
        Beta-distribution parameter for mixup coefficients. We sample
        `lam ~ Beta(alpha, alpha)` and mix as `lam * a + (1 - lam) * b`.
        Larger values concentrate `lam` near 0.5; smaller positive values push
        `lam` toward 0 or 1. If `alpha <= 0`, we fall back to `lam = 0.5`.
    state : Optional[np.random.RandomState]
        Random state used for deterministic construction.

    Returns
    -------
    dict
        Flat batch with keys:
          * `signal`: batched AudioSignal
          * `signal_lengths`: LongTensor
          * `label_idx`: LongTensor
          * `label`: list[str]
          * `mixed`: BoolTensor
          * `mix_weight`: Tensor
          * `mix_src_label_idx`: LongTensor
          * `mix_dst_label_idx`: LongTensor
          * `mix_src_label`: list[str]
          * `mix_dst_label`: list[str]
          * `path`: list[str]
          * `mix_path`: list[str]
          * `idx`: LongTensor
    """
    if p_mixup_intra_class < 0.0 or p_mixup_inter_class < 0.0:
        raise ValueError("Mixup probabilities must be >= 0.")
    if p_mixup_intra_class + p_mixup_inter_class > 1.0:
        raise ValueError("`p_mixup_intra_class + p_mixup_inter_class` must be <= 1.0.")

    state = random_state(state)
    pool = _extract_candidate_pool(batch)
    available_labels = [label for label, items in pool.items() if len(items) > 0]
    weights = _normalize_class_weights(available_labels, class_weights=class_weights)

    if out_batch_size is None:
        out_batch_size = max(len(pool[label]) for label in available_labels)
    out_batch_size = int(out_batch_size)
    if out_batch_size <= 0:
        raise ValueError("`out_batch_size` must be positive.")

    label_probs = np.asarray(
        [weights[label] for label in available_labels], dtype=float
    )
    label_probs = label_probs / label_probs.sum()

    examples = []
    for _ in range(out_batch_size):
        base_label = str(state.choice(available_labels, p=label_probs))
        base_pool = pool[base_label]
        base_idx = int(state.randint(len(base_pool)))
        base = base_pool[base_idx]

        p = float(state.rand())
        if p < p_mixup_intra_class and len(base_pool) > 1:
            partner_idx = int(state.randint(len(base_pool) - 1))
            if partner_idx >= base_idx:
                partner_idx += 1
            partner = base_pool[partner_idx]
            lam = float(state.beta(alpha, alpha)) if alpha > 0 else 0.5
            examples.append(_mix_pair(base, partner, lam=lam))
            continue

        if p < p_mixup_intra_class + p_mixup_inter_class and len(available_labels) > 1:
            other_labels = [label for label in available_labels if label != base_label]
            other_weights = np.asarray(
                [weights[label] for label in other_labels], dtype=float
            )
            if other_weights.sum() > 0:
                other_probs = other_weights / other_weights.sum()
                other_label = str(state.choice(other_labels, p=other_probs))
                other_pool = pool[other_label]
                partner = other_pool[int(state.randint(len(other_pool)))]
                lam = float(state.beta(alpha, alpha)) if alpha > 0 else 0.5
                lam = max(lam, 1.0 - lam)
                examples.append(_mix_pair(base, partner, lam=lam))
                continue

        examples.append(
            {
                "signal": base["signal"].clone().ensure_max_of_audio(),
                "length": int(base["length"]),
                "label": base["label"],
                "label_idx": int(base["label_idx"]),
                "mixed": False,
                "mix_weight": 1.0,
                "mix_src_label_idx": int(base["label_idx"]),
                "mix_dst_label_idx": int(base["label_idx"]),
                "mix_src_label": base["label"],
                "mix_dst_label": base["label"],
                "path": base["path"],
                "mix_path": base["path"],
                "idx": base["idx"],
            }
        )

    flat = []
    for ex in examples:
        flat.append(
            {
                "signal": ex["signal"],
                "label_idx": torch.tensor(ex["label_idx"], dtype=torch.long),
                "label": ex["label"],
                "mixed": torch.tensor(ex["mixed"], dtype=torch.bool),
                "mix_weight": torch.tensor(ex["mix_weight"], dtype=torch.float32),
                "mix_src_label_idx": torch.tensor(
                    ex["mix_src_label_idx"], dtype=torch.long
                ),
                "mix_dst_label_idx": torch.tensor(
                    ex["mix_dst_label_idx"], dtype=torch.long
                ),
                "mix_src_label": ex["mix_src_label"],
                "mix_dst_label": ex["mix_dst_label"],
                "path": ex["path"],
                "mix_path": ex["mix_path"],
                "idx": torch.tensor(ex["idx"], dtype=torch.long),
            }
        )

    return collate(flat)
