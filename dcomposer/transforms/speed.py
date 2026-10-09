from typing import Optional

import numpy as np
import torch
from audiotools import AudioSignal
from audiotools.core.util import sample_from_dist

from ..dsp import resample
from .base import NormalizedBaseTransform
from .pitch import _cached_fast_ratios_and_semis

################################################################################
# Speed
################################################################################


def _to_list(v, B: int):
    if isinstance(v, torch.Tensor):
        v = v.detach().cpu().numpy()
    if isinstance(v, (list, tuple, np.ndarray)):
        v = list(map(float, v))
        if len(v) not in (1, B):
            raise ValueError(f"ratio length {len(v)} must be 1 or {B}")
        return v if len(v) == B else v * B
    return [float(v)] * B


class Speed(NormalizedBaseTransform):
    """
    Speed change via resampling.

    `ratio > 1` speeds audio up; `ratio < 1` slows it down.
    """

    def __init__(
        self,
        ratio: tuple = ("uniform", 0.9, 1.1),
        force_fast: bool = True,
        name: str = None,
        prob: float = 1.0,
        match_energy: bool = True,
        clamp_gain: Optional[float] = None,
        ensure_max_of_audio: bool = True,
    ):
        super().__init__(
            name=name,
            prob=prob,
            match_energy=match_energy,
            clamp_gain=clamp_gain,
            ensure_max_of_audio=ensure_max_of_audio,
        )
        self.ratio = ratio
        self.force_fast = bool(force_fast)

    def _instantiate(self, state, signal: Optional[AudioSignal] = None):
        return {"ratio": float(sample_from_dist(self.ratio, state))}

    @torch.no_grad()
    def _transform(self, signal: AudioSignal, ratio):
        x = signal.audio_data
        n_batch, _n_channels, n_samples = x.shape
        sample_rate = int(signal.sample_rate)
        device = x.device
        dtype = x.dtype

        ratios = _to_list(ratio, n_batch)
        if self.force_fast:
            fast_ratios, _fast_semis = _cached_fast_ratios_and_semis(sample_rate)
        else:
            fast_ratios = ()

        target_srs = []
        for ratio in ratios:
            if fast_ratios:
                ratio = float(
                    min(fast_ratios, key=lambda r: abs(float(r) - float(ratio)))
                )
            target_srs.append(max(1, int(round(sample_rate / float(ratio)))))

        groups = {}
        for i, target_sr in enumerate(target_srs):
            groups.setdefault(target_sr, []).append(i)

        y_out = torch.empty_like(x, device=device, dtype=dtype)
        tmp_sig = signal.clone()

        for target_sr, idxs in groups.items():
            idx = torch.tensor(idxs, device=device, dtype=torch.long)
            yg = x.index_select(0, idx)

            tmp_sig.audio_data = yg
            tmp_sig.sample_rate = sample_rate
            tmp_sig = resample(tmp_sig, target_sr, inplace=True)
            tmp_sig.sample_rate = sample_rate

            yg = tmp_sig.audio_data
            n_samples_out = yg.shape[-1]
            if n_samples_out > n_samples:
                yg = yg[..., :n_samples]
            elif n_samples_out < n_samples:
                yg = torch.nn.functional.pad(yg, (0, n_samples - n_samples_out))

            y_out.index_copy_(0, idx, yg)

        out = signal.clone()
        out.audio_data = y_out
        out.ensure_max_of_audio()
        return out
