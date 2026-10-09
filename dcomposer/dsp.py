import math
from collections import OrderedDict
from dataclasses import dataclass
from typing import Dict
from typing import Generic
from typing import Optional
from typing import TypeVar
from typing import Union

import julius
import numpy as np
import torch
from audiotools import AudioSignal
from audiotools.core.util import ensure_tensor
from audiotools.core.util import sample_from_dist

################################################################################
# Signal processing utilities for filters, etc.
################################################################################

K = TypeVar("K")  # Hashable key type for caching


@dataclass(frozen=True)
class FilterKey:
    """
    Cache low/high-pass kernels to avoid re-initialization and device movement
    """

    kind: str  # "lp" or "hp"
    cutoff_norm_q: float  # Normalized cutoff (Hz / sr), quantized
    zeros: int
    device: str
    dtype: torch.dtype


@dataclass(frozen=True)
class ResampleKey:
    """
    Cache resampling kernels to avoid re-initialization and device movement
    """

    old_sr: int
    new_sr: int
    zeros: int
    rolloff: float
    device: str
    dtype: torch.dtype


class _PerDeviceModuleCache(Generic[K]):
    """
    Base class for caching kernels keyed by K, with per-device LRU and automatic
    eviction when memory budget is exceeded.

    Subclasses implement:
      - _make_module(key, *, device, dtype) -> torch.nn.Module
    """

    def __init__(
        self,
        max_bytes_per_device: Optional[Dict[str, int]] = None,
        default_max_bytes: int = 256 * 1024 * 1024,
    ):
        self.default_max_bytes = int(default_max_bytes)
        self.max_bytes_per_device = dict(max_bytes_per_device or {})

        self._cache: Dict[str, OrderedDict[K, torch.nn.Module]] = {}
        self._bytes: Dict[str, int] = {}

    def set_max_bytes(self, device: str, max_bytes: int):
        self.max_bytes_per_device[str(device)] = int(max_bytes)

    def get_max_bytes(self, device: str) -> int:
        return int(self.max_bytes_per_device.get(str(device), self.default_max_bytes))

    def clear(self, device: Optional[str] = None):
        if device is None:
            self._cache.clear()
            self._bytes.clear()
            return
        d = str(device)
        self._cache.pop(d, None)
        self._bytes.pop(d, None)

    @staticmethod
    def _module_bytes(m: torch.nn.Module) -> int:
        n = 0
        for _, buf in m.named_buffers(recurse=True):
            n += buf.numel() * buf.element_size()
        for _, p in m.named_parameters(recurse=True):
            n += p.numel() * p.element_size()
        return int(n)

    def _evict_if_needed(self, d: str):
        budget = self.get_max_bytes(d)
        if budget <= 0:
            self.clear(d)
            return

        cache_d = self._cache.get(d)
        if not cache_d:
            return

        cur = int(self._bytes.get(d, 0))
        while cache_d and cur > budget:
            _, m = cache_d.popitem(last=False)  # LRU eviction
            cur -= self._module_bytes(m)
        self._bytes[d] = max(0, cur)

    def get(
        self, key: K, *, device: torch.device, dtype: torch.dtype
    ) -> torch.nn.Module:
        d = str(device)

        # Caching disabled for this device: construct + return without tracking
        if self.get_max_bytes(d) <= 0:
            return self._make_module(key, device=device, dtype=dtype)

        cache_d = self._cache.setdefault(d, OrderedDict())
        m = cache_d.get(key)
        if m is not None:
            cache_d.move_to_end(key)
            return m

        m = self._make_module(key, device=device, dtype=dtype)

        cache_d[key] = m
        cache_d.move_to_end(key)

        self._bytes[d] = int(self._bytes.get(d, 0)) + self._module_bytes(m)
        self._evict_if_needed(d)
        return m

    def _make_module(
        self, key: K, *, device: torch.device, dtype: torch.dtype
    ) -> torch.nn.Module:
        raise NotImplementedError


class FilterCache(_PerDeviceModuleCache[FilterKey]):
    def __init__(
        self,
        max_bytes_per_device: Optional[Dict[str, int]] = None,
        default_max_bytes: int = 256 * 1024 * 1024,
    ):
        super().__init__(
            max_bytes_per_device=max_bytes_per_device,
            default_max_bytes=default_max_bytes,
        )

    def get(
        self,
        *,
        kind: str,
        cutoff_norm_q: float,
        zeros: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.nn.Module:
        kind = str(kind)
        if kind not in ("lp", "hp"):
            raise ValueError(f"kind must be 'lp' or 'hp', got {kind}")

        d = str(device)
        key = FilterKey(
            kind=kind,
            cutoff_norm_q=float(cutoff_norm_q),
            zeros=int(zeros),
            device=d,
            dtype=dtype,
        )
        return super().get(key, device=device, dtype=dtype)

    def _make_module(
        self, key: FilterKey, *, device: torch.device, dtype: torch.dtype
    ) -> torch.nn.Module:
        if key.kind == "lp":
            m = julius.LowPassFilter(float(key.cutoff_norm_q), zeros=int(key.zeros))
        else:
            m = julius.HighPassFilter(float(key.cutoff_norm_q), zeros=int(key.zeros))
        return m.to(device=device, dtype=dtype)


class ResampleCache(_PerDeviceModuleCache[ResampleKey]):
    def __init__(
        self,
        max_bytes_per_device: Optional[Dict[str, int]] = None,
        default_max_bytes: int = 512 * 1024 * 1024,
    ):
        super().__init__(
            max_bytes_per_device=max_bytes_per_device,
            default_max_bytes=default_max_bytes,
        )

    def get(
        self,
        old_sr: int,
        new_sr: int,
        *,
        zeros: int,
        rolloff: float,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.nn.Module:
        old_sr = int(old_sr)
        new_sr = int(new_sr)
        d = str(device)
        key = ResampleKey(
            old_sr=old_sr,
            new_sr=new_sr,
            zeros=int(zeros),
            rolloff=float(rolloff),
            device=d,
            dtype=dtype,
        )
        return super().get(key, device=device, dtype=dtype)

    def _make_module(
        self, key: ResampleKey, *, device: torch.device, dtype: torch.dtype
    ) -> torch.nn.Module:
        return julius.ResampleFrac(
            int(key.old_sr), int(key.new_sr), int(key.zeros), float(key.rolloff)
        ).to(device=device, dtype=dtype)


def set_filter_cache_max_bytes(device: str, max_bytes: int):
    _FILTER_CACHE.set_max_bytes(device, max_bytes)


def clear_filter_cache(device: Optional[str] = None):
    _FILTER_CACHE.clear(device=device)


def set_resample_cache_max_bytes(device: str, max_bytes: int):
    _RESAMPLE_CACHE.set_max_bytes(device, max_bytes)


def clear_resample_cache(device: Optional[str] = None):
    _RESAMPLE_CACHE.clear(device=device)


# Initialize caches per process
_FILTER_CACHE = FilterCache()
_RESAMPLE_CACHE = ResampleCache()


def _quantize_hz(x_hz: torch.Tensor, q_hz: float) -> torch.Tensor:
    q = float(q_hz)
    if q <= 0:
        return x_hz  # If q_hz <= 0, do not quantize (may lead to large cache size)
    return torch.round(x_hz / q) * q


def _get_window(name: str, length: int, *, device: torch.device) -> torch.Tensor:
    """
    Build a 1D window on-device.
    """
    length = int(length)
    if length <= 0:
        return torch.zeros(0, device=device)
    if name == "hann":
        if length == 1:
            return torch.ones(1, device=device)
        n = torch.arange(length, device=device, dtype=torch.float32)
        return 0.5 - 0.5 * torch.cos((2.0 * math.pi * n) / float(length - 1))
    raise ValueError(f"Unsupported window type: {name}")


def _decompose_ir(x: torch.Tensor, sample_rate: int):
    """
    Decompose impulse response into early and late fields using a channel-wise
    peak-centered window.

    Parameters
    ----------
    x : torch.Tensor
        Impulse response, shape (n_batch, n_channels, n_samples)
    """
    n_batch, n_channels, n_samples = x.shape
    device = x.device

    td = x.abs().argmax(dim=-1, keepdim=True)  # (n_batch, n_channels, 1)
    t0 = int(sample_rate * 0.0025)

    t = torch.arange(n_samples, device=device)[None, None, :]  # (1, 1, n_samples)
    early_idx = (t >= (td - t0)) & (t <= (td + t0))  # (n_batch, n_channels, n_samples)

    early_response = torch.zeros_like(x)
    early_response[early_idx] = x[early_idx]

    late_field = torch.zeros_like(x)
    late_field[~early_idx] = x[~early_idx]

    window = torch.zeros_like(x)
    for b in range(n_batch):
        for c in range(n_channels):
            idx = early_idx[b, c].nonzero(as_tuple=False).view(-1)
            if idx.numel() == 0:
                continue
            window[b, c, idx] = _get_window("hann", idx.numel(), device=device).to(
                x.dtype
            )

    return early_response, late_field, window


def _solve_alpha(
    early_response: torch.Tensor,
    late_field: torch.Tensor,
    window: torch.Tensor,
    target_drr,
) -> torch.Tensor:
    """
    Solve for alpha in the DRR-alteration quadratic with basic numerical
    safeguards.
    """
    target_drr = ensure_tensor(target_drr, 2, early_response.shape[0]).to(
        early_response.device
    )
    target_drr = target_drr.to(dtype=early_response.dtype)

    wd_sq = window.square()
    wd_sq_1 = (1.0 - window).square()
    e_sq = early_response.square()
    l_sq = late_field.square()

    a = (wd_sq * e_sq).sum(dim=-1)  # (n_batch, n_channels)
    b = (2.0 * (1.0 - window) * window * e_sq).sum(dim=-1)
    c = (wd_sq_1 * e_sq).sum(dim=-1) - torch.pow(10.0, target_drr / 10.0) * l_sq.sum(
        dim=-1
    )

    disc = (b.square() - 4.0 * a * c).clamp_min(0.0)
    expr = disc.sqrt()

    denom = (2.0 * a).clamp_min(1e-8)
    alpha0 = (-b - expr) / denom
    alpha1 = (-b + expr) / denom
    alpha = torch.maximum(alpha0, alpha1)

    # When the windowed early response is degenerate, fall back to no change.
    alpha = torch.where(a > 1e-8, alpha, torch.ones_like(alpha))
    return alpha


@torch.no_grad()
def alter_drr(
    ir: AudioSignal,
    drr,
    *,
    inplace: bool = True,
) -> AudioSignal:
    """
    Safely alter the direct-to-reverberant ratio of an impulse response.
    """
    out = ir if inplace else ir.clone()

    early_response, late_field, window = _decompose_ir(
        out.audio_data, int(out.sample_rate)
    )
    alpha = _solve_alpha(early_response, late_field, window, drr)

    early_peak = early_response.abs().amax(dim=-1).clamp_min(1e-8)
    late_peak = late_field.abs().amax(dim=-1)
    min_alpha = late_peak / early_peak
    alpha = torch.maximum(alpha, min_alpha)[..., None]

    out.audio_data = (
        alpha * window * early_response + (1.0 - window) * early_response + late_field
    )
    out.ensure_max_of_audio()
    return out


@torch.no_grad()
def low_pass(
    sig: AudioSignal,
    cutoffs_hz: Union[torch.Tensor, np.ndarray, float],
    *,
    zeros: int = 51,
    quantize_hz: float = 1.0,
    inplace: bool = True,
) -> AudioSignal:
    """
    Apply low-pass filtering with cached Julius kernel.

    Parameters
    ----------
    cutoffs_hz : torch.Tensor
        Shape (n_batch,)
    """
    out = sig if inplace else sig.clone()
    n_batch = int(out.batch_size)
    sr = float(out.sample_rate)

    cut = ensure_tensor(cutoffs_hz, 2, n_batch).to(out.device).view(-1).float()
    cut = cut.clamp(min=0.0, max=0.5 * sr)

    # Quantize in Hz to keep cache small
    cut_q = _quantize_hz(cut, quantize_hz)

    # Normalize (0..0.5)
    cut_norm_q = (cut_q / sr).clamp(min=0.0, max=0.5)

    x = out.audio_data  # (n_batch, n_channels, n_samples)
    y = torch.empty_like(x)

    # Group identical quantized cutoffs to reduce module lookups/calls
    uniq = torch.unique(cut_norm_q.detach().cpu())
    for v in uniq.tolist():
        m = _FILTER_CACHE.get(
            kind="lp",
            cutoff_norm_q=float(v),
            zeros=int(zeros),
            device=x.device,
            dtype=x.dtype,
        )
        mask = cut_norm_q == float(v)
        idx = mask.nonzero(as_tuple=False).view(-1)
        if idx.numel() == 0:
            continue
        y[idx] = m(x[idx].contiguous())

    out.audio_data = y
    out.stft_data = None
    return out


@torch.no_grad()
def high_pass(
    sig: AudioSignal,
    cutoffs_hz: Union[torch.Tensor, np.ndarray, float],
    *,
    zeros: int = 51,
    quantize_hz: float = 1.0,
    inplace: bool = True,
) -> AudioSignal:
    """
    Apply high-pass filtering with cached Julius kernel.

    Parameters
    ----------
    cutoffs_hz : torch.Tensor
        Shape (n_batch,)
    """
    out = sig if inplace else sig.clone()
    n_batch = int(out.batch_size)
    sr = float(out.sample_rate)

    cut = ensure_tensor(cutoffs_hz, 2, n_batch).to(out.device).view(-1).float()
    cut = cut.clamp(min=0.0, max=0.5 * sr)

    cut_q = _quantize_hz(cut, quantize_hz)
    cut_norm_q = (cut_q / sr).clamp(min=0.0, max=0.5)

    x = out.audio_data
    y = torch.empty_like(x)

    uniq = torch.unique(cut_norm_q.detach().cpu())
    for v in uniq.tolist():
        m = _FILTER_CACHE.get(
            kind="hp",
            cutoff_norm_q=float(v),
            zeros=int(zeros),
            device=x.device,
            dtype=x.dtype,
        )
        mask = cut_norm_q == float(v)
        idx = mask.nonzero(as_tuple=False).view(-1)
        if idx.numel() == 0:
            continue
        y[idx] = m(x[idx].contiguous())

    out.audio_data = y
    out.stft_data = None
    return out


@torch.no_grad()
def resample_tensor(
    x: torch.Tensor,
    old_sr: int,
    new_sr: int,
    *,
    zeros: int = 24,
    rolloff: float = 0.945,
    output_length: Optional[int] = None,
    full: bool = False,
    kernel_dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    """
    Cached resampling with cached Julius kernel.
    """
    old_sr = int(old_sr)
    new_sr = int(new_sr)
    if old_sr == new_sr:
        return x

    kd = kernel_dtype if kernel_dtype is not None else x.dtype
    m = _RESAMPLE_CACHE.get(
        old_sr,
        new_sr,
        zeros=zeros,
        rolloff=rolloff,
        device=x.device,
        dtype=kd,
    )
    y = m(x, output_length=output_length, full=full)

    if y.dtype != x.dtype:
        y = y.to(dtype=x.dtype)
    return y


@torch.no_grad()
def resample(
    sig: AudioSignal,
    new_sr: int,
    *,
    zeros: int = 24,
    rolloff: float = 0.945,
    output_length: Optional[int] = None,
    full: bool = False,
    inplace: bool = True,
    kernel_dtype: Optional[torch.dtype] = None,
) -> AudioSignal:
    """
    Resample signal to target rate.
    """
    new_sr = int(new_sr)
    if int(sig.sample_rate) == new_sr:
        return sig

    out = sig if inplace else sig.clone()
    out.audio_data = resample_tensor(
        out.audio_data,
        int(out.sample_rate),
        new_sr,
        zeros=zeros,
        rolloff=rolloff,
        output_length=output_length,
        full=full,
        kernel_dtype=kernel_dtype,
    )
    out.sample_rate = new_sr
    return out


def _roll_to_peak(x: torch.Tensor) -> torch.Tensor:
    """
    Roll each IR so the maximum magnitude lands at t=0, avoiding delay.
    """
    n_batch, n_channels, n_samples = x.shape

    flat = x.abs().reshape(n_batch, n_channels * n_samples)
    idx_lin = flat.argmax(dim=-1)  # (n_batch,)
    idx_t = (idx_lin % n_samples).to(torch.long)  # (n_batch,)

    t = torch.arange(n_samples, device=x.device)  # (n_samples,)
    gather_idx = (
        t[None, None, :] + idx_t[:, None, None]
    ) % n_samples  # (n_batch, 1, n_samples)
    gather_idx = gather_idx.expand(n_batch, n_channels, n_samples)
    return x.gather(dim=-1, index=gather_idx)


def _convolve(
    signal: AudioSignal,
    ir: AudioSignal,
    *,
    start_at_peak: bool = True,
) -> AudioSignal:
    """
    Perform convolution via FFT multiplication, padding/truncating IR to signal
    length and optionally rolling IR to peak to avoid delay,
    """
    ir = ir.clone()

    pad_len = signal.signal_length - ir.signal_length
    if pad_len > 0:
        ir.zero_pad(0, pad_len)
    else:
        ir.truncate_samples(signal.signal_length)

    if ir.num_channels not in (1, signal.num_channels):
        raise ValueError(
            f"IR channels ({ir.num_channels}) must be 1 or match signal channels ({signal.num_channels})."
        )
    if ir.num_channels == 1 and signal.num_channels > 1:
        ir.audio_data = ir.audio_data.repeat(1, signal.num_channels, 1)

    if start_at_peak:
        ir = AudioSignal(_roll_to_peak(ir.audio_data), ir.sample_rate)

    length = signal.signal_length
    delta = torch.zeros_like(ir.audio_data)
    delta[..., 0] = 1

    delta_fft = torch.fft.rfft(delta, length)
    ir_fft = torch.fft.rfft(ir.audio_data, length)
    sig_fft = torch.fft.rfft(signal.audio_data, length)

    y = torch.fft.irfft(ir_fft * sig_fft, length)
    delta_y = torch.fft.irfft(ir_fft * delta_fft, length)

    delta_max = delta_y.abs().max(dim=-1, keepdim=True).values  # (B,C,1)
    y = y * (1.0 / delta_max.clamp(1e-5))

    signal.audio_data = y
    return signal


def apply_ir(
    signal: AudioSignal,
    ir: AudioSignal,
    drr=None,
    ir_eq=None,
    use_original_phase: bool = False,
    start_at_peak: bool = True,
) -> AudioSignal:
    ir = ir.clone()

    if ir_eq is not None:
        ir = ir.equalizer(ir_eq)
    if drr is not None:
        ir = alter_drr(ir, drr, inplace=True)

    max_spk = signal.audio_data.abs().max(dim=-1, keepdim=True).values

    phase = signal.phase if use_original_phase else None
    _convolve(signal, ir, start_at_peak=start_at_peak)

    if use_original_phase:
        signal.stft()
        signal.stft_data = signal.magnitude * torch.exp(1j * phase)
        signal.istft()

    max_transformed = signal.audio_data.abs().max(dim=-1, keepdim=True).values
    scale_factor = max_spk.clamp(1e-8) / max_transformed.clamp(1e-8)
    signal.audio_data = signal.audio_data * scale_factor

    return signal
