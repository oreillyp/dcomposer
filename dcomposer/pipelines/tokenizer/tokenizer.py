from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Dict
from typing import Optional
from typing import Union

import torch
from audiotools import AudioSignal

from ...constants import PRETRAINED_DIR
from ...dsp import resample
from .codicodec import CodiCodecVAE
from .melodyflow import MelodyFlowVAE
from .stable_audio import StableAudioVAE

################################################################################
# Pipeline for continuous audio tokenizers
################################################################################


@dataclass
class TokenSequence:
    extras: Dict
    tokens: torch.Tensor
    scaled: bool = False
    scale_mean: Optional[torch.Tensor] = None
    scale_std: Optional[torch.Tensor] = None

    def clone(self) -> "TokenSequence":
        extras = {}
        for k, v in self.extras.items():
            extras[k] = v.clone() if isinstance(v, torch.Tensor) else v
        return TokenSequence(
            extras=extras,
            tokens=self.tokens.clone(),
            scaled=self.scaled,
            scale_mean=None if self.scale_mean is None else self.scale_mean.clone(),
            scale_std=None if self.scale_std is None else self.scale_std.clone(),
        )


def _format_stats(
    value: Optional[Union[float, torch.Tensor]],
    n_channels: int,
    name: str,
) -> Optional[torch.Tensor]:
    if value is None:
        return None

    if isinstance(value, torch.Tensor):
        stats = value.detach().to(dtype=torch.float32, device="cpu")
    else:
        stats = torch.tensor(value, dtype=torch.float32)

    if stats.ndim == 0:
        stats = stats.reshape(1, 1)
    elif stats.ndim == 1:
        stats = stats.reshape(-1, 1)
    elif stats.ndim == 2 and stats.shape[-1] == 1:
        pass
    else:
        raise ValueError(
            f"`{name}` must be a scalar, [n_channels], or [n_channels, 1]; "
            f"got shape {tuple(stats.shape)}"
        )

    if stats.shape[0] not in [1, n_channels]:
        raise ValueError(
            f"`{name}` channel dim must be 1 or {n_channels}; got {stats.shape[0]}"
        )

    if stats.shape[0] == 1 and n_channels > 1:
        stats = stats.expand(n_channels, 1).contiguous()

    return stats


class Tokenizer(torch.nn.Module):
    valid = ["codicodec", "stable_audio", "melodyflow"]
    ckpt_dir = PRETRAINED_DIR / "tokenizer"

    def __init__(
        self,
        name: str = "codicodec",
        sample_rate: int = None,
        load_saved_scale: bool = True,
        scale_mean: bool = True,
        scale_std: bool = True,
        scale_per_channel: bool = False,
        saved_scale_name: str = "scale_stats.pt",
        **kwargs,
    ):
        super().__init__()

        assert (
            name in self.valid
        ), f"Invalid tokenizer name {name}; must be one of {self.valid}"

        self.name = None
        self.model = None

        self.sample_rate = None
        self.frame_rate = None
        self.n_channels = None
        self.latent_dim = None
        self.n_latent_channels = None
        self.hop_length = None
        self.normalize_db = kwargs.get("normalize_db", None)
        self.loudness_exclude_silence = kwargs.get("loudness_exclude_silence", True)
        self.load_saved_scale_enabled = bool(load_saved_scale)
        self.scale_mean_enabled = bool(scale_mean)
        self.scale_std_enabled = bool(scale_std)
        self.scale_per_channel = bool(scale_per_channel)
        self.saved_scale_name = saved_scale_name

        # Token scaling state. Defaults to no-op (mean=0, std=1).
        self._scale_mean = None
        self._scale_std = None

        if name == "codicodec":
            expected_sample_rate = 48_000
            assert (sample_rate or expected_sample_rate) == expected_sample_rate

            self.name = "codicodec"
            self.sample_rate = expected_sample_rate
            self.frame_rate = 11.0  # (Asymptotically; summary latent tokenizer is not frame-rate based)
            self.n_channels = 2
            self.latent_dim = kwargs.get("desired_channels", 64)
            self.hop_length = 256

            self.model = CodiCodecVAE(
                checkpoint_path=kwargs.get(
                    "ckpt_pth",
                    self.ckpt_dir / "codicodec" / "codicodec_continuous.pt",
                ),
                desired_channels=self.latent_dim,
                decode_mode=kwargs.get("decode_mode", "parallel"),
                denoising_steps=kwargs.get("denoising_steps", 5),
                max_batch_size_encode=kwargs.get("max_batch_size_encode", 64),
                max_batch_size_decode=kwargs.get("max_batch_size_decode", 32),
            )
        elif name == "stable_audio":
            self.name = "stable_audio"
            self.model = StableAudioVAE(
                checkpoint_path=kwargs.get(
                    "ckpt_pth",
                    self.ckpt_dir / "stable_audio" / "stable_audio_vae.pt",
                ),
            )

            self.sample_rate = self.model.sample_rate
            self.frame_rate = 21.533203125  # 44.1k / 2048
            self.n_channels = self.model.n_channels
            self.latent_dim = self.model.latent_dim
            self.hop_length = self.model.hop_length

            if sample_rate is not None:
                assert sample_rate == self.sample_rate

        elif name == "melodyflow":
            self.name = "melodyflow"
            self.model = MelodyFlowVAE(
                checkpoint_path=kwargs.get(
                    "ckpt_pth",
                    self.ckpt_dir / "melodyflow" / "melodyflow_encodec_vae.pt",
                ),
            )

            self.sample_rate = sample_rate or self.model.sample_rate
            self.frame_rate = 25.0  # 48k / 1920
            self.n_channels = self.model.n_channels
            self.latent_dim = self.model.latent_dim
            self.hop_length = self.model.hop_length
        else:
            raise NotImplementedError(f"Tokenizer {name} not yet implemented")

        self.n_latent_channels = self.latent_dim

        # Put tokenizer in eval mode and disable gradients by default.
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

        # Initialize to no-op scaling.
        self.set_scale(mean=0.0, std=1.0)
        if load_saved_scale:
            self.load_saved_scale(
                scale_mean=self.scale_mean_enabled,
                scale_std=self.scale_std_enabled,
                per_channel=self.scale_per_channel,
            )

        for a in [
            self.name,
            self.model,
            self.sample_rate,
            self.frame_rate,
            self.n_channels,
            self.latent_dim,
            self.n_latent_channels,
            self.hop_length,
        ]:
            assert a is not None

    @property
    def scale_mean(self) -> torch.Tensor:
        return self._scale_mean

    @property
    def scale_std(self) -> torch.Tensor:
        return self._scale_std

    def __repr__(self):
        mean = self._scale_mean
        std = self._scale_std
        mean_desc = "None" if mean is None else str(tuple(mean.shape))
        std_desc = "None" if std is None else str(tuple(std.shape))
        return (
            f"Tokenizer(name='{self.name}', sample_rate={self.sample_rate}, "
            f"frame_rate={self.frame_rate}, n_channels={self.n_channels}, "
            f"n_latent_channels={self.n_latent_channels}, "
            f"scale_mean={mean_desc}, scale_std={std_desc})"
        )

    def _is_scalable(self) -> bool:
        return True

    def set_scale(
        self,
        mean: Optional[Union[float, torch.Tensor]] = None,
        std: Optional[Union[float, torch.Tensor]] = None,
        tokens: Optional[Union[TokenSequence, torch.Tensor]] = None,
        per_channel: bool = True,
        eps: float = 1e-6,
    ):
        if mean is None and std is None and tokens is not None:
            t = tokens.tokens if isinstance(tokens, TokenSequence) else tokens
            if t.ndim != 3:
                raise ValueError("tokens must be 3D [B, C, T] to compute scale")
            stats = t.detach().to(dtype=torch.float32, device="cpu")
            if per_channel:
                mean = stats.mean(dim=(0, 2), keepdim=False).reshape(-1, 1)
                std = stats.std(dim=(0, 2), unbiased=False, keepdim=False).reshape(
                    -1, 1
                )
            else:
                mean = stats.mean()
                std = stats.std(unbiased=False)

        mean = 0.0 if mean is None else mean
        std = 1.0 if std is None else std

        scale_mean = _format_stats(mean, self.latent_dim, "mean")
        scale_std = _format_stats(std, self.latent_dim, "std")

        scale_std = torch.clamp(scale_std, min=eps)
        if torch.any(scale_std <= 0):
            raise ValueError("`std` must be strictly positive")

        self._scale_mean = scale_mean
        self._scale_std = scale_std

    def get_scale(self):
        return {
            "mean": None if self._scale_mean is None else self._scale_mean.clone(),
            "std": None if self._scale_std is None else self._scale_std.clone(),
        }

    def _saved_scale_path(self) -> Path:
        return self.ckpt_dir / self.name / self.saved_scale_name

    def load_saved_scale(
        self,
        path: Optional[Union[str, Path]] = None,
        scale_mean: bool = True,
        scale_std: bool = True,
        per_channel: bool = False,
    ) -> bool:
        path = self._saved_scale_path() if path is None else Path(path)
        if not path.exists():
            return False

        stats = torch.load(path, map_location="cpu", weights_only=False)
        mean_key = "mean_per_channel" if per_channel else "mean_scalar"
        std_key = "std_per_channel" if per_channel else "std_scalar"

        mean = stats.get(mean_key, None) if scale_mean else 0.0
        std = stats.get(std_key, None) if scale_std else 1.0
        self.set_scale(mean=mean, std=std, per_channel=per_channel)
        return True

    def _apply_scale(self, tokens: torch.Tensor, mean: torch.Tensor, std: torch.Tensor):
        mean = mean.to(device=tokens.device, dtype=tokens.dtype).unsqueeze(0)
        std = std.to(device=tokens.device, dtype=tokens.dtype).unsqueeze(0)
        return (tokens - mean) / std

    def _apply_unscale(
        self, tokens: torch.Tensor, mean: torch.Tensor, std: torch.Tensor
    ):
        mean = mean.to(device=tokens.device, dtype=tokens.dtype).unsqueeze(0)
        std = std.to(device=tokens.device, dtype=tokens.dtype).unsqueeze(0)
        return (tokens * std) + mean

    def scale(
        self,
        seq: TokenSequence,
        mean: Optional[Union[float, torch.Tensor]] = None,
        std: Optional[Union[float, torch.Tensor]] = None,
    ) -> TokenSequence:
        out = seq.clone()

        scale_mean = _format_stats(mean, self.latent_dim, "mean")
        scale_std = _format_stats(std, self.latent_dim, "std")
        if scale_mean is None:
            scale_mean = self._scale_mean
        if scale_std is None:
            scale_std = self._scale_std

        if scale_mean is None or scale_std is None:
            raise ValueError("No scale stats available; call set_scale() first")

        out.tokens = self._apply_scale(out.tokens, scale_mean, scale_std)
        out.scaled = True
        out.scale_mean = scale_mean.clone()
        out.scale_std = scale_std.clone()
        return out

    def unscale(
        self,
        seq: TokenSequence,
        mean: Optional[Union[float, torch.Tensor]] = None,
        std: Optional[Union[float, torch.Tensor]] = None,
    ) -> TokenSequence:
        out = seq.clone()

        if not out.scaled:
            out.scaled = False
            return out

        scale_mean = _format_stats(mean, self.latent_dim, "mean")
        scale_std = _format_stats(std, self.latent_dim, "std")
        if scale_mean is None:
            scale_mean = (
                out.scale_mean if out.scale_mean is not None else self._scale_mean
            )
        if scale_std is None:
            scale_std = out.scale_std if out.scale_std is not None else self._scale_std

        if scale_mean is None or scale_std is None:
            raise ValueError("No scale stats available to unscale tokens")

        out.tokens = self._apply_unscale(out.tokens, scale_mean, scale_std)
        out.scaled = False
        out.scale_mean = scale_mean.clone()
        out.scale_std = scale_std.clone()
        return out

    def _rms_db(self, x: torch.Tensor, eps: float = 1e-12):
        assert x.dim() == 3
        if self.loudness_exclude_silence:
            # Measure loudness over active regions only, using a relative
            # threshold so the same logic works across different drum classes.
            mono = x.abs().amax(dim=1)  # (n_batch, n_samples)
            peak = mono.amax(dim=-1, keepdim=True)  # (n_batch, 1)
            active = mono >= (peak * 0.01)  # within 40 dB of peak

            x_sq = x * x
            active = active[:, None, :].to(dtype=x_sq.dtype)
            denom = active.sum(dim=(1, 2)).clamp_min(1.0)
            ms = (x_sq * active).sum(dim=(1, 2)) / denom
        else:
            ms = (x * x).mean(dim=(1, 2))

        rms = torch.sqrt(ms + eps)
        return 20.0 * torch.log10(torch.clamp(rms, min=eps))

    def _db_to_gain(self, db: torch.Tensor, device, dtype):
        return torch.pow(torch.tensor(10.0, device=device, dtype=dtype), db / 20.0)

    def _preprocess(self, signal: AudioSignal):
        x = signal.audio_data
        assert x.ndim == 3
        n_batch, orig_n_channels, orig_signal_length = x.shape
        orig_sample_rate = signal.sample_rate
        orig_loudness = self._rms_db(x)

        if self.normalize_db is not None:
            target_db = float(self.normalize_db)
            gain_db = target_db - orig_loudness
            gain = self._db_to_gain(
                gain_db.unsqueeze(-1).unsqueeze(-1), x.device, x.dtype
            )
            x = x * gain
            x = (
                AudioSignal(x.clone(), sample_rate=orig_sample_rate)
                .ensure_max_of_audio()
                .audio_data
            )

        if self.n_channels == 1 and orig_n_channels > 1:
            x = x.reshape(n_batch * orig_n_channels, 1, orig_signal_length)
        elif self.n_channels > 1 and orig_n_channels == 1:
            assert (
                self.n_channels // orig_n_channels
            ) * orig_n_channels == self.n_channels
            x = x.repeat(1, self.n_channels // orig_n_channels, 1)
        elif self.n_channels != orig_n_channels:
            raise ValueError(
                f"Channel mismatch not supported: model expects {self.n_channels}, "
                f"but got {orig_n_channels}"
            )

        preprocessed = resample(
            AudioSignal(x, sample_rate=orig_sample_rate),
            self.sample_rate,
        )

        return (
            preprocessed,
            orig_sample_rate,
            orig_n_channels,
            orig_loudness,
            orig_signal_length,
        )

    def encode(
        self,
        signal: AudioSignal,
        scale: bool = False,
        no_grad: bool = True,
    ) -> TokenSequence:
        ctx = torch.inference_mode if no_grad else nullcontext
        with ctx():
            (
                preprocessed,
                orig_sample_rate,
                orig_n_channels,
                orig_loudness,
                orig_signal_length,
            ) = self._preprocess(signal)

            extras = {
                "sample_rate": orig_sample_rate,
                "n_channels": orig_n_channels,
                "loudness": orig_loudness,
                "signal_length": orig_signal_length,
            }

            tokens = self.model.encode(preprocessed.audio_data)
            assert tokens.ndim == 3

            out = TokenSequence(tokens=tokens, extras=extras, scaled=False)
            return self.scale(out) if scale else out

    def decode(self, tokens: TokenSequence, no_grad: bool = True) -> AudioSignal:
        ctx = torch.inference_mode if no_grad else nullcontext
        with ctx():
            seq = self.unscale(tokens) if tokens.scaled else tokens
            _tokens, extras = seq.tokens, seq.extras

            decoded = self.model.decode(_tokens)
            assert decoded.ndim == 3

            out = AudioSignal(decoded, sample_rate=self.sample_rate)

            (orig_sample_rate, orig_n_channels, orig_loudness, orig_signal_length) = (
                extras.get("sample_rate", self.sample_rate),
                extras.get("n_channels", self.n_channels),
                extras.get("loudness", None),
                extras.get("signal_length", None),
            )
            assert out.num_channels == self.n_channels

            out = resample(out, orig_sample_rate)

            if self.n_channels < orig_n_channels:
                assert self.n_channels == 1
                n_batch_folded = out.shape[0]
                fold_factor = orig_n_channels // self.n_channels
                n_batch = n_batch_folded // fold_factor
                out.audio_data = out.audio_data.view(
                    n_batch, fold_factor, out.num_channels, -1
                ).reshape(n_batch, orig_n_channels, -1)
            elif self.n_channels > orig_n_channels:
                out = out.to_mono()

            n_batch = out.shape[0]
            x = out.audio_data

            if orig_loudness is not None:
                cur_db = (
                    self._rms_db(x)
                    .to(device=x.device, dtype=x.dtype)
                    .view(n_batch, 1, 1)
                )
                target_db = orig_loudness.to(device=x.device, dtype=x.dtype).view(
                    n_batch, 1, 1
                )
                gain_db = target_db - cur_db
                gain = self._db_to_gain(gain_db, x.device, x.dtype)
                out.audio_data = x * gain
                out = out.ensure_max_of_audio()

            if orig_signal_length is not None:
                cur_len = out.audio_data.shape[-1]
                if cur_len > orig_signal_length:
                    out.audio_data = out.audio_data[..., :orig_signal_length]
                elif cur_len < orig_signal_length:
                    out = out.zero_pad_to(orig_signal_length)

            return out
