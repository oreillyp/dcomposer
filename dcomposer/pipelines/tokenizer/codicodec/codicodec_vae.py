from pathlib import Path

import torch
from einops import rearrange

from .core.hparams import bottleneck_channels
from .core.hparams import fac
from .core.hparams import hop
from .core.hparams import stereo
from .core.hparams_inference import sigma_rescale
from .core.inference import decode_latent_inference
from .core.inference import encode_audio_inference
from .core.models import UNet


class CodiCodecVAE(torch.nn.Module):
    """Minimal continuous CoDiCodec wrapper with encode/decode."""

    def __init__(
        self,
        checkpoint_path: Path,
        desired_channels: int = 64,
        decode_mode: str = "parallel",  # "autoregressive"
        denoising_steps: int = 5,  # 2
        max_batch_size_encode: int = 64,
        max_batch_size_decode: int = 32,
    ):
        super().__init__()
        checkpoint_path = Path(checkpoint_path)

        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"Missing tokenizer checkpoint: {checkpoint_path}. "
                "Place a local .pt file under pretrained/tokenizer/."
            )

        self.bottleneck_channels = bottleneck_channels
        self.sigma_rescale = sigma_rescale
        self.decode_latent_inference = decode_latent_inference
        self.encode_audio_inference = encode_audio_inference

        self.gen = UNet()
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        state_dict = (
            checkpoint["gen_state_dict"]
            if "gen_state_dict" in checkpoint
            else checkpoint
        )
        self.gen.load_state_dict(state_dict, strict=True)
        self.gen.eval()

        self.desired_channels = int(desired_channels)
        if self.desired_channels % self.bottleneck_channels != 0:
            raise ValueError(
                "desired_channels must be divisible by "
                f"bottleneck_channels={self.bottleneck_channels}"
            )

        self.decode_mode = decode_mode
        self.denoising_steps = denoising_steps
        self.max_batch_size_encode = int(max_batch_size_encode)
        self.max_batch_size_decode = int(max_batch_size_decode)
        self.n_channels = 2 if stereo else 1

    def _latents2dim(self, latents: torch.Tensor) -> torch.Tensor:
        return rearrange(
            latents,
            "... (l d) c -> ... l (d c)",
            d=self.desired_channels // self.bottleneck_channels,
        )

    def _dim2latents(self, latents: torch.Tensor) -> torch.Tensor:
        return rearrange(
            latents,
            "... l (d c) -> ... (l d) c",
            c=self.bottleneck_channels,
        )

    @torch.no_grad()
    def encode(self, audio: torch.Tensor) -> torch.Tensor:
        device = audio.device

        # CoDiCodec STFT frontend requires at least fac * hop samples.
        min_samples = int(fac * hop)
        if audio.shape[-1] < min_samples:
            audio = torch.nn.functional.pad(audio, (0, min_samples - audio.shape[-1]))

        latents = self.encode_audio_inference(
            audio,
            trainer=self,
            max_batch_size_encode=self.max_batch_size_encode,
            device=device,
            dont_quantize=True,
            preprocess_on_gpu=(device.type == "cuda"),
            fix_batch_size=False,
        )

        if latents.ndim == 4:
            b, chunks, n_latents, d = latents.shape
            latents = latents.reshape(b, chunks * n_latents, d)
        elif latents.ndim != 3:
            raise ValueError(
                f"Unexpected CoDiCodec latent shape {tuple(latents.shape)}"
            )

        latents = self._latents2dim(latents)
        latents = torch.atanh(torch.clamp(latents, min=-0.999999, max=0.999999))
        latents = latents / self.sigma_rescale
        return latents.transpose(1, 2)

    @torch.no_grad()
    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        device = latents.device

        latents = latents.transpose(1, 2)
        latents = torch.tanh(latents * self.sigma_rescale)
        latents = self._dim2latents(latents)

        # Core decoder expects 4D input when a batch dimension is present;
        # otherwise it may squeeze batch=1 and return channel-first audio
        # without an explicit batch dimension.
        latents_for_decode = latents.unsqueeze(1)

        audio = self.decode_latent_inference(
            latents_for_decode,
            trainer=self,
            mode=self.decode_mode,
            max_batch_size_decode=self.max_batch_size_decode,
            denoising_steps=self.denoising_steps,
            device=device,
            preprocess_on_gpu=(device.type == "cuda"),
            time_prompt=None,
        )

        if audio.ndim == 2:
            # Ambiguous 2D output: prefer [C, T] -> [1, C, T], fallback [B, T].
            if audio.shape[0] == self.n_channels:
                audio = audio.unsqueeze(0)
            else:
                audio = audio.unsqueeze(1)
        elif (
            audio.ndim == 3
            and audio.shape[-1] == self.n_channels
            and audio.shape[1] != self.n_channels
        ):
            # Convert [B, T, C] to [B, C, T] if needed.
            audio = audio.transpose(1, 2).contiguous()
        return audio
