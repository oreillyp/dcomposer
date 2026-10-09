from pathlib import Path

import torch

from .core.seanet import SEANetDecoder, SEANetEncoder


class MelodyFlowVAE(torch.nn.Module):
    """Minimal quantizer-free MelodyFlow EnCodec-style wrapper."""

    def __init__(self, checkpoint_path: Path):
        super().__init__()
        checkpoint_path = Path(checkpoint_path)

        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"Missing tokenizer checkpoint: {checkpoint_path}. "
                "Place a local .pt file under pretrained/tokenizer/."
            )

        self.encoder = SEANetEncoder(
            channels=2,
            dimension=256,
            n_filters=64,
            n_residual_layers=1,
            ratios=[8, 8, 6, 5],
            activation="snake",
            activation_params={"alpha": 1.0},
            norm="weight_norm",
            norm_params={},
            kernel_size=7,
            residual_kernel_size=3,
            last_kernel_size=7,
            dilation_base=2,
            pad_mode="reflect",
            true_skip=True,
            compress=2,
            lstm=2,
            disable_norm_outer_blocks=0,
            causal=False,
        )
        self.decoder = SEANetDecoder(
            channels=2,
            dimension=128,
            n_filters=64,
            n_residual_layers=1,
            ratios=[8, 8, 6, 5],
            activation="snake",
            activation_params={"alpha": 1.0},
            norm="weight_norm",
            norm_params={},
            kernel_size=7,
            residual_kernel_size=3,
            last_kernel_size=7,
            dilation_base=2,
            pad_mode="reflect",
            true_skip=True,
            compress=2,
            lstm=2,
            disable_norm_outer_blocks=0,
            causal=False,
            trim_right_ratio=1.0,
        )

        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            checkpoint = checkpoint["state_dict"]

        enc_sd = {k[8:]: v for k, v in checkpoint.items() if k.startswith("encoder.")}
        dec_sd = {k[8:]: v for k, v in checkpoint.items() if k.startswith("decoder.")}
        if not enc_sd or not dec_sd:
            enc_sd = {
                k[14:]: v for k, v in checkpoint.items() if k.startswith("model.encoder.")
            }
            dec_sd = {
                k[14:]: v for k, v in checkpoint.items() if k.startswith("model.decoder.")
            }

        if not enc_sd or not dec_sd:
            raise ValueError(
                "MelodyFlow checkpoint must include encoder.* and decoder.* keys"
            )

        self.encoder.load_state_dict(enc_sd, strict=True)
        self.decoder.load_state_dict(dec_sd, strict=True)
        self.eval()

        self.sample_rate = 48_000
        self.latent_dim = 128
        self.n_channels = 2
        self.hop_length = int(self.encoder.hop_length)

    @torch.no_grad()
    def encode(self, audio: torch.Tensor) -> torch.Tensor:
        out = self.encoder(audio)
        if out.ndim != 3:
            raise ValueError(f"MelodyFlow encoder returned shape {tuple(out.shape)}")
        if out.shape[1] != (2 * self.latent_dim):
            raise ValueError(
                f"Expected encoder channels {2 * self.latent_dim}, got {out.shape[1]}"
            )
        mean, scale = out.chunk(2, dim=1)
        stdev = torch.nn.functional.softplus(scale) + 1e-4
        latents = mean + torch.randn_like(mean) * stdev
        return latents

    @torch.no_grad()
    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        out = self.decoder(latents)
        if out.ndim != 3:
            raise ValueError(f"MelodyFlow decode returned shape {tuple(out.shape)}")
        return out
