from pathlib import Path
from typing import Any, Dict, List, Literal

import torch
from torch import nn
from torch.nn.utils import weight_norm


class Activation1d(nn.Module):
    """Fallback for alias_free_torch.Activation1d (identity wrapper here)."""

    def __init__(self, activation: nn.Module):
        super().__init__()
        self.activation = activation

    def forward(self, x):
        return self.activation(x)


def snake_beta(x, alpha, beta):
    return x + (1.0 / (beta + 1e-9)) * torch.pow(torch.sin(x * alpha), 2)


class SnakeBeta(nn.Module):
    def __init__(self, in_features, alpha=1.0, alpha_trainable=True, alpha_logscale=True):
        super().__init__()
        self.alpha_logscale = alpha_logscale
        if self.alpha_logscale:
            self.alpha = nn.Parameter(torch.zeros(in_features) * alpha)
            self.beta = nn.Parameter(torch.zeros(in_features) * alpha)
        else:
            self.alpha = nn.Parameter(torch.ones(in_features) * alpha)
            self.beta = nn.Parameter(torch.ones(in_features) * alpha)
        self.alpha.requires_grad = alpha_trainable
        self.beta.requires_grad = alpha_trainable

    def forward(self, x):
        alpha = self.alpha.unsqueeze(0).unsqueeze(-1)
        beta = self.beta.unsqueeze(0).unsqueeze(-1)
        if self.alpha_logscale:
            alpha = torch.exp(alpha)
            beta = torch.exp(beta)
        return snake_beta(x, alpha, beta)


def WNConv1d(*args, **kwargs):
    return weight_norm(nn.Conv1d(*args, **kwargs))


def WNConvTranspose1d(*args, **kwargs):
    return weight_norm(nn.ConvTranspose1d(*args, **kwargs))


def get_activation(
    activation: Literal["elu", "snake", "none"],
    antialias: bool = False,
    channels: int = None,
) -> nn.Module:
    if activation == "elu":
        act = nn.ELU()
    elif activation == "snake":
        act = SnakeBeta(channels)
    elif activation == "none":
        act = nn.Identity()
    else:
        raise ValueError(f"Unknown activation {activation}")

    if antialias:
        act = Activation1d(act)

    return act


class ResidualUnit(nn.Module):
    def __init__(self, in_channels, out_channels, dilation, use_snake=False, antialias_activation=False):
        super().__init__()
        padding = (dilation * (7 - 1)) // 2
        self.layers = nn.Sequential(
            get_activation("snake" if use_snake else "elu", antialias=antialias_activation, channels=out_channels),
            WNConv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=7, dilation=dilation, padding=padding),
            get_activation("snake" if use_snake else "elu", antialias=antialias_activation, channels=out_channels),
            WNConv1d(in_channels=out_channels, out_channels=out_channels, kernel_size=1),
        )

    def forward(self, x):
        return self.layers(x) + x


class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride, use_snake=False, antialias_activation=False):
        super().__init__()
        self.layers = nn.Sequential(
            ResidualUnit(in_channels, in_channels, dilation=1, use_snake=use_snake),
            ResidualUnit(in_channels, in_channels, dilation=3, use_snake=use_snake),
            ResidualUnit(in_channels, in_channels, dilation=9, use_snake=use_snake),
            get_activation("snake" if use_snake else "elu", antialias=antialias_activation, channels=in_channels),
            WNConv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=2 * stride, stride=stride, padding=(stride + 1) // 2),
        )

    def forward(self, x):
        return self.layers(x)


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride, use_snake=False, antialias_activation=False):
        super().__init__()
        self.layers = nn.Sequential(
            get_activation("snake" if use_snake else "elu", antialias=antialias_activation, channels=in_channels),
            WNConvTranspose1d(in_channels=in_channels, out_channels=out_channels, kernel_size=2 * stride, stride=stride, padding=(stride + 1) // 2),
            ResidualUnit(out_channels, out_channels, dilation=1, use_snake=use_snake),
            ResidualUnit(out_channels, out_channels, dilation=3, use_snake=use_snake),
            ResidualUnit(out_channels, out_channels, dilation=9, use_snake=use_snake),
        )

    def forward(self, x):
        return self.layers(x)


class OobleckEncoder(nn.Module):
    def __init__(self, in_channels=2, channels=128, latent_dim=32, c_mults=None, strides=None, use_snake=False, antialias_activation=False):
        super().__init__()
        if c_mults is None:
            c_mults = [1, 2, 4, 8]
        if strides is None:
            strides = [2, 4, 8, 8]

        c_mults = [1] + c_mults
        layers = [WNConv1d(in_channels=in_channels, out_channels=c_mults[0] * channels, kernel_size=7, padding=3)]
        for i in range(len(c_mults) - 1):
            layers.append(EncoderBlock(c_mults[i] * channels, c_mults[i + 1] * channels, stride=strides[i], use_snake=use_snake))
        layers.extend([
            get_activation("snake" if use_snake else "elu", antialias=antialias_activation, channels=c_mults[-1] * channels),
            WNConv1d(in_channels=c_mults[-1] * channels, out_channels=latent_dim, kernel_size=3, padding=1),
        ])
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class OobleckDecoder(nn.Module):
    def __init__(self, out_channels=2, channels=128, latent_dim=32, c_mults=None, strides=None, use_snake=False, antialias_activation=False, final_tanh=False):
        super().__init__()
        if c_mults is None:
            c_mults = [1, 2, 4, 8]
        if strides is None:
            strides = [2, 4, 8, 8]

        c_mults = [1] + c_mults
        layers = [WNConv1d(in_channels=latent_dim, out_channels=c_mults[-1] * channels, kernel_size=7, padding=3)]
        for i in range(len(c_mults) - 1, 0, -1):
            layers.append(DecoderBlock(c_mults[i] * channels, c_mults[i - 1] * channels, stride=strides[i - 1], use_snake=use_snake, antialias_activation=antialias_activation))
        layers.extend([
            get_activation("snake" if use_snake else "elu", antialias=antialias_activation, channels=c_mults[0] * channels),
            WNConv1d(in_channels=c_mults[0] * channels, out_channels=out_channels, kernel_size=7, padding=3, bias=False),
            nn.Tanh() if final_tanh else nn.Identity(),
        ])
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


def vae_sample(mean, scale):
    stdev = nn.functional.softplus(scale) + 1e-4
    var = stdev * stdev
    logvar = torch.log(var)
    latents = torch.randn_like(mean) * stdev + mean
    kl = (mean * mean + var - logvar - 1).sum(1).mean()
    return latents, kl


class VAEBottleneck(nn.Module):
    def encode(self, x, return_info=False, **kwargs):
        info = {}
        mean, scale = x.chunk(2, dim=1)
        x, kl = vae_sample(mean, scale)
        info["kl"] = kl
        if return_info:
            return x, info
        return x

    def decode(self, x):
        return x


class AudioAutoencoder(nn.Module):
    def __init__(self, encoder, decoder, bottleneck, latent_dim, downsampling_ratio, sample_rate, io_channels):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.bottleneck = bottleneck
        self.latent_dim = latent_dim
        self.downsampling_ratio = downsampling_ratio
        self.sample_rate = sample_rate
        self.out_channels = io_channels

    def encode(self, audio):
        latents = self.encoder(audio)
        latents, _ = self.bottleneck.encode(latents, return_info=True)
        return latents

    def decode(self, latents):
        latents = self.bottleneck.decode(latents)
        return self.decoder(latents)

    def encode_audio(self, audio, **kwargs):
        return self.encode(audio)

    def decode_audio(self, latents, **kwargs):
        return self.decode(latents)


def build_stable_audio_autoencoder() -> AudioAutoencoder:
    # Stable Audio 2.0 VAE defaults (minimal fixed config).
    sample_rate = 44_100
    io_channels = 2
    downsampling_ratio = 2048
    latent_dim = 64

    encoder = OobleckEncoder(
        in_channels=2,
        channels=128,
        c_mults=[1, 2, 4, 8, 16],
        strides=[2, 4, 4, 8, 8],
        latent_dim=128,
        use_snake=True,
    )
    decoder = OobleckDecoder(
        out_channels=2,
        channels=128,
        c_mults=[1, 2, 4, 8, 16],
        strides=[2, 4, 4, 8, 8],
        latent_dim=64,
        use_snake=True,
        final_tanh=False,
    )
    bottleneck = VAEBottleneck()

    return AudioAutoencoder(
        encoder=encoder,
        decoder=decoder,
        bottleneck=bottleneck,
        latent_dim=latent_dim,
        downsampling_ratio=downsampling_ratio,
        sample_rate=sample_rate,
        io_channels=io_channels,
    )


class StableAudioVAE(torch.nn.Module):
    """Minimal Stable Audio VAE wrapper with local model code only."""

    def __init__(self, checkpoint_path: Path):
        super().__init__()
        checkpoint_path = Path(checkpoint_path)

        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"Missing tokenizer checkpoint: {checkpoint_path}. "
                "Place a local .pt file under pretrained/tokenizer/."
            )
        self.model = build_stable_audio_autoencoder()

        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            checkpoint = checkpoint["state_dict"]
        elif isinstance(checkpoint, dict) and "model" in checkpoint:
            checkpoint = checkpoint["model"]

        self.model.load_state_dict(checkpoint, strict=True)
        self.model.eval()

        self.sample_rate = self.model.sample_rate
        self.latent_dim = self.model.latent_dim
        self.n_channels = self.model.out_channels
        self.hop_length = int(self.model.downsampling_ratio)

    @torch.no_grad()
    def encode(self, audio: torch.Tensor) -> torch.Tensor:
        out = self.model.encode_audio(audio)
        if out.ndim != 3:
            raise ValueError(f"Stable Audio encode returned shape {tuple(out.shape)}")
        return out

    @torch.no_grad()
    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        out = self.model.decode_audio(latents)
        if out.ndim != 3:
            raise ValueError(f"Stable Audio decode returned shape {tuple(out.shape)}")
        return out
