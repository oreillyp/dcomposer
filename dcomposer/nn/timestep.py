import math
from typing import Optional

import torch

################################################################################
# Timestep embedder for diffusion/flow models
################################################################################


def sinusoidal_timestep_embedding(
    t: torch.Tensor,
    dim: int,
    max_period: float = 10000.0,
    dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    """
    Create sinusoidal timestep embeddings.

    Parameters
    ----------
    t : torch.Tensor
        Timestep, shape (n_batch,)
    dim : int
        Embedding dimension
    max_period : float
        Controls minimum frequency (higher => slower)
    dtype : torch.dtype
        Output dtype

    Returns
    -------
    torch.Tensor
        Embeddings of shape (n_batch, dim).
    """
    if t.ndim == 2 and t.shape[1] == 1:
        t = t[:, 0]
    t = t.to(torch.float32) if not t.is_floating_point() else t

    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(0, half, device=t.device, dtype=torch.float32) / half
    )
    args = t[:, None] * freqs[None, :]  # (n_batch, half)
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)  # (n_batch, 2*half)

    if dim % 2 == 1:
        emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)

    out_dtype = dtype if dtype is not None else (t.dtype if t.is_floating_point() else torch.float32)
    return emb.to(out_dtype)


class TimestepEmbedder(torch.nn.Module):
    """
    Sinusoidal timestep embedding followed by MLP.
    """

    def __init__(
        self,
        out_channels: int,
        emb_channels: Optional[int] = None,
        hidden_channels: Optional[int] = None,
        scale: float = 1000.0,
        max_period: float = 10000.0,
    ):
        super().__init__()

        # Defaults
        emb_channels = emb_channels or out_channels
        hidden_channels = hidden_channels or out_channels

        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(emb_channels, hidden_channels),
            torch.nn.SiLU(),
            torch.nn.Linear(hidden_channels, out_channels),
        )
        
        self.emb_channels = emb_channels
        self.scale = scale
        self.max_period = max_period

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        x = sinusoidal_timestep_embedding(t * self.scale, self.emb_channels, max_period=self.max_period)
        return self.mlp(x)  # (n_batch, n_channels)

