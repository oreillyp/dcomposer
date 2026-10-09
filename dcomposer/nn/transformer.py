from typing import Optional
from typing import Tuple

import torch
import torch.nn as nn

from .adaln import apply_gate
from .adaln import apply_scale_shift
from .attention import MultiheadAttention
from .norm import RMSNorm

################################################################################
# Transformer
################################################################################


def lengths_to_mask(
    lengths: torch.Tensor, max_len: Optional[int] = None
) -> torch.Tensor:
    """
    Parameters
    ----------
    lengths : torch.Tensor
        Shape (n_batch,)
    max_len : int
    """
    if max_len is None:
        max_len = int(lengths.amax())
    rng = torch.arange(max_len, device=lengths.device)
    return rng[None, :] < lengths[:, None]  # (n_batch, max_len)


class MLP(nn.Module):
    def __init__(
        self, n_channels: int, mult: int = 4, p_dropout: float = 0.1, bias: bool = True
    ):
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Linear(n_channels, n_channels * mult),
            nn.GELU(),
            nn.Linear(n_channels * mult, n_channels),
            nn.Dropout(p_dropout),
        )

    def forward(self, x: torch.Tensor):
        assert x.ndim == 3  # (n_batch, seq_len, n_channels)
        return self.mlp(x)  # (n_batch, seq_len, n_channels)


class TransformerBlock(nn.Module):
    def __init__(
        self,
        n_channels: int,
        n_heads: int,
        mult: int = 4,
        p_dropout: float = 0.0,
        bias: bool = True,
        max_len: int = 8192,
        pos_enc_self_attn: Optional[str] = "rope",
        pos_enc_cross_attn: Optional[str] = "absolute",
        qk_norm: bool = True,
        use_sdpa: bool = True,
        cross_attn: bool = False,
        norm: str = "layer",
        causal: bool = False,
    ):
        super().__init__()

        assert norm in ["layer", "rms", "none", None]
        if norm == "rms":
            norm_cls = RMSNorm
        elif norm == "layer":
            norm_cls = nn.LayerNorm
        else:
            norm_cls = nn.Identity

        self.norm_1 = norm_cls(n_channels)
        self.self_attn = MultiheadAttention(
            n_channels=n_channels,
            n_heads=n_heads,
            p_dropout=p_dropout,
            bias=bias,
            max_len=max_len,
            pos_enc=pos_enc_self_attn,
            qk_norm=qk_norm,
            use_sdpa=use_sdpa,
            causal=causal,
        )

        self.cross_attn = cross_attn
        if cross_attn:
            self.norm_x = norm_cls(n_channels)
            self.norm_c = norm_cls(n_channels)
            self.cross = MultiheadAttention(
                n_channels=n_channels,
                n_heads=n_heads,
                p_dropout=p_dropout,
                bias=bias,
                max_len=max_len,
                pos_enc=pos_enc_cross_attn,
                qk_norm=qk_norm,
                use_sdpa=use_sdpa,
            )

        self.norm_2 = norm_cls(n_channels)
        self.mlp = MLP(n_channels=n_channels, mult=mult, p_dropout=p_dropout, bias=bias)

    def forward(
        self,
        x: torch.Tensor,
        c: Optional[torch.Tensor] = None,
        mask_x: Optional[torch.Tensor] = None,
        mask_c: Optional[torch.Tensor] = None,
        adaln_self: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = None,
        adaln_cross: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = None,
        adaln_mlp: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            Input sequence, shape (n_batch, seq_len_x, n_channels)
        c : torch.Tensor
            Conditioning sequence, shape (n_batch, seq_len_c, n_channels)
        mask_x : torch.Tensor
            Boolean mask indicating valid positions in input sequence, shape
            (n_batch, seq_len_x)
        mask_c : torch.Tensor
            Boolean mask indicating valid positions in conditioning sequence,
            shape (n_batch, seq_len_c)
        adaln_self : (shift, scale, gate)
            AdaLN params for self-attention, each tensor shape (n_batch, n_channels)
        adaln_cross : (shift, scale, gate)
            AdaLN params for cross-attention, each tensor shape (n_batch, n_channels)
        adaln_mlp : (shift, scale, gate)
            AdaLN params for MLP, each tensor shape (n_batch, n_channels)
        """

        if self.cross_attn:
            assert c is not None

        # Self-attention
        y = self.norm_1(x)
        if adaln_self is not None:
            shift, scale, gate = adaln_self
            y = apply_scale_shift(y, scale=scale, shift=shift)
        y = self.self_attn(y, y, y, mask_q=mask_x, mask_k=mask_x)
        if adaln_self is not None:
            x = apply_gate(x, y, gate=gate)
        else:
            x = x + y

        # Cross-attention
        if self.cross_attn and c is not None:
            q = self.norm_x(x)
            if adaln_cross is not None:
                shift, scale, gate = adaln_cross
                q = apply_scale_shift(q, scale=scale, shift=shift)

            k = self.norm_c(c)
            v = k
            y = self.cross(q, k, v, mask_q=mask_x, mask_k=mask_c)

            if adaln_cross is not None:
                x = apply_gate(x, y, gate=gate)
            else:
                x = x + y

        # MLP
        y = self.norm_2(x)
        if adaln_mlp is not None:
            shift, scale, gate = adaln_mlp
            y = apply_scale_shift(y, scale=scale, shift=shift)
        y = self.mlp(y)
        if adaln_mlp is not None:
            x = apply_gate(x, y, gate=gate)
        else:
            x = x + y

        # Zero invalid outputs
        if mask_x is not None:
            with torch.no_grad():
                x.masked_fill_(~mask_x[:, :, None], 0.0)

        return x

    def forward_cached(
        self,
        x: torch.Tensor,
        c: Optional[torch.Tensor] = None,
        mask_x: Optional[torch.Tensor] = None,
        mask_c: Optional[torch.Tensor] = None,
        cache: Optional[dict] = None,
        x_start: int = 0,
    ) -> Tuple[torch.Tensor, dict]:
        if cache is None:
            cache = {}

        if self.cross_attn:
            assert c is not None

        y = self.norm_1(x)
        y, self_cache = self.self_attn.forward_cached(
            y,
            y,
            y,
            mask_q=mask_x,
            mask_k=mask_x,
            cache=cache.get("self", None),
            static_kv=False,
            q_start=x_start,
            k_start=x_start,
        )
        x = x + y

        cross_cache = cache.get("cross", None)
        if self.cross_attn and c is not None:
            q = self.norm_x(x)
            k = self.norm_c(c)
            v = k
            y, cross_cache = self.cross.forward_cached(
                q,
                k,
                v,
                mask_q=mask_x,
                mask_k=mask_c,
                cache=cross_cache,
                static_kv=True,
                q_start=x_start,
                k_start=0,
            )
            x = x + y

        y = self.norm_2(x)
        y = self.mlp(y)
        x = x + y

        if mask_x is not None:
            with torch.no_grad():
                x.masked_fill_(~mask_x[:, :, None], 0.0)

        return x, {"self": self_cache, "cross": cross_cache}


class Transformer(nn.Module):
    def __init__(
        self,
        n_channels: int,
        n_heads: int,
        n_layers: int,
        mult: int,
        p_dropout: float = 0.0,
        bias: bool = True,
        max_len: int = 8192,
        pos_enc_self_attn: Optional[str] = "rope",
        pos_enc_cross_attn: Optional[str] = "absolute",
        qk_norm: bool = True,
        use_sdpa: bool = True,
        cross_attn: bool = False,
        causal: bool = False,
    ):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    n_channels=n_channels,
                    n_heads=n_heads,
                    mult=mult,
                    p_dropout=p_dropout,
                    bias=bias,
                    max_len=max_len,
                    pos_enc_self_attn=pos_enc_self_attn,
                    pos_enc_cross_attn=pos_enc_cross_attn,
                    qk_norm=qk_norm,
                    use_sdpa=use_sdpa,
                    cross_attn=cross_attn,
                    causal=causal,
                )
                for _ in range(n_layers)
            ]
        )
        self.n_channels = n_channels
        self.max_len = max_len
        self.pos_enc_self_attn = pos_enc_self_attn
        self.pos_enc_cross_attn = pos_enc_cross_attn

    @torch.no_grad()
    def _masks_from_lengths(
        self,
        mask_x: Optional[torch.Tensor],
        mask_c: Optional[torch.Tensor],
        lengths_x: Optional[torch.Tensor],
        lengths_c: Optional[torch.Tensor],
        seq_len_x: int,
        seq_len_c: Optional[int],
        device,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        if mask_x is None and lengths_x is not None:
            mask_x = lengths_to_mask(lengths_x.to(device), seq_len_x)
        if mask_c is None and lengths_c is not None:
            assert seq_len_c is not None
            mask_c = lengths_to_mask(lengths_c.to(device), seq_len_c)
        if mask_x is not None:
            mask_x = mask_x.bool()
        if mask_c is not None:
            mask_c = mask_c.bool()
        return mask_x, mask_c

    def forward(
        self,
        x: torch.Tensor,
        c: Optional[torch.Tensor] = None,
        mask_x: Optional[torch.Tensor] = None,
        mask_c: Optional[torch.Tensor] = None,
        lengths_x: Optional[torch.Tensor] = None,
        lengths_c: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            Input sequence, shape (n_batch, seq_len_x, n_channels)
        c : torch.Tensor
            Conditioning sequence, shape (n_batch, seq_len_c, n_channels)
        mask_x : torch.Tensor
            Boolean mask indicating valid positions in input sequence, shape
            (n_batch, seq_len_x)
        mask_c : torch.Tensor
            Boolean mask indicating valid positions in conditioning sequence,
            shape (n_batch, seq_len_c)
        lengths_x : torch.Tensor
            Valid lengths of input sequences, shape (n_batch,)
        lengths_c : torch.Tensor
            Valid lengths of conditioning sequences, shape (n_batch,)
        """

        assert x.ndim == 3
        n_batch, seq_len_x, n_channels = x.shape
        assert n_channels == self.n_channels
        seq_len_c = c.shape[1] if c is not None else None

        # Create valid masks from lengths if necessary
        mask_x, mask_c = self._masks_from_lengths(
            mask_x, mask_c, lengths_x, lengths_c, seq_len_x, seq_len_c, x.device
        )

        for i, block in enumerate(self.layers):
            x = block(x=x, c=c, mask_x=mask_x, mask_c=mask_c)

        return x

    def forward_cached(
        self,
        x: torch.Tensor,
        c: Optional[torch.Tensor] = None,
        mask_x: Optional[torch.Tensor] = None,
        mask_c: Optional[torch.Tensor] = None,
        lengths_x: Optional[torch.Tensor] = None,
        lengths_c: Optional[torch.Tensor] = None,
        cache: Optional[list] = None,
        x_start: int = 0,
    ) -> Tuple[torch.Tensor, list]:
        assert x.ndim == 3
        n_batch, seq_len_x, n_channels = x.shape
        assert n_channels == self.n_channels
        seq_len_c = c.shape[1] if c is not None else None

        mask_x, mask_c = self._masks_from_lengths(
            mask_x, mask_c, lengths_x, lengths_c, seq_len_x, seq_len_c, x.device
        )

        if cache is None:
            cache = [None] * len(self.layers)

        new_cache = []
        for block, layer_cache in zip(self.layers, cache):
            x, layer_cache = block.forward_cached(
                x=x,
                c=c,
                mask_x=mask_x,
                mask_c=mask_c,
                cache=layer_cache,
                x_start=x_start,
            )
            new_cache.append(layer_cache)

        return x, new_cache
