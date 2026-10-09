import math
from typing import Optional
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .norm import QKNorm
from .pos_enc import apply_rope
from .pos_enc import apply_sinusoidal
from .pos_enc import build_rope_cache
from .pos_enc import build_sinusoidal_cache

################################################################################
# Multihead attention operation
################################################################################


def ensure_masks(
    n_batch: int,
    seq_len_q: int,
    seq_len_k: int,
    device,
    mask_q: Optional[torch.Tensor],
    mask_k: Optional[torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Parameters
    ----------
    n_batch : int
    seq_len_q : int
    seq_len_k : int
    mask_q : torch.Tensor
        Shape (n_batch, seq_len_q)
    mask_k : torch.Tensor
        Shape (n_batch, seq_len_k)
    """
    if mask_q is None:
        mask_q = torch.ones(n_batch, seq_len_q, dtype=torch.bool, device=device)
    if mask_k is None:
        mask_k = torch.ones(n_batch, seq_len_k, dtype=torch.bool, device=device)
    return mask_q, mask_k


def make_attn_mask(
    mask_q: torch.Tensor,
    mask_k: torch.Tensor,
    dtype,
) -> torch.Tensor:
    """
    Use "key padding mask" convention to prevent empty rows in attention score
    matrix (and thus softmax issues).

    Parameters
    ----------
    mask_q : torch.Tensor
        Query sequence mask, shape (n_batch, seq_len_q)
    mask_k : torch.Tensor
        Key sequence mask, shape (n_batch, seq_len_k)

    Returns
    -------
    torch.Tensor
        Additive attention mask for scaled_dot_product_attention, shape
        (n_batch, 1, seq_len_q, seq_len_k)
    """
    n_batch, seq_len_q = mask_q.shape
    seq_len_k = mask_k.shape[1]

    exclude = (
        (~mask_k)[:, None, :].expand(n_batch, seq_len_q, seq_len_k).unsqueeze(1)
    )  # (n_batch, 1, seq_len_q, seq_len_k)
    mask = exclude.to(dtype=dtype).masked_fill(exclude, float("-inf"))

    return mask  # (n_batch, 1, seq_len_q, seq_len_k)


def make_causal_mask(
    seq_len_q: int,
    seq_len_k: int,
    device,
    dtype,
    q_start: int = 0,
    k_start: int = 0,
) -> torch.Tensor:
    """
    Additive causal mask for self-attention / autoregressive attention.

    Returns
    -------
    torch.Tensor
        Shape (1, 1, seq_len_q, seq_len_k)
    """
    q_pos = torch.arange(q_start, q_start + seq_len_q, device=device)[:, None]
    k_pos = torch.arange(k_start, k_start + seq_len_k, device=device)[None, :]
    exclude = k_pos > q_pos
    return (
        exclude[None, None, :, :]
        .to(dtype=dtype)
        .masked_fill(exclude[None, None, :, :], float("-inf"))
    )


def sdpa_with_fallback(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    attn_mask: Optional[torch.Tensor],
    p_dropout: float,
    training: bool,
    use_sdpa: bool = True,
    causal: bool = False,
) -> torch.Tensor:
    """
    Optionally use PyTorch scaled_dot_product_attention (SDPA), which picks
    efficient attention implementations (e.g. flash attention) if available

    Parameters
    ----------
    q : torch.Tensor
        Query, shape (n_batch, n_heads, seq_len_q, head_channels)
    k : torch.Tensor
        Key, shape (n_batch, n_heads, seq_len_k, head_channels)
    v : torch.Tensor
        Value, shape (n_batch, n_heads, seq_len_k, head_channels)
    attn_mask : torch.Tensor
        Additive attention mask (0 or -inf), shape (n_batch, 1, seq_len_q, seq_len_k)

    Returns
    -------
    torch.Tensor
        Shape (n_batch, n_heads, seq_len_q, head_channels)
    """

    n_batch, n_heads, seq_len_q, head_channels = q.shape
    seq_len_k = k.shape[2]

    if use_sdpa and q.is_cuda:
        out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
            dropout_p=p_dropout if training else 0.0,
            is_causal=causal and attn_mask is None,
        )
        return out

    # Fallback
    scale = 1.0 / math.sqrt(head_channels)
    scores = torch.einsum("bhtd,bhsd->bhts", q, k) * scale
    if attn_mask is not None:
        scores = scores + attn_mask  # Additive mask
    attn = scores.softmax(dim=-1)
    if training and p_dropout > 0.0:
        attn = F.dropout(attn, p=p_dropout)
    out = torch.einsum("bhts,bhsd->bhtd", attn, v)
    return out


class MultiheadAttention(nn.Module):
    def __init__(
        self,
        n_channels: int,
        n_heads: int,
        p_dropout: float = 0.0,
        bias: bool = True,
        max_len: int = 8192,
        pos_enc: Optional[str] = "rope",
        qk_norm: bool = True,
        use_sdpa: bool = True,
        causal: bool = False,
        legacy_key_positions: bool = False,
    ):
        super().__init__()
        assert n_channels % n_heads == 0, "`n_channels` must be divisible by `n_heads`"
        assert pos_enc in ("rope", "absolute", "none", None)

        self.n_channels = n_channels
        self.n_heads = n_heads
        self.head_channels = n_channels // n_heads
        self.p_dropout = p_dropout
        self.pos_enc = pos_enc
        self.max_len = max_len
        self.use_sdpa = use_sdpa
        self.causal = causal
        self.legacy_key_positions = legacy_key_positions

        self.q_proj = nn.Linear(n_channels, n_channels, bias=bias)
        self.k_proj = nn.Linear(n_channels, n_channels, bias=bias)
        self.v_proj = nn.Linear(n_channels, n_channels, bias=bias)
        self.o_proj = nn.Linear(n_channels, n_channels, bias=bias)

        self.o_dropout = nn.Dropout(p_dropout)

        self.qk_norm = QKNorm(self.head_channels) if qk_norm else None
        self.pos_cache = None

    def _project_qkv(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        n_batch, seq_len_q, _ = q.shape
        seq_len_k = k.shape[1]
        q = (
            self.q_proj(q)
            .view(n_batch, seq_len_q, self.n_heads, self.head_channels)
            .transpose(1, 2)
        )
        k = (
            self.k_proj(k)
            .view(n_batch, seq_len_k, self.n_heads, self.head_channels)
            .transpose(1, 2)
        )
        v = (
            self.v_proj(v)
            .view(n_batch, seq_len_k, self.n_heads, self.head_channels)
            .transpose(1, 2)
        )
        return q, k, v

    def _apply_pos_enc(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        q_start: int = 0,
        k_start: int = 0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.pos_enc == "absolute":
            cache = self.pos_cache
            q = apply_sinusoidal(q, cache, start_idx=q_start)
            k = apply_sinusoidal(k, cache, start_idx=k_start)
        elif self.pos_enc == "rope":
            cos, sin = self.pos_cache
            q = apply_rope(q, cos, sin, start_idx=q_start)
            k = apply_rope(k, cos, sin, start_idx=k_start)
        return q, k

    def _attend(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask_q: Optional[torch.Tensor],
        mask_k: Optional[torch.Tensor],
        attn_mask: Optional[torch.Tensor],
        q_start: int = 0,
        k_start: int = 0,
    ) -> torch.Tensor:
        n_batch, _, seq_len_q, _ = q.shape
        seq_len_k = k.shape[2]
        device, dtype = q.device, q.dtype

        # A cached single query attends only to past/current keys. No mask or
        # GPU-to-CPU mask inspection is needed in this common inference case.
        unmasked = (
            self.use_sdpa
            and mask_q is None
            and mask_k is None
            and attn_mask is None
            and (
                not self.causal
                or (seq_len_q == 1 and q_start >= k_start + seq_len_k - 1)
            )
        )
        if unmasked:
            pad_mask = None
        else:
            mask_q, mask_k = ensure_masks(
                n_batch, seq_len_q, seq_len_k, device, mask_q, mask_k
            )
            pad_mask = make_attn_mask(mask_q, mask_k, dtype)
            if self.causal:
                pad_mask = pad_mask + make_causal_mask(
                    seq_len_q,
                    seq_len_k,
                    device,
                    dtype,
                    q_start=q_start,
                    k_start=k_start,
                )
            if attn_mask is not None:
                pad_mask = pad_mask + attn_mask

        y = sdpa_with_fallback(
            q,
            k,
            v,
            attn_mask=pad_mask,
            p_dropout=self.p_dropout,
            training=self.training,
            use_sdpa=self.use_sdpa,
            causal=self.causal and not unmasked,
        )
        y = y.transpose(1, 2).contiguous().view(n_batch, seq_len_q, self.n_channels)
        y = self.o_proj(y)
        y = self.o_dropout(y)
        if mask_q is not None:
            with torch.no_grad():
                y.masked_fill_(~mask_q[:, :, None], 0.0)
        return y

    def _maybe_build_pos_cache(self, device, dtype):
        if self.pos_enc in [None, "none"] or self.pos_cache is not None:
            return
        if self.pos_enc == "absolute":
            self.pos_cache = build_sinusoidal_cache(
                self.max_len, self.head_channels, device, dtype=torch.float32
            )
        elif self.pos_enc == "rope":
            cos, sin = build_rope_cache(
                self.max_len, self.head_channels, device, dtype=torch.float32
            )
            self.pos_cache = (cos, sin)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask_q: Optional[torch.Tensor] = None,
        mask_k: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        q : torch.Tensor
            Query, shape (n_batch, seq_len_q, n_channels)
        k : torch.Tensor
            Key, shape (n_batch, seq_len_k, n_channels)
        v : torch.Tensor
            Value, shape (n_batch, seq_len_k, n_channels)
        mask_q : torch.Tensor
            Boolean mask, `True` for valid positions; shape (n_batch, seq_len_q)
        mask_k : torch.Tensor
            Boolean mask, `True` for valid positions; shape (n_batch, seq_len_k)
        attn_mask : torch.tensor
            Additive (0, -inf) mask; shape (n_batch, 1, seq_len_q, seq_len_k)
        """

        n_batch, seq_len_q, _ = q.shape
        seq_len_k = k.shape[1]
        device, dtype = q.device, q.dtype

        q, k, v = self._project_qkv(q, k, v)

        # Positional encoding
        self._maybe_build_pos_cache(device=device, dtype=dtype)
        q, k = self._apply_pos_enc(q, k, q_start=0, k_start=0)

        # QK-Norm
        if self.qk_norm is not None:
            q, k = self.qk_norm(q, k)

        return self._attend(q, k, v, mask_q=mask_q, mask_k=mask_k, attn_mask=attn_mask)

    def forward_cached(
        self,
        q: torch.Tensor,
        k: Optional[torch.Tensor] = None,
        v: Optional[torch.Tensor] = None,
        mask_q: Optional[torch.Tensor] = None,
        mask_k: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
        cache: Optional[dict] = None,
        static_kv: bool = False,
        q_start: int = 0,
        k_start: int = 0,
    ) -> Tuple[torch.Tensor, dict]:
        if k is None:
            k = q
        if v is None:
            v = k
        if cache is None:
            cache = {}

        device, dtype = q.device, q.dtype
        self._maybe_build_pos_cache(device=device, dtype=dtype)

        if static_kv and "k" in cache and "v" in cache:
            # Audio conditioning is unchanged across autoregressive steps.
            q_proj = (
                self.q_proj(q)
                .view(q.shape[0], q.shape[1], self.n_heads, self.head_channels)
                .transpose(1, 2)
            )
            if self.pos_enc == "absolute":
                q_proj = apply_sinusoidal(q_proj, self.pos_cache, start_idx=q_start)
            elif self.pos_enc == "rope":
                q_proj = apply_rope(q_proj, *self.pos_cache, start_idx=q_start)
            if self.qk_norm is not None:
                q_proj = self.qk_norm.normalize(q_proj, self.qk_norm.g_q)
            y = self._attend(
                q_proj,
                cache["k"],
                cache["v"],
                mask_q,
                mask_k,
                attn_mask,
                q_start=q_start,
            )
            return y, cache

        q_proj, k_proj_new, v_proj_new = self._project_qkv(q, k, v)
        q_proj, k_proj_new = self._apply_pos_enc(
            q_proj,
            k_proj_new,
            q_start=q_start,
            k_start=k_start if static_kv or self.legacy_key_positions else q_start,
        )
        if self.qk_norm is not None:
            q_proj, k_proj_new = self.qk_norm(q_proj, k_proj_new)

        if static_kv:
            k_proj = cache.get("k", None)
            v_proj = cache.get("v", None)
            if k_proj is None or v_proj is None:
                k_proj = k_proj_new
                v_proj = v_proj_new
                cache = {"k": k_proj, "v": v_proj}
            mask_k_eff = mask_k
            k_start_eff = 0
        else:
            seq_new = int(k_proj_new.shape[2])
            write_start = int(q_start)
            write_end = write_start + seq_new
            if write_end > self.max_len:
                raise ValueError(
                    f"KV cache write would exceed max_len={self.max_len}: "
                    f"need {write_end}."
                )

            k_buf = cache.get("k", None)
            v_buf = cache.get("v", None)
            if k_buf is None or v_buf is None:
                n_batch = int(k_proj_new.shape[0])
                k_buf = torch.empty(
                    n_batch,
                    self.n_heads,
                    self.max_len,
                    self.head_channels,
                    device=device,
                    dtype=k_proj_new.dtype,
                )
                v_buf = torch.empty(
                    n_batch,
                    self.n_heads,
                    self.max_len,
                    self.head_channels,
                    device=device,
                    dtype=v_proj_new.dtype,
                )

            k_buf[:, :, write_start:write_end, :] = k_proj_new
            v_buf[:, :, write_start:write_end, :] = v_proj_new
            cache = {"k": k_buf, "v": v_buf, "seq_len": write_end}
            k_proj = k_buf[:, :, :write_end, :]
            v_proj = v_buf[:, :, :write_end, :]
            mask_k_eff = mask_k
            k_start_eff = 0

        y = self._attend(
            q_proj,
            k_proj,
            v_proj,
            mask_q=mask_q,
            mask_k=mask_k_eff,
            attn_mask=attn_mask,
            q_start=q_start,
            k_start=k_start_eff,
        )
        return y, cache
