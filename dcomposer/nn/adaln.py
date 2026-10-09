from typing import Optional, Tuple
from contextlib import nullcontext

import torch

################################################################################
# Adaptive layer normalization (AdaLN) parameterization
################################################################################


def apply_scale_shift(x_norm: torch.Tensor, scale: torch.Tensor, shift: torch.Tensor) -> torch.Tensor:
    """
    Parameters
    ----------
    x_norm : torch.Tensor
        Normalized inputs of shape (n_batch, n_seq, n_channels)
    scale : torch.Tensor
        Scale parameters of shape (n_batch, n_channels)
    shift : torch.Tensor
        Shift parameters of shape (n_batch, n_channels)

    Returns
    -------
    torch.Tensor
        Modulated inputs of shape (n_batch, n_seq, n_channels)
    """
    device = x_norm.device

    if device.type == "cuda":
        ctx = torch.autocast(device_type="cuda", enabled=False)
    else:
        ctx = nullcontext()
    with ctx:
        x = x_norm.float()
        s = scale.float()
        b = shift.float()
        out = x * (1.0 + s[:, None, :]) + b[:, None, :]
    return out.to(dtype=x_norm.dtype)


def apply_gate(x: torch.Tensor, y: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
    """
    Parameters
    ----------
    x : torch.Tensor
        Residual stream of shape (n_batch, n_seq, n_channels)
    y : torch.Tensor
        Branch output to be gated and added, shape (n_batch, n_seq, n_channels)
    gate : torch.Tensor
        Gate parameters of shape (n_batch, n_channels)

    Returns
    -------
    torch.Tensor
        Gated residual output of shape (n_batch, n_seq, n_channels)
    """
    device = x.device

    if device.type == "cuda":
        ctx = torch.autocast(device_type="cuda", enabled=False)
    else:
        ctx = nullcontext()
    with ctx:
        x0 = x.float()
        y0 = y.float()
        g = gate.float()
        out = x0 + y0 * g[:, None, :]
    return out.to(dtype=x.dtype)


class AdaLN(torch.nn.Module):
    """
    Map a conditioning vector of shape (n_batch, n_channels_in) to scale/shift/gate
    tensors of shape (n_batch, n_channels_out) for modulating inputs to self-
    attention, cross-attention, and MLP in transformer block.
    """

    def __init__(
        self,
        n_channels_in: int,
        n_channels_out: int,
        n_channels_hidden: Optional[int] = None,
        has_cross_attn: bool = False,
        adaln_zero: bool = True,
    ):
        super().__init__()

        n_channels_hidden = n_channels_hidden or n_channels_in

        self.n_channels_in = n_channels_in
        self.n_channels_out = n_channels_out
        self.has_cross_attn = has_cross_attn
        self.n_sets = 3 if has_cross_attn else 2

        self.net = torch.nn.Sequential(
            torch.nn.Linear(n_channels_in, n_channels_hidden),
            torch.nn.SiLU(),
            torch.nn.Linear(
                n_channels_hidden,
                self.n_sets * 3 * n_channels_out
            ),
        )

        # Initialize final layer to zeros so that scale/shift/gate start at 0
        if adaln_zero:
            torch.nn.init.zeros_(self.net[-1].weight)
            torch.nn.init.zeros_(self.net[-1].bias)

    def forward(
        self, cond: torch.Tensor
    ) -> Tuple[
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ]:
        """
        Parameters
        ----------
        cond : torch.Tensor
            Conditioning, shape (n_batch, n_channels_in)

        Returns
        -------
        (shift, scale, gate) tuples, each tensor of shape (n_batch, n_channels_out)
        """
        x = self.net(cond).view(cond.shape[0], self.n_sets, 3, self.n_channels_out)

        shift = x[:, :, 0]
        scale = x[:, :, 1]
        gate = x[:, :, 2]

        self_p = (shift[:, 0], scale[:, 0], gate[:, 0])
        if self.has_cross_attn:
            cross_p = (shift[:, 1], scale[:, 1], gate[:, 1])
            mlp_p = (shift[:, 2], scale[:, 2], gate[:, 2])
        else:
            cross_p = None
            mlp_p = (shift[:, 1], scale[:, 1], gate[:, 1])

        return self_p, cross_p, mlp_p