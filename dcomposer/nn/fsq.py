import torch

################################################################################
# Finite Scalar Quantization (FSQ)
################################################################################


def round_ste(x: torch.Tensor) -> torch.Tensor:
    """
    Round with straight-through gradients.
    """
    x_round = torch.round(x)
    return x + (x_round - x).detach()


class FSQ(torch.nn.Module):
    """
    Finite Scalar Quantizer for per-scalar quantization in [-1, 1] with
    defensive bounding:

    bound(z) = tanh(z + shift) * half_l - offset
    """

    def __init__(
        self,
        n_levels: int,
        eps: float = 1e-3,
    ):
        super().__init__()

        if n_levels < 2:
            raise ValueError("`n_levels` must be >= 2.")

        self.n_levels = int(n_levels)
        self.eps = float(eps)

        self.register_buffer("_levels_tensor", torch.tensor(float(self.n_levels)))
        self.register_buffer("_half_width", torch.tensor(float(self.n_levels // 2)))
        self.register_buffer(
            "_offset",
            torch.tensor(0.0 if (self.n_levels % 2 == 1) else 0.5),
        )

    @property
    def n_vocab(self) -> int:
        """
        Number of discrete values per scalar.
        """
        return self.n_levels

    def _half_l(self, z: torch.Tensor) -> torch.Tensor:
        levels = self._levels_tensor.to(device=z.device, dtype=z.dtype)
        return (levels - 1.0) * (1.0 - self.eps) / 2.0

    def _bound_raw(self, z: torch.Tensor) -> torch.Tensor:
        """
        Bounded, pre-normalized representation before quantization.
        """
        half_l = self._half_l(z)
        offset = self._offset.to(device=z.device, dtype=z.dtype)
        shift = torch.tan(offset / half_l)
        return torch.tanh(z + shift) * half_l - offset

    def bound(self, z: torch.Tensor) -> torch.Tensor:
        """
        Bounded continuous representation in approximately [-1, 1].
        """
        half_width = self._half_width.to(device=z.device, dtype=z.dtype)
        return self._bound_raw(z) / half_width

    def quantize(self, z: torch.Tensor) -> torch.Tensor:
        """
        Quantized representation in [-1, 1] with STE gradients.
        """
        half_width = self._half_width.to(device=z.device, dtype=z.dtype)
        z_q = round_ste(self._bound_raw(z))
        return z_q / half_width

    def encode(
        self, z: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Convert continuous inputs to bounded values, quantized values, and codes.

        Returns
        -------
        z_bound : torch.Tensor
            Bounded continuous values in approximately [-1, 1], same shape as `z`.
        z_quant : torch.Tensor
            Quantized values in [-1, 1], same shape as `z`.
        codes : torch.Tensor
            Integer code indices in [0, n_levels - 1], same shape as `z`.
        """
        z_bound = self.bound(z)
        z_quant = self.quantize(z)
        codes = self.to_codes(z_quant)
        return z_bound, z_quant, codes

    def to_codes(self, z: torch.Tensor) -> torch.Tensor:
        """
        Map normalized values in [-1, 1] to integer code indices.
        """
        half_width = self._half_width.to(device=z.device, dtype=z.dtype)
        codes = torch.round(z * half_width + half_width).to(torch.long)
        return codes.clamp(0, self.n_levels - 1)

    def from_codes(self, codes: torch.Tensor) -> torch.Tensor:
        """
        Map integer code indices back to normalized quantized values in [-1, 1].
        """
        half_width = self._half_width.to(device=codes.device, dtype=torch.float32)
        return (codes.to(torch.float32) - half_width) / half_width
