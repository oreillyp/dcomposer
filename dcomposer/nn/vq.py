import torch
import torch.nn.functional as F

################################################################################
# Vector Quantization (VQ)
################################################################################


class VQ(torch.nn.Module):
    """
    Straight-through vector quantizer with optional low-dimensional codebook space.
    """

    def __init__(
        self,
        input_dim: int,
        n_vocab: int,
        codebook_dim: int = None,
    ):
        super().__init__()
        if input_dim < 1:
            raise ValueError("`input_dim` must be >= 1.")
        if n_vocab < 2:
            raise ValueError("`n_vocab` must be >= 2.")

        self.input_dim = int(input_dim)
        self.n_vocab = int(n_vocab)
        self.codebook_dim = int(codebook_dim or input_dim)

        self.in_proj = torch.nn.Linear(self.input_dim, self.codebook_dim, bias=False)
        self.out_proj = torch.nn.Linear(self.codebook_dim, self.input_dim, bias=False)
        self.codebook = torch.nn.Embedding(self.n_vocab, self.codebook_dim)
        torch.nn.init.normal_(self.codebook.weight, std=self.codebook_dim**-0.5)

    def forward(
        self, z: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Training path: quantize vectors and return auxiliary VQ losses.
        """
        assert z.shape[-1] == self.input_dim

        z_e = self.in_proj(z)
        flat = F.normalize(z_e.reshape(-1, self.codebook_dim), dim=-1)
        codebook = F.normalize(self.codebook.weight, dim=-1)
        dist = (
            flat.pow(2).sum(dim=1, keepdim=True)
            - 2.0 * flat @ codebook.t()
            + codebook.pow(2).sum(dim=1)[None, :]
        )
        codes = dist.argmin(dim=1)
        z_q_e = self.codebook(codes).view_as(z_e)

        reduce_dims = tuple(range(1, z_e.ndim))
        commitment_loss = F.mse_loss(z_e, z_q_e.detach(), reduction="none").mean(
            dim=reduce_dims
        )
        codebook_loss = F.mse_loss(z_q_e, z_e.detach(), reduction="none").mean(
            dim=reduce_dims
        )

        z_q_e = z_e + (z_q_e - z_e).detach()
        z_q = self.out_proj(z_q_e)
        return z_e, z_q, codes.view(*z.shape[:-1]), commitment_loss, codebook_loss

    def quantize(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Inference path: quantize vectors without returning auxiliary losses.
        """
        _z_e, z_q, codes, _commitment_loss, _codebook_loss = self.forward(z)
        return z_q, codes.view(*z.shape[:-1])

    def from_codes(self, codes: torch.Tensor) -> torch.Tensor:
        return self.out_proj(self.codebook(codes.to(torch.long)))
