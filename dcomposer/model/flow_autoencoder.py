import math
from contextlib import nullcontext
from typing import Iterable
from typing import Optional
from typing import Union

import torch

from ..nn.adaln import AdaLN
from ..nn.adaln import apply_scale_shift
from ..nn.fsq import FSQ
from ..nn.norm import RMSNorm
from ..nn.timestep import TimestepEmbedder
from ..nn.transformer import TransformerBlock
from ..nn.vq import VQ

################################################################################
# Latent autoencoder with quantized bottleneck
################################################################################


def flow(x_0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor = None):
    """
    Given original variable x_0 and timestep t (between 0 and 1), produce noisy
    interpolation phi and the corresponding flow prediction target.
    """
    if noise is None:
        noise = torch.randn_like(x_0)

    while t.ndim < x_0.ndim:
        t = t.unsqueeze(-1)

    phi = (1 - t) * noise + t * x_0

    return phi, x_0 - noise


class LatentFlowAutoencoder(torch.nn.Module):
    """
    Given a sequence of compressed audio latents of shape
    (n_batch, n_channels_in, max_len), further compress to obtain a sequence of
    "latent latents" or "summary latents" of shape
    (n_batch, n_channels_summary, n_summary) and apply quantization. This allows
    for compressing information content through a combination of reducing latent
    sequence length and quantizing to a small set of discrete codes.

    Encoding
    --------

    Summary latents are obtained by
    (1) Prepending a sequence of `n_summary` learnable "register"
        embeddings to the input latent sequence
    (2) Processing the combined sequence with a bidirectional transformer
    (3) Taking only the output embeddings for the register sequence positions
    (4) Projecting these embeddings to a dimension of `n_channels_summary`,
        resulting in a sequence of shape (n_batch, n_channels_summary, n_summary)

    Quantization
    ------------

    Summary latents are mapped to discrete codes via either:

    (1) Finite Scalar Quantization (FSQ), where each summary channel is
        quantized independently and optional groups of adjacent channels are
        packed into a single token via `fsq_group_size`, or
    (2) Vector Quantization (VQ), where each full summary vector is quantized
        to one code at each summary position.

    Quantization can optionally be skipped during training via quantizer
    dropout.

    Finally, because summary latents are quantized elementwise prior to any
    grouping/packing, we can reshape at will along the sequence dimension by a
    factor of `squeeze`. For example given `n_summary` = 128,
    `n_channels_summary` = 4`, and `squeeze` = 2, we can reshape our quantized
    latents from (n_batch, 4, 128) to (n_batch, 8, 64). Note that this can affect
    the training of a decoder, as it alters the latent sequence to increase the
    channel dimension while correspondingly shrinking the sequence dimension.

    Decoding
    --------

    Decoding is factorized into an upsampler and a decoder. First, summary
    latents are expanded to full sequence length using a learnable mask
    embedding sequence and a transformer conditioned only on the summary
    latents. The resulting full-length feature maps are then injected into a
    decoder that operates directly on noisy latent sequences while also
    receiving summary latents as direct conditioning.
    """

    def __init__(
        self,
        # Summary latents
        max_len: int = 128,
        n_channels_in: int = 128,
        n_channels_summary: int = 4,
        n_summary: int = 8,
        squeeze: int = 1,
        # Quantization
        n_fsq: int = 5,
        fsq_group_size: int = 1,
        bottleneck_type: str = "fsq",
        vq_codebook_size: int = 1024,
        vq_codebook_dim: int = None,
        p_quant_dropout: float = 0.0,
        register_init: str = "default",
        bottleneck_norm_position: str = "post_proj",
        # Decoding method
        upsample: bool = True,
        upsampler_cond: str = "prepend",  # "cross"
        decoder_cond: str = "prepend",  # "cross"
        adaln_zero: bool = True,
        # CFG
        p_cond_dropout: float = 0.0,
        # Transformer backbones
        n_channels_enc: int = 512,
        n_channels_dec: int = 512,
        mult_enc: int = 4,
        mult_dec: int = 4,
        n_layers_enc: int = 12,
        n_layers_dec: int = 12,
        n_heads_enc: int = 8,
        n_heads_dec: int = 8,
        p_dropout: float = 0.0,
        bias: bool = False,
        pos_enc: str = "rope",
        qk_norm: bool = True,
        use_sdpa: bool = True,
    ):
        """
        Parameters
        ----------
        max_len : int
            Maximum
        n_channels_in : int
            Input (latent) channel dimension
        n_channels_summary : int
            Summary latent channel dimension
        n_summary : int
            Number of summary latents
        squeeze : int
            Reshaping factor for reducing summary latent sequence dimension and
            increasing summary latent channel dimension
        n_fsq : int
            Number of FSQ levels per scalar when `bottleneck_type="fsq"`
        fsq_group_size : int
            Number of adjacent summary channels to pack into one discrete token
            at each summary position
        register_init : str
            Initialization scheme for learnable register and mask embeddings.
            `"default"` preserves the existing unit-std normal init; `"codicodec"`
            uses `dim**-0.5 * randn(...)` to match CoDiCodec.
        bottleneck_norm_position : str
            Position of the bottleneck normalization layer. `"post_proj"`
            preserves the existing behavior of normalizing summary latents after
            projection to `n_channels_summary`; `"pre_proj"` applies
            normalization in encoder hidden space immediately before the
            summary-latent projection.
        upsample : bool
            If `True`, use the factorized upsampler-decoder path; otherwise,
            decode directly from summary latents without upsampling features
        upsampler_cond : str
            Conditioning method for the summary-latent upsampler
        decoder_cond : str
            Conditioning method for the summary-latent decoder
        """

        super().__init__()

        assert upsampler_cond in ["prepend", "cross"]
        assert decoder_cond in ["prepend", "cross"]
        assert bottleneck_type in ["fsq", "vq"]
        assert register_init in ["default", "codicodec"]
        assert bottleneck_norm_position in ["post_proj", "pre_proj"]
        assert squeeze >= 1
        assert n_summary % squeeze == 0
        assert fsq_group_size >= 1
        if bottleneck_type == "fsq":
            assert n_channels_summary % fsq_group_size == 0

        # TODO: which if any of our projection layers should have bias? What is common in DiTs?
        self.in_proj_enc = torch.nn.Linear(n_channels_in, n_channels_enc, bias=bias)
        self.in_proj_up_summary = torch.nn.Linear(
            n_channels_summary * squeeze, n_channels_dec, bias=bias
        )
        self.in_proj_dec = torch.nn.Linear(n_channels_in, n_channels_dec, bias=bias)

        self.out_proj_enc = torch.nn.Linear(
            n_channels_enc, n_channels_summary, bias=bias
        )
        self.out_proj_dec = torch.nn.Linear(n_channels_dec, n_channels_in, bias=bias)

        if bottleneck_norm_position == "pre_proj":
            self.register_norm_pre = torch.nn.LayerNorm(n_channels_enc)
            self.register_norm_post = torch.nn.Identity()
        else:
            self.register_norm_pre = torch.nn.Identity()
            self.register_norm_post = torch.nn.LayerNorm(n_channels_summary)
        self.out_norm = torch.nn.LayerNorm(n_channels_dec)

        self.encoder = torch.nn.ModuleList(
            [
                TransformerBlock(
                    n_channels=n_channels_enc,
                    n_heads=n_heads_enc,
                    mult=mult_enc,
                    p_dropout=p_dropout,
                    bias=False,
                    max_len=max_len + n_summary + 1,
                    pos_enc_self_attn=pos_enc,
                    pos_enc_cross_attn="absolute",
                    qk_norm=qk_norm,
                    use_sdpa=use_sdpa,
                    cross_attn=False,
                )
                for _ in range(n_layers_enc)
            ]
        )

        if upsample:
            self.upsampler = torch.nn.ModuleList(
                [
                    TransformerBlock(
                        n_channels=n_channels_dec,
                        n_heads=n_heads_dec,
                        mult=mult_dec,
                        p_dropout=p_dropout,
                        bias=False,
                        max_len=max_len + (n_summary // squeeze) + 1,
                        pos_enc_self_attn=pos_enc,
                        pos_enc_cross_attn="absolute",
                        qk_norm=qk_norm,
                        use_sdpa=use_sdpa,
                        cross_attn=upsampler_cond == "cross",
                    )
                    for _ in range(n_layers_dec)
                ]
            )
            self.up_registers = torch.nn.Parameter(
                self._init_register_param(
                    shape=(max_len, n_channels_dec),
                    dim=n_channels_dec,
                    mode=register_init,
                )
            )
        else:
            self.upsampler = None
            self.register_parameter("up_registers", None)

        self.decoder = torch.nn.ModuleList(
            [
                TransformerBlock(
                    n_channels=n_channels_dec,
                    n_heads=n_heads_dec,
                    mult=mult_dec,
                    p_dropout=p_dropout,
                    bias=False,
                    max_len=max_len + (n_summary // squeeze) + 1,
                    pos_enc_self_attn=pos_enc,
                    pos_enc_cross_attn="absolute",
                    qk_norm=qk_norm,
                    use_sdpa=use_sdpa,
                    cross_attn=decoder_cond == "cross",
                )
                for _ in range(n_layers_dec)
            ]
        )

        self.adaln = torch.nn.ModuleList(
            [
                AdaLN(
                    n_channels_dec,
                    n_channels_dec,
                    has_cross_attn=decoder_cond == "cross",
                    adaln_zero=adaln_zero,
                )
                for _ in range(n_layers_dec + 1)  # Account for output projection layer
            ]
        )

        self.t_emb = TimestepEmbedder(n_channels_dec)

        # TODO: should these be natively low-dim (n_channels_summary), and
        # projected up to encoder channel dimension?
        self.registers = torch.nn.Parameter(
            self._init_register_param(
                shape=(n_summary, n_channels_enc),
                dim=n_channels_enc,
                mode=register_init,
            )
        )

        # Unconditional embedding for CFG
        self.uncond_emb = torch.nn.Parameter(
            torch.zeros(n_summary // squeeze, n_channels_dec)
        )

        if bottleneck_type == "fsq":
            self.bottleneck = FSQ(n_levels=n_fsq)
        else:
            self.bottleneck = VQ(
                input_dim=n_channels_summary,
                n_vocab=vq_codebook_size,
                codebook_dim=vq_codebook_dim,
            )

        # Match modern DiT/FM practice: start from a zero output head so the
        # model initially predicts a near-zero vector field.
        torch.nn.init.zeros_(self.out_proj_dec.weight)
        if self.out_proj_dec.bias is not None:
            torch.nn.init.zeros_(self.out_proj_dec.bias)

        # Attributes
        self.max_len = max_len
        self.n_channels_in = n_channels_in
        self.n_channels_summary = n_channels_summary
        self.n_summary = n_summary
        self.squeeze = squeeze
        self.n_fsq = n_fsq
        self.fsq_group_size = fsq_group_size
        self.bottleneck_type = bottleneck_type
        self.vq_codebook_size = vq_codebook_size
        self.vq_codebook_dim = vq_codebook_dim
        self.register_init = register_init
        self.bottleneck_norm_position = bottleneck_norm_position
        self.upsample = upsample
        self.p_quant_dropout = p_quant_dropout
        self.p_cond_dropout = p_cond_dropout
        self.upsampler_cond = upsampler_cond
        self.decoder_cond = decoder_cond

    @staticmethod
    def _init_register_param(
        shape: tuple[int, ...],
        dim: int,
        mode: str,
    ) -> torch.Tensor:
        if mode == "default":
            return torch.randn(*shape)
        if mode == "codicodec":
            return torch.randn(*shape) * (float(dim) ** -0.5)
        raise ValueError(f"Unknown register init mode: {mode}")

    @property
    def n_vocab(self):
        """
        Effective token vocabulary size at each emitted code position.
        """
        if self.bottleneck_type == "fsq":
            return self.bottleneck.n_vocab**self.fsq_group_size
        return self.bottleneck.n_vocab

    @property
    def n_codebooks(self):
        """
        Number of discrete tokens emitted by the quantizer.
        """
        if self.bottleneck_type == "fsq":
            return self.n_summary * (self.n_channels_summary // self.fsq_group_size)
        return self.n_summary

    def encode(self, x: torch.Tensor):
        """
        Encode latent sequence as a sequence of summary latents via register pooling.

        Parameters
        ----------
        x : torch.Tensor
            Input latents, shape (n_batch, n_channels_in, max_len)

        Returns
        -------
        latents : torch.Tensor
            Continuous summary latents of shape (n_batch, n_channels_summary, n_summary)
        """
        assert x.ndim == 3
        n_batch, n_channels_in, max_len = x.shape
        assert n_channels_in == self.n_channels_in
        assert max_len == self.max_len

        # Transpose and project to encoder dimension
        x = x.transpose(1, 2)  # (n_batch, max_len, n_channels_in)
        x = self.in_proj_enc(x)  # (n_batch, max_len, n_channels_enc)

        # Prepend learnable registers
        regs = self.registers[None, :, :].expand(
            n_batch, -1, -1
        )  # (n_batch, n_summary, n_channels_enc)
        x = torch.cat(
            [regs, x], dim=1
        )  # (n_batch, n_summary + max_len, n_channels_enc)

        for block in self.encoder:
            x = block(x)

        # Keep only register positions, project down to summary dim
        x = x[:, : self.n_summary]  # (n_batch, n_summary, n_channels_enc)
        x = self.register_norm_pre(x)
        latents = self.out_proj_enc(x)  # (n_batch, n_summary, n_channels_summary)
        latents = self.register_norm_post(
            latents
        )  # (n_batch, n_summary, n_channels_summary)

        return latents.transpose(1, 2)  # (n_batch, n_channels_summary, n_summary)

    def _quantize_with_losses(self, latents: torch.Tensor) -> dict:
        """
        Apply the configured bottleneck quantizer.

        Parameters
        ----------
        latents : torch.Tensor
            Continuous summary latents of shape (n_batch, n_channels_summary, n_summary)

        Returns
        -------
        dict
        """
        assert latents.ndim == 3
        n_batch, n_channels_summary, n_summary = latents.shape
        assert n_channels_summary == self.n_channels_summary
        assert n_summary == self.n_summary
        if self.bottleneck_type == "fsq":
            assert n_channels_summary % self.fsq_group_size == 0
            latents_prequant, latents_quant, idx = self.bottleneck.encode(latents)
            idx = idx.view(
                n_batch,
                n_channels_summary // self.fsq_group_size,
                self.fsq_group_size,
                n_summary,
            )

            base = self.bottleneck.n_vocab
            codes = torch.zeros(
                n_batch,
                n_channels_summary // self.fsq_group_size,
                n_summary,
                device=latents.device,
                dtype=torch.long,
            )
            scale = 1
            for i in range(self.fsq_group_size):
                codes = codes + idx[:, :, i, :] * scale
                scale *= base
            codes = codes.permute(0, 2, 1).reshape(n_batch, self.n_codebooks)
            zero = torch.zeros(n_batch, device=latents.device, dtype=latents.dtype)
            return {
                "prequant": latents_prequant,
                "quantized": latents_quant,
                "codes": codes,
                "commitment_loss": zero,
                "codebook_loss": zero,
            }

        z = latents.transpose(1, 2).contiguous()
        _z_e, z_quant, codes, commitment_loss, codebook_loss = self.bottleneck(z)
        latents_prequant = z.transpose(1, 2).contiguous()
        latents_quant = z_quant.transpose(1, 2).contiguous()
        codes = codes.reshape(n_batch, self.n_codebooks)
        return {
            "prequant": latents_prequant,
            "quantized": latents_quant,
            "codes": codes,
            "commitment_loss": commitment_loss,
            "codebook_loss": codebook_loss,
        }

    def quantize(self, latents: torch.Tensor):
        q = self._quantize_with_losses(latents)
        return q["prequant"], q["quantized"], q["codes"]

    def _apply_quant_dropout(
        self,
        q: dict,
        n_batch: int,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.training and self.p_quant_dropout > 0.0:
            drop_quant = torch.rand(n_batch, device=device) < self.p_quant_dropout
        else:
            drop_quant = torch.zeros(n_batch, device=device, dtype=torch.bool)

        keep_quant = (~drop_quant).to(q["prequant"].dtype)[:, None, None]
        latents_used = q["prequant"] + keep_quant * (q["quantized"] - q["prequant"])
        return latents_used, drop_quant

    def from_codes(self, codes: torch.Tensor):
        """
        Map token indices to quantized summary latents.

        Parameters
        ----------
        codes : torch.Tensor
            Token indices, shape `(n_batch, n_codebooks)`.

        Returns
        -------
        latents : torch.Tensor
             Continuous quantized summary latents, shape (n_batch, n_channels_summary, n_summary)
        """
        assert codes.ndim == 2
        n_batch, n_codes = codes.shape
        assert n_codes == self.n_codebooks

        if self.bottleneck_type == "fsq":
            base = self.bottleneck.n_vocab
            codes = codes.to(torch.long).view(
                n_batch, self.n_summary, self.n_channels_summary // self.fsq_group_size
            )
            codes = codes.permute(0, 2, 1).contiguous()

            idx = torch.empty(
                n_batch,
                self.n_channels_summary // self.fsq_group_size,
                self.fsq_group_size,
                self.n_summary,
                device=codes.device,
                dtype=torch.long,
            )
            rem = codes
            for i in range(self.fsq_group_size):
                idx[:, :, i, :] = rem % base
                rem = torch.div(rem, base, rounding_mode="floor")

            idx = idx.view(n_batch, self.n_channels_summary, self.n_summary)
            return self.bottleneck.from_codes(idx)

        codes = codes.to(torch.long).view(n_batch, self.n_summary)
        z = self.bottleneck.from_codes(codes)
        return z.transpose(1, 2).contiguous()

    def decode(
        self,
        latents: torch.Tensor,
        x_t: torch.Tensor,
        t: torch.Tensor,
        uncond: Optional[torch.Tensor] = None,
    ):
        """
        Perform a single denoising (flow-matching) step on the partially-noised
        sequence `x_t` corresponding to timestep `t`, conditioned on summary
        latents `latents`.

        Parameters
        ----------
        latents : torch.Tensor
            Continuous quantized summary latents, shape (n_batch, n_channels_summary, n_summary)
        x_t : torch.Tensor
            Partially-noised input sequence, shape (n_batch, n_channels_in, max_len)
        t : torch.Tensor
            Denoising timestep, shape (n_batch,)
        uncond : torch.Tensor
            If `True`, drop latents and perform unconditional pass; shape (n_batch,)

        Returns
        -------
        torch.Tensor
            Partially denoised output, shape (n_batch, n_channels_in, max_len)
        """
        assert latents.ndim == 3
        assert x_t.ndim == 3
        assert t.ndim == 1

        n_batch, device = latents.shape[0], latents.device
        assert x_t.shape[0] == t.shape[0] == n_batch

        # Transpose
        latents = latents.transpose(1, 2)  # (n_batch, n_summary, n_channels_summary)
        x_t = x_t.transpose(1, 2)  # (n_batch, max_len, n_channels_in)

        # Project partially-noised input to model dimension
        x_t = self.in_proj_dec(x_t)  # (n_batch, max_len, n_channels_dec)

        # Reshape quantized summary latents according to `self.squeeze` and use
        # them to build full-length conditioning features for the decoder.
        n_summary, n_channels_summary = latents.shape[1], latents.shape[2]
        assert n_summary == self.n_summary
        assert n_channels_summary == self.n_channels_summary
        assert n_summary % self.squeeze == 0

        latents = latents.reshape(
            n_batch, n_summary // self.squeeze, n_channels_summary * self.squeeze
        )
        latents = self.in_proj_up_summary(
            latents
        )  # (n_batch, n_summary // squeeze, n_channels_dec)

        # For unconditional passes, replace latents with null embedding
        if uncond is not None:
            uncond = uncond[:, None, None].float()
            uncond_emb = self.uncond_emb[None, :, :].expand(
                n_batch, -1, -1
            )  # (n_batch, n_summary // squeeze, n_channels_dec)
            latents = uncond * uncond_emb + (1 - uncond) * latents

        features = None
        if self.upsample:
            # Upsampler: stretch summary latents to a full-sequence feature map using
            # learnable mask embeddings, then collect one feature map per decoder
            # block.
            x_up = self.up_registers[None, :, :].expand(n_batch, -1, -1)
            if self.upsampler_cond == "prepend":
                x_up = torch.cat([latents, x_up], dim=1)

            features = []
            for block in self.upsampler:
                x_up = block(
                    x=x_up,
                    c=latents if self.upsampler_cond == "cross" else None,
                )

                feat = (
                    x_up[:, n_summary // self.squeeze :]
                    if self.upsampler_cond == "prepend"
                    else x_up
                )
                features.append(feat)

        # Direct summary-latent conditioning of the decoder matches CoDiCodec:
        # summary latents are either concatenated to the decoder token sequence
        # or provided via cross-attention in each transformer block.
        if self.decoder_cond == "prepend":
            x_t = torch.cat([latents, x_t], dim=1)

        # Get timestep embedding
        t_emb = self.t_emb(t)  # (n_batch, n_channels_dec)

        # Iterate over transformer blocks, injecting corresponding upsampler
        # feature maps and passing corresponding AdaLN parameters obtained from
        # timestep embedding.
        decoder_iter = (
            zip(self.decoder, self.adaln[:-1], features)
            if features is not None
            else (
                (block, adaln, None)
                for block, adaln in zip(self.decoder, self.adaln[:-1])
            )
        )
        for block, adaln, feat in decoder_iter:
            if feat is not None:
                if self.decoder_cond == "prepend":
                    x_lat, x_sig = torch.split(
                        x_t, [n_summary // self.squeeze, self.max_len], dim=1
                    )
                    x_sig = (x_sig + feat) / math.sqrt(2.0)
                    x_t = torch.cat([x_lat, x_sig], dim=1)
                else:
                    x_t = (x_t + feat) / math.sqrt(2.0)

            c_sa, c_ca, c_mlp = adaln(t_emb)

            x_t = block(
                x=x_t,
                c=latents if self.decoder_cond == "cross" else None,
                adaln_self=c_sa,
                adaln_cross=c_ca,
                adaln_mlp=c_mlp,
            )

        if self.decoder_cond == "prepend":
            x_t = x_t[:, n_summary // self.squeeze :]

        # Final projection with AdaLN
        _, _, c_out = self.adaln[-1](t_emb)  # use the "mlp" slot for output modulation
        shift, scale, _gate = c_out

        x_t = self.out_norm(x_t)
        x_t = apply_scale_shift(x_t, scale=scale, shift=shift)
        x_t = self.out_proj_dec(x_t)  # (n_batch, max_len, n_channels_in)

        return x_t.transpose(1, 2)  # (n_batch, n_channels_in, max_len)

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
    ):
        """
        Perform encoding, quantization, and decoding for a single training step,
        optionally skipping quantization. Input latents are encoded to obtain a
        summary latent sequence which is optionally quantized on a per-example
        basis. Then, a single decoding/denoising step is performed on a copy of
        the input latent sequence noised at a level determined by the given
        timestep.

        Parameters
        ----------
        x : torch.Tensor
            Input latents, shape (n_batch, n_channels_in, max_len)
        t : torch.Tensor
            Flow timestep in [0, 1], shape (n_batch,)

        Returns
        -------
        torch.Tensor
            Partially-denoised input latents, shape (n_batch, n_channels_in, max_len)
        """
        assert x.ndim == 3
        assert t.ndim == 1
        n_batch = x.shape[0]
        assert t.shape[0] == n_batch

        # Encode to obtain summary latents
        latents = self.encode(x)  # (n_batch, n_channels_summary, n_summary)

        # Quantize summary latents
        q = self._quantize_with_losses(latents)

        latents_used, drop_quant = self._apply_quant_dropout(
            q, n_batch=n_batch, device=x.device
        )

        # Apply noise to `x` at timestep `t` to obtain `x_t`
        x_t, target = flow(x_0=x, t=t)

        # Decoding/denoising step (predict flow target)
        if self.training and self.p_cond_dropout > 0.0:
            uncond = torch.rand(n_batch, device=x.device) < self.p_cond_dropout
        else:
            uncond = torch.zeros(n_batch, device=x.device, dtype=torch.bool)
        pred = self.decode(latents=latents_used, x_t=x_t, t=t, uncond=uncond)

        return pred, target, q, drop_quant, uncond

    @torch.inference_mode()
    def inference(
        self,
        codes: torch.Tensor,
        n_steps: int,
        latents: Optional[torch.Tensor] = None,
        t_init: Optional[torch.Tensor] = None,
        x_t: Optional[torch.Tensor] = None,
        cfg_weight: Optional[float] = None,
        solver: str = "midpoint",  # "euler"
        t_schedule: str = "linear",
        t_schedule_scale: float = 1.0,
    ):
        """
        Given codes, decode fully from noise (or from a provided x_t at t_init).

        """
        assert solver in ["euler", "midpoint"]
        assert t_schedule in [
            "linear",
            "cosine",
            "scaled_cosine",
            "power",
            "early_heavy",
            "late_heavy",
        ]
        assert t_schedule_scale > 0.0

        assert n_steps >= 1

        if codes is None:
            assert latents is not None
            assert latents.ndim == 3
            n_batch, device = latents.shape[0], latents.device

        if latents is None:
            assert codes.ndim == 2
            n_batch, device = codes.shape[0], codes.device

            # Map codes to quantized summary latents
            latents = self.from_codes(codes).to(
                device=device
            )  # (n_batch, n_channels_summary, n_summary)

        # Initialize timestep(s)
        if t_init is None:
            t = torch.zeros(n_batch, device=device, dtype=torch.float32)
        else:
            assert t_init.shape == (n_batch,)
            t = t_init.to(device=device, dtype=torch.float32).clamp(0.0, 1.0)

        # Initialize state
        if x_t is None:
            x = torch.randn(
                n_batch,
                self.n_channels_in,
                self.max_len,
                device=device,
                dtype=torch.float32,
            )
        else:
            assert x_t.shape == (n_batch, self.n_channels_in, self.max_len)
            x = x_t.to(device=device)
            if x.dtype != torch.float32:
                x = x.float()

        t0 = t.clone()

        def schedule_map(u: torch.Tensor) -> torch.Tensor:
            """
            Map normalized solver progress u in [0, 1] to normalized time progress.
            """
            u = u.clamp(0.0, 1.0)
            if t_schedule == "linear":
                return u
            if t_schedule == "cosine":
                return 1.0 - torch.cos(0.5 * torch.pi * u)
            if t_schedule == "scaled_cosine":
                base = 1.0 - torch.cos(0.5 * torch.pi * u)
                return base.pow(t_schedule_scale)
            if t_schedule == "power":
                return u.pow(t_schedule_scale)
            if t_schedule == "early_heavy":
                # More (smaller) steps near t ~ 0.
                return u.pow(max(t_schedule_scale, 1.0))
            if t_schedule == "late_heavy":
                # More (smaller) steps near t ~ 1.
                return u.pow(min(t_schedule_scale, 1.0))
            raise NotImplementedError(f"Unknown schedule {t_schedule}")

        def v_theta(x_in: torch.Tensor, t_in: torch.Tensor) -> torch.Tensor:
            """
            Predict flow field v(x,t | codes). If cfg_weight is set, apply CFG where the
            unconditional pass is produced exactly as in training (via uncond_emb).
            """
            if cfg_weight is None:
                return self.decode(latents=latents, x_t=x_in, t=t_in, uncond=None)

            # Unconditional/conditional flags (match training path that swaps to uncond_emb)
            uncond_true = torch.ones(n_batch, device=device, dtype=torch.bool)
            uncond_false = torch.zeros(n_batch, device=device, dtype=torch.bool)

            # Batch for efficiency: first unconditional, then conditional
            x_cat = torch.cat([x_in, x_in], dim=0)
            t_cat = torch.cat([t_in, t_in], dim=0)
            lat_cat = torch.cat([latents, latents], dim=0)
            uncond_cat = torch.cat([uncond_true, uncond_false], dim=0)

            v_cat = self.decode(latents=lat_cat, x_t=x_cat, t=t_cat, uncond=uncond_cat)
            v_u, v_c = v_cat[:n_batch], v_cat[n_batch:]

            # Standard CFG combine
            return v_u + cfg_weight * (v_c - v_u)

        # Integrate from current t to 1.0 following selected schedule.
        for i in range(n_steps):
            u0 = torch.tensor(
                float(i) / float(n_steps), device=device, dtype=torch.float32
            )
            u1 = torch.tensor(
                float(i + 1) / float(n_steps), device=device, dtype=torch.float32
            )
            s0 = schedule_map(u0)
            s1 = schedule_map(u1)

            t = (t0 + (1.0 - t0) * s0).clamp(0.0, 1.0)
            t_next = (t0 + (1.0 - t0) * s1).clamp(0.0, 1.0)
            dt = t_next - t

            if solver == "euler":
                v = v_theta(x, t)
                x = x + v * dt[:, None, None]

            elif solver == "midpoint":
                v1 = v_theta(x, t)
                t_mid = (t + 0.5 * dt).clamp(0.0, 1.0)
                x_mid = x + v1 * (0.5 * dt)[:, None, None]

                v2 = v_theta(x_mid, t_mid)
                x = x + v2 * dt[:, None, None]

            else:
                # TODO: skewed schedule? F5-TTS solver?
                raise NotImplementedError(f"Solver {solver} not implemented")

        return x
