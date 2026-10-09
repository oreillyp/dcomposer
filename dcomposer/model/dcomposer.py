from typing import Optional

import torch

from ..constants import ACOUSTIC_TOK_OFFSET
from ..constants import BOS_TOK
from ..constants import EOS_TOK
from ..constants import MIDI_TO_TOK
from ..constants import ONSET_TOK_OFFSET
from ..nn.transformer import Transformer
from .sample import sample
from .sample import top_p_top_k


class DComposer(torch.nn.Module):
    """
    Minimal encoder-decoder transformer for DComposer.

    The encoder consumes a sequence of frozen pretrained-tokenizer latents of
    shape (n_batch, n_latent_channels, seq_len_latent). The decoder is a causal
    transformer over target token embeddings, with optional cross-attention to
    the encoded latent sequence.
    """

    def __init__(
        self,
        n_vocab: int = None,
        n_latent_channels: int = None,
        n_acoustic_tokens: int = None,
        d_model: int = 512,
        n_heads: int = 8,
        n_layers_enc: int = 8,
        n_layers_dec: int = 8,
        mult: int = 4,
        p_dropout: float = 0.0,
        bias: bool = False,
        max_len_encoder: int = 256,
        max_len_decoder: int = 2048,
        pos_enc: str = "rope",
        qk_norm: bool = True,
        use_sdpa: bool = True,
        use_encoder: bool = True,
    ):
        super().__init__()
        if n_vocab is None:
            raise ValueError("`n_vocab` must be provided.")
        if n_latent_channels is None:
            raise ValueError("`n_latent_channels` must be provided.")
        if n_acoustic_tokens is None:
            raise ValueError("`n_acoustic_tokens` must be provided.")
        self.n_vocab = int(n_vocab)
        self.n_latent_channels = int(n_latent_channels)
        self.n_acoustic_tokens = int(n_acoustic_tokens)
        self.d_model = int(d_model)
        self.use_encoder = bool(use_encoder)
        self.event_span = 2 + self.n_acoustic_tokens
        self.note_tok_lo = EOS_TOK + 1
        self.note_tok_hi = max(MIDI_TO_TOK.values())

        self.token_embed = torch.nn.Embedding(self.n_vocab, self.d_model)
        self.in_proj_enc = torch.nn.Linear(
            self.n_latent_channels, self.d_model, bias=bias
        )
        self.out_norm = torch.nn.LayerNorm(self.d_model)
        self.out_proj = torch.nn.Linear(self.d_model, self.n_vocab, bias=False)

        if self.use_encoder:
            self.encoder = Transformer(
                n_channels=self.d_model,
                n_heads=int(n_heads),
                n_layers=int(n_layers_enc),
                mult=int(mult),
                p_dropout=float(p_dropout),
                bias=bool(bias),
                max_len=int(max_len_encoder),
                pos_enc_self_attn=pos_enc,
                pos_enc_cross_attn="absolute",
                qk_norm=bool(qk_norm),
                use_sdpa=bool(use_sdpa),
                cross_attn=False,
                causal=False,
            )
        else:
            self.encoder = None

        self.decoder = Transformer(
            n_channels=self.d_model,
            n_heads=int(n_heads),
            n_layers=int(n_layers_dec),
            mult=int(mult),
            p_dropout=float(p_dropout),
            bias=bool(bias),
            max_len=int(max_len_decoder),
            pos_enc_self_attn=pos_enc,
            pos_enc_cross_attn="absolute",
            qk_norm=bool(qk_norm),
            use_sdpa=bool(use_sdpa),
            cross_attn=bool(use_encoder),
            causal=True,
        )

    def forward(
        self,
        tokens: torch.Tensor,
        input_latents: Optional[torch.Tensor] = None,
        input_latent_lengths: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        tokens : torch.Tensor
            Decoder input tokens, shape (n_batch, seq_len_dec)
        input_latents : torch.Tensor, optional
            Frozen tokenizer latents, shape (n_batch, n_latent_channels, seq_len_latent)
        input_latent_lengths : torch.Tensor, optional
            Valid latent sequence lengths, shape (n_batch,)
        """
        assert tokens.ndim == 2
        x = self.token_embed(tokens)

        c = None
        if self.use_encoder:
            assert input_latents is not None
            assert input_latents.ndim == 3
            c = input_latents.transpose(1, 2)
            c = self.in_proj_enc(c)
            c = self.encoder(c, lengths_x=input_latent_lengths)

        x = self.decoder(
            x,
            c=c,
            lengths_c=input_latent_lengths if self.use_encoder else None,
        )
        x = self.out_norm(x)
        return self.out_proj(x)

    def _encode_conditioning(
        self,
        input_latents: Optional[torch.Tensor] = None,
        input_latent_lengths: Optional[torch.Tensor] = None,
    ):
        c = None
        if self.use_encoder:
            assert input_latents is not None
            assert input_latents.ndim == 3
            c = input_latents.transpose(1, 2)
            c = self.in_proj_enc(c)
            c = self.encoder(c, lengths_x=input_latent_lengths)
        return c

    def _decode_step(
        self,
        step_tokens: torch.Tensor,
        c: Optional[torch.Tensor],
        input_latent_lengths: Optional[torch.Tensor],
        cache: Optional[list],
        position: int,
    ):
        x = self.token_embed(step_tokens)
        x, cache = self.decoder.forward_cached(
            x,
            c=c,
            lengths_c=input_latent_lengths if self.use_encoder else None,
            cache=cache,
            x_start=position,
        )
        x = self.out_norm(x)
        logits = self.out_proj(x)
        return logits, cache

    def _phase_from_prefix_len(self, prefix_len: torch.Tensor) -> torch.Tensor:
        return (prefix_len - 1) % self.event_span

    def _sampling_params_for_prefix_len(
        self,
        prefix_len,
        *,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        top_p_cls: Optional[float] = None,
        top_k_cls: Optional[int] = None,
        top_p_ons: Optional[float] = None,
        top_k_ons: Optional[int] = None,
        top_p_acoustic: Optional[float] = None,
        top_k_acoustic: Optional[int] = None,
    ):
        if isinstance(prefix_len, torch.Tensor):
            prefix_len = int(prefix_len.reshape(-1)[0].item())
        else:
            prefix_len = int(prefix_len)
        phase = (prefix_len - 1) % self.event_span
        if phase == 0:
            return (
                top_p if top_p_cls is None else top_p_cls,
                top_k if top_k_cls is None else top_k_cls,
            )
        if phase == 1:
            return (
                top_p if top_p_ons is None else top_p_ons,
                top_k if top_k_ons is None else top_k_ons,
            )
        return (
            top_p if top_p_acoustic is None else top_p_acoustic,
            top_k if top_k_acoustic is None else top_k_acoustic,
        )

    def _allowed_mask(self, prefix_len: torch.Tensor, device) -> torch.Tensor:
        """
        Build a boolean allowed-token mask for each prefix length.

        Parameters
        ----------
        prefix_len : torch.Tensor
            Shape (...,), number of tokens already present in the decoder input.
        """
        prefix_len = prefix_len.to(device=device, dtype=torch.long)
        phase = self._phase_from_prefix_len(prefix_len)
        vocab = torch.arange(self.n_vocab, device=device)
        while vocab.ndim < (prefix_len.ndim + 1):
            vocab = vocab.unsqueeze(0)

        note_vocab = (vocab >= self.note_tok_lo) & (vocab <= self.note_tok_hi)
        note_vocab = note_vocab | (vocab == EOS_TOK)
        onset_vocab = (vocab >= ONSET_TOK_OFFSET) & (vocab < ACOUSTIC_TOK_OFFSET)
        acoustic_vocab = vocab >= ACOUSTIC_TOK_OFFSET

        return (
            ((phase == 0).unsqueeze(-1) & note_vocab)
            | ((phase == 1).unsqueeze(-1) & onset_vocab)
            | ((phase >= 2).unsqueeze(-1) & acoustic_vocab)
        )

    def constrain_next_logits(
        self,
        logits: torch.Tensor,
        prefix_len: torch.Tensor,
        tokens: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Mask logits so only the valid DComposer sub-vocabulary is available at the
        next decode position for the given prefix lengths.
        """
        allowed = self._allowed_mask(prefix_len, logits.device)
        if tokens is not None and tokens.shape[1] > 2:
            last_onset = tokens[:, 2 :: self.n_acoustic_tokens + 2].amax(1)
            vocab = torch.arange(self.n_vocab, device=logits.device)
            backwards = (vocab >= ONSET_TOK_OFFSET) & (vocab < last_onset[:, None])
            onset_phase = self._phase_from_prefix_len(prefix_len) == 1
            allowed = allowed & ~(onset_phase[:, None] & backwards)
        masked = logits.masked_fill(~allowed, float("-inf"))
        return masked

    def _finish_inference(self, tokens, probs, lengths, complete_events, max_events):
        if max_events < 0:
            raise ValueError(
                "max_events must be nonnegative (0 means token-budget limited)"
            )
        lengths = torch.where(
            lengths < 0, torch.full_like(lengths, tokens.shape[1]), lengths
        )
        repaired = torch.zeros_like(lengths, dtype=torch.bool)
        if complete_events:
            rows, probabilities = [], []
            stride = self.n_acoustic_tokens + 2
            for i, length in enumerate(lengths.tolist()):
                row = tokens[i, :length]
                ended = int(row[-1]) == EOS_TOK
                count = (length - 1 - int(ended)) // stride
                if max_events:
                    count = min(count, max_events)
                end = 1 + count * stride
                repaired[i] = not ended or end + 1 != length
                rows.append(torch.cat([row[:end], row.new_tensor([EOS_TOK])]))
                probabilities.append(
                    torch.cat([probs[i, : end - 1], probs.new_ones(1)])
                    if repaired[i]
                    else probs[i, :end]
                )
            lengths = torch.tensor([len(row) for row in rows], device=tokens.device)
            tokens = torch.nn.utils.rnn.pad_sequence(
                rows, batch_first=True, padding_value=EOS_TOK
            )
            probs = torch.nn.utils.rnn.pad_sequence(probabilities, batch_first=True)
        return {
            "tokens": tokens,
            "probs": probs,
            "lengths": lengths,
            "completion_added": repaired,
        }

    def constrain_teacher_forcing_logits(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Apply position-wise vocabulary constraints to a full teacher-forcing
        logits tensor of shape (n_batch, seq_len, n_vocab).
        """
        seq_len = logits.shape[1]
        prefix_len = torch.arange(
            1, seq_len + 1, device=logits.device, dtype=torch.long
        )
        prefix_len = prefix_len[None, :].expand(logits.shape[0], -1)
        allowed = self._allowed_mask(prefix_len, logits.device)
        return logits.masked_fill(~allowed, float("-inf"))

    @torch.inference_mode()
    def inference_cached(
        self,
        input_latents: Optional[torch.Tensor] = None,
        input_latent_lengths: Optional[torch.Tensor] = None,
        max_new_tokens: int = 1024,
        prompt: Optional[torch.Tensor] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        top_p_cls: Optional[float] = None,
        top_k_cls: Optional[int] = None,
        top_p_ons: Optional[float] = None,
        top_k_ons: Optional[int] = None,
        top_p_acoustic: Optional[float] = None,
        top_k_acoustic: Optional[int] = None,
        temp: float = 1.0,
        argmax: bool = False,
        eos_threshold: Optional[float] = None,
        monotonic_onsets: bool = True,
        complete_events: bool = True,
        max_events: int = 0,
        progress=None,
    ):
        """
        Autoregressively generate target tokens until EOS is sampled/emitted or
        `max_new_tokens` is reached.
        """
        if prompt is None:
            if input_latents is None:
                batch_size, device = 1, torch.device("cpu")
            else:
                batch_size, device = input_latents.shape[0], input_latents.device
            tokens = torch.full(
                (batch_size, 1), BOS_TOK, dtype=torch.long, device=device
            )
        else:
            if input_latents is not None:
                device = input_latents.device
            else:
                device = prompt.device
            tokens = prompt.to(device=device, dtype=torch.long)
            batch_size = tokens.shape[0]

        c = self._encode_conditioning(
            input_latents=input_latents,
            input_latent_lengths=input_latent_lengths,
        )

        cache = None
        logits = None
        for position in range(tokens.shape[1]):
            step_tokens = tokens[:, position : position + 1]
            logits, cache = self._decode_step(
                step_tokens=step_tokens,
                c=c,
                input_latent_lengths=input_latent_lengths,
                cache=cache,
                position=position,
            )

        done = torch.zeros(batch_size, dtype=torch.bool, device=device)
        lengths = torch.full((batch_size,), -1, dtype=torch.long, device=device)
        probs_out = []

        for _ in range(int(max_new_tokens)):
            assert logits is not None
            next_logits = logits[:, -1, :]
            prefix_len = torch.full(
                (batch_size,),
                tokens.shape[1],
                dtype=torch.long,
                device=device,
            )
            next_logits = self.constrain_next_logits(
                next_logits, prefix_len, tokens if monotonic_onsets else None
            )
            phase_top_p, phase_top_k = self._sampling_params_for_prefix_len(
                tokens.shape[1],
                top_p=top_p,
                top_k=top_k,
                top_p_cls=top_p_cls,
                top_k_cls=top_k_cls,
                top_p_ons=top_p_ons,
                top_k_ons=top_k_ons,
                top_p_acoustic=top_p_acoustic,
                top_k_acoustic=top_k_acoustic,
            )
            next_logits = top_p_top_k(
                next_logits,
                top_p=phase_top_p,
                top_k=phase_top_k,
            )

            if eos_threshold is not None:
                probs = torch.softmax(next_logits / max(float(temp), 1e-8), dim=-1)
                eos_prob = probs[:, EOS_TOK]
                force_eos = (eos_prob >= float(eos_threshold)) & (
                    self._phase_from_prefix_len(prefix_len) == 0
                )
            else:
                force_eos = torch.zeros(batch_size, dtype=torch.bool, device=device)

            next_tok, next_prob = sample(next_logits, temp=float(temp), argmax=argmax)
            next_tok = torch.where(
                force_eos, torch.full_like(next_tok, EOS_TOK), next_tok
            )
            next_prob = torch.where(force_eos, torch.ones_like(next_prob), next_prob)
            next_tok = torch.where(done, torch.full_like(next_tok, EOS_TOK), next_tok)
            next_prob = torch.where(done, torch.ones_like(next_prob), next_prob)

            probs_out.append(next_prob)
            tokens = torch.cat([tokens, next_tok[:, None]], dim=1)
            if progress is not None and (tokens.shape[1] - 1) % 32 == 0:
                progress(f"Generating: {tokens.shape[1] - 1} tokens")
            newly_done = (~done) & (next_tok == EOS_TOK)
            lengths = torch.where(
                newly_done,
                torch.full_like(lengths, tokens.shape[1]),
                lengths,
            )
            done = done | (next_tok == EOS_TOK)
            if done.all():
                break

            logits, cache = self._decode_step(
                step_tokens=next_tok[:, None],
                c=c,
                input_latent_lengths=input_latent_lengths,
                cache=cache,
                position=tokens.shape[1] - 1,
            )

        probs_out = (
            torch.stack(probs_out, dim=1)
            if probs_out
            else torch.empty(batch_size, 0, device=device)
        )
        return self._finish_inference(
            tokens, probs_out, lengths, complete_events, max_events
        )

    @torch.inference_mode()
    def inference(
        self,
        input_latents: Optional[torch.Tensor] = None,
        input_latent_lengths: Optional[torch.Tensor] = None,
        max_new_tokens: int = 1024,
        prompt: Optional[torch.Tensor] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        top_p_cls: Optional[float] = None,
        top_k_cls: Optional[int] = None,
        top_p_ons: Optional[float] = None,
        top_k_ons: Optional[int] = None,
        top_p_acoustic: Optional[float] = None,
        top_k_acoustic: Optional[int] = None,
        temp: float = 1.0,
        argmax: bool = False,
        eos_threshold: Optional[float] = None,
        monotonic_onsets: bool = True,
        complete_events: bool = True,
        max_events: int = 0,
    ):
        """
        Legacy uncached autoregressive inference path.
        Kept as the default to avoid behavior drift in existing callsites.
        """
        if prompt is None:
            if input_latents is None:
                batch_size, device = 1, torch.device("cpu")
            else:
                batch_size, device = input_latents.shape[0], input_latents.device
            tokens = torch.full(
                (batch_size, 1),
                BOS_TOK,
                dtype=torch.long,
                device=device,
            )
        else:
            if input_latents is not None:
                device = input_latents.device
            else:
                device = prompt.device
            tokens = prompt.to(device=device, dtype=torch.long)
            batch_size = tokens.shape[0]

        done = torch.zeros(batch_size, dtype=torch.bool, device=device)
        lengths = torch.full((batch_size,), -1, dtype=torch.long, device=device)
        probs_out = []

        for _ in range(int(max_new_tokens)):
            logits = self(
                tokens,
                input_latents=input_latents,
                input_latent_lengths=input_latent_lengths,
            )
            next_logits = logits[:, -1, :]
            prefix_len = torch.full(
                (batch_size,),
                tokens.shape[1],
                dtype=torch.long,
                device=device,
            )
            next_logits = self.constrain_next_logits(
                next_logits, prefix_len, tokens if monotonic_onsets else None
            )
            phase_top_p, phase_top_k = self._sampling_params_for_prefix_len(
                tokens.shape[1],
                top_p=top_p,
                top_k=top_k,
                top_p_cls=top_p_cls,
                top_k_cls=top_k_cls,
                top_p_ons=top_p_ons,
                top_k_ons=top_k_ons,
                top_p_acoustic=top_p_acoustic,
                top_k_acoustic=top_k_acoustic,
            )
            next_logits = top_p_top_k(
                next_logits,
                top_p=phase_top_p,
                top_k=phase_top_k,
            )

            if eos_threshold is not None:
                probs = torch.softmax(next_logits / max(float(temp), 1e-8), dim=-1)
                eos_prob = probs[:, EOS_TOK]
                force_eos = (eos_prob >= float(eos_threshold)) & (
                    self._phase_from_prefix_len(prefix_len) == 0
                )
            else:
                force_eos = torch.zeros(batch_size, dtype=torch.bool, device=device)

            next_tok, next_prob = sample(next_logits, temp=float(temp), argmax=argmax)
            next_tok = torch.where(
                force_eos, torch.full_like(next_tok, EOS_TOK), next_tok
            )
            next_prob = torch.where(force_eos, torch.ones_like(next_prob), next_prob)
            next_tok = torch.where(done, torch.full_like(next_tok, EOS_TOK), next_tok)
            next_prob = torch.where(done, torch.ones_like(next_prob), next_prob)

            probs_out.append(next_prob)
            tokens = torch.cat([tokens, next_tok[:, None]], dim=1)
            newly_done = (~done) & (next_tok == EOS_TOK)
            lengths = torch.where(
                newly_done,
                torch.full_like(lengths, tokens.shape[1]),
                lengths,
            )
            done = done | (next_tok == EOS_TOK)
            if done.all():
                break

        probs_out = (
            torch.stack(probs_out, dim=1)
            if probs_out
            else torch.empty(batch_size, 0, device=device)
        )
        return self._finish_inference(
            tokens, probs_out, lengths, complete_events, max_events
        )
