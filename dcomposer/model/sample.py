import math
import torch
import torch.nn.functional as F

from typing import Iterable, Union, Optional
import numpy as np
from numpy.random import RandomState

################################################################################
# Utilities for sampling from trained TRIA model
################################################################################


def top_p_top_k(
    logits: torch.Tensor, 
    top_p: float = None, 
    top_k: int = None,
):
    """
    Adapted from `vampnet.modules.transformer.sample_from_logits` by Hugo Flores
    Garcia. See: https://github.com/hugofloresgarcia/vampnet/
    
    Parameters
    ----------
    logits : torch.Tensor
        Shape (..., n_classes)
    """
    logits = logits.clone()
    n_classes = logits.shape[-1]

    # Mask logits outside top-k by setting to -inf
    if top_k is not None and 0 < top_k < n_classes:
        thresh = logits.topk(top_k, dim=-1).values[..., -1:]  # (..., 1)
        logits[logits < thresh] = float("-inf")

    # Mask logits outside top-p by setting to -inf
    if top_p is not None and 0.0 < top_p < 1.0:
        # Sort descending
        sorted_logits, sorted_idx = logits.sort(dim=-1, descending=True)   # (..., n_classes)
        sorted_probs = F.softmax(sorted_logits, dim=-1)                    # (..., n_classes)
        cumsum = sorted_probs.cumsum(dim=-1)                               # (..., n_classes)

        # Keep at least one logit
        to_remove = cumsum > top_p
        to_remove[..., 0] = False
        remove_idx = torch.zeros_like(to_remove).scatter(-1, sorted_idx, to_remove)
        logits[remove_idx] = float("-inf")
        
    return logits


def sample(
    logits: torch.Tensor,
    temp: float,
    argmax: bool = False,
):
    """
    Adapted from `vampnet.modules.transformer.sample_from_logits` by Hugo Flores
    Garcia. See: https://github.com/hugofloresgarcia/vampnet/
    
    Parameters
    ----------
    logits : torch.Tensor
        Shape (..., n_classes)

    Returns
    -------
    torch.Tensor
        Sampled tokens, shape of `logits` with trailing `n_classes` dimension
        removed
    torch.Tensor
        Probabilities of sampled tokens, shape of `logits` with trailing 
        `n_classes` dimension removed
    """
    if temp <= 0:
        argmax = True
        temp = 1.0

    if argmax:
        sampled = logits.argmax(dim=-1)
        probs = F.softmax(
            logits, dim=-1
        ).take_along_dim(sampled.unsqueeze(-1), dim=-1).squeeze(-1)
        return sampled, probs

    probs = F.softmax(logits / temp, dim=-1)
    flat = probs.reshape(-1, probs.shape[-1])
    draws = torch.multinomial(flat, 1).squeeze(-1)
    sampled = draws.view(*probs.shape[:-1])
    chosen = probs.take_along_dim(sampled.unsqueeze(-1), dim=-1).squeeze(-1)
    return sampled, chosen


