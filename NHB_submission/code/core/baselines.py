"""Condition-specific shuffled baseline computation for Experiment 1A.

For condition X and context length c:
1. Get revealed = condition_context[-c:]
2. Shuffle exactly those c tokens (n_shuffles times)
3. Compute target PPL/NLL using each shuffled prefix
4. Average across shuffles

This is the core technical requirement: the shuffled baseline must use
the exact prefix tokens revealed by condition X at context length c.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

import numpy as np
import torch

from src.disruptions import get_revealed_context


@dataclass
class PPLResult:
    """PPL and NLL for one context length."""
    ppl: float
    nll: float  # mean negative log likelihood


@dataclass
class CurveResult:
    """Full PPL/NLL curves and marginals for one target × one condition."""
    # Metadata
    corpus_id: str = ""
    document_id: str = ""
    target_id: str = ""
    condition: str = ""
    cutpoint_M: int = 50
    n_shuffles: int = 50
    seed: int = 0

    # Per context-length data (c = 0..max_c)
    context_lengths: list[int] = field(default_factory=list)
    ordered_ppl: list[float] = field(default_factory=list)
    ordered_nll: list[float] = field(default_factory=list)
    shuffled_ppl_mean: list[float] = field(default_factory=list)
    shuffled_ppl_sd: list[float] = field(default_factory=list)
    shuffled_nll_mean: list[float] = field(default_factory=list)
    shuffled_nll_sd: list[float] = field(default_factory=list)

    # Marginals (d = 1..max_c)
    distances: list[int] = field(default_factory=list)
    m_ordered_ppl: list[float] = field(default_factory=list)
    m_shuffled_ppl: list[float] = field(default_factory=list)
    delta_ppl: list[float] = field(default_factory=list)
    m_ordered_nll: list[float] = field(default_factory=list)
    m_shuffled_nll: list[float] = field(default_factory=list)
    delta_nll: list[float] = field(default_factory=list)


@torch.no_grad()
def compute_target_ppl(
    model,
    tokenizer,
    context_tokens: list[int],
    target_tokens: list[int],
) -> PPLResult:
    """Compute target PPL and NLL using teacher forcing.

    Input to model: context_tokens + target_tokens
    Loss computed only over target token predictions.

    For empty context, just target tokens alone.

    Returns PPLResult with both ppl and mean nll.
    """
    if len(target_tokens) < 2:
        return PPLResult(ppl=float('inf'), nll=float('inf'))

    full_ids = list(context_tokens) + list(target_tokens)
    target_start = len(context_tokens)
    target_end = len(full_ids)

    input_ids = torch.tensor([full_ids], device=model.device)
    outputs = model(input_ids)
    logits = outputs.logits[0]

    total_nll = 0.0
    count = 0
    for i in range(target_start, target_end - 1):
        log_probs = torch.log_softmax(logits[i], dim=-1)
        total_nll += -log_probs[full_ids[i + 1]].item()
        count += 1

    del outputs, logits
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if count == 0:
        return PPLResult(ppl=float('inf'), nll=float('inf'))

    mean_nll = total_nll / count
    ppl = math.exp(mean_nll)
    return PPLResult(ppl=ppl, nll=mean_nll)


def compute_shuffled_ppl_for_condition(
    model,
    tokenizer,
    condition_context_tokens: list[int],
    target_tokens: list[int],
    c: int,
    n_shuffles: int,
    seed: int,
) -> tuple[float, float, float, float]:
    """Compute mean shuffled PPL/NLL for condition X at context length c.

    Args:
        model: loaded LLM
        tokenizer: corresponding tokenizer
        condition_context_tokens: disrupted context (temporal order, oldest to newest)
        target_tokens: target region token IDs
        c: context length (0 = empty)
        n_shuffles: number of random permutations to average
        seed: random seed for reproducibility

    Returns:
        (mean_ppl, sd_ppl, mean_nll, sd_nll) across n_shuffles
    """
    if c == 0:
        # No context to shuffle — return the no-context PPL
        result = compute_target_ppl(model, tokenizer, [], target_tokens)
        return result.ppl, 0.0, result.nll, 0.0

    # Get the exact c-token prefix for this condition
    revealed = get_revealed_context(condition_context_tokens, c)

    rng = random.Random(seed + c)  # deterministic per (seed, c)
    ppls = []
    nlls = []

    for _ in range(n_shuffles):
        shuffled = list(revealed)
        rng.shuffle(shuffled)
        result = compute_target_ppl(model, tokenizer, shuffled, target_tokens)
        if not math.isinf(result.ppl):
            ppls.append(result.ppl)
            nlls.append(result.nll)

    if not ppls:
        # All shuffles produced inf — return inf
        return float('inf'), 0.0, float('inf'), 0.0

    return (
        float(np.mean(ppls)),
        float(np.std(ppls)),
        float(np.mean(nlls)),
        float(np.std(nlls)),
    )


def compute_condition_curves(
    model,
    tokenizer,
    condition_context_tokens: list[int],
    target_tokens: list[int],
    n_shuffles: int = 50,
    seed: int = 42,
    metadata: dict | None = None,
) -> CurveResult:
    """Compute full PPL/NLL curves and corrected marginals for one condition.

    Args:
        model: loaded LLM
        tokenizer: corresponding tokenizer
        condition_context_tokens: disrupted context (temporal order)
        target_tokens: target region token IDs
        n_shuffles: shuffles per context length
        seed: base random seed
        metadata: optional metadata dict to populate CurveResult fields

    Returns:
        CurveResult with all curves and marginals.
    """
    max_c = len(condition_context_tokens)

    result = CurveResult(
        n_shuffles=n_shuffles,
        seed=seed,
    )
    if metadata:
        result.corpus_id = metadata.get('corpus_id', '')
        result.document_id = metadata.get('document_id', '')
        result.target_id = metadata.get('target_id', '')
        result.condition = metadata.get('condition', '')
        result.cutpoint_M = metadata.get('cutpoint_M', 50)

    # Compute PPL at each context length c = 0..max_c
    for c in range(max_c + 1):
        result.context_lengths.append(c)

        # Ordered
        if c == 0:
            ord_result = compute_target_ppl(model, tokenizer, [], target_tokens)
        else:
            prefix = get_revealed_context(condition_context_tokens, c)
            ord_result = compute_target_ppl(model, tokenizer, prefix, target_tokens)

        result.ordered_ppl.append(ord_result.ppl)
        result.ordered_nll.append(ord_result.nll)

        # Shuffled
        mean_ppl, sd_ppl, mean_nll, sd_nll = compute_shuffled_ppl_for_condition(
            model, tokenizer,
            condition_context_tokens, target_tokens,
            c, n_shuffles, seed,
        )
        result.shuffled_ppl_mean.append(mean_ppl)
        result.shuffled_ppl_sd.append(sd_ppl)
        result.shuffled_nll_mean.append(mean_nll)
        result.shuffled_nll_sd.append(sd_nll)

    # Compute marginals for d = 1..max_c
    for d in range(1, max_c + 1):
        result.distances.append(d)

        mo_ppl = result.ordered_ppl[d - 1] - result.ordered_ppl[d]
        ms_ppl = result.shuffled_ppl_mean[d - 1] - result.shuffled_ppl_mean[d]
        result.m_ordered_ppl.append(mo_ppl)
        result.m_shuffled_ppl.append(ms_ppl)
        result.delta_ppl.append(mo_ppl - ms_ppl)

        mo_nll = result.ordered_nll[d - 1] - result.ordered_nll[d]
        ms_nll = result.shuffled_nll_mean[d - 1] - result.shuffled_nll_mean[d]
        result.m_ordered_nll.append(mo_nll)
        result.m_shuffled_nll.append(ms_nll)
        result.delta_nll.append(mo_nll - ms_nll)

    return result
