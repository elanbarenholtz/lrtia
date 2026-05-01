"""Probe wrapper for Llama and Mistral models.

Provides a clean interface for loading models and computing target
perplexity given a context prefix. Matches the existing notebook
implementation exactly (no KV caching, full forward pass per call).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


@dataclass
class ProbeConfig:
    hf_name: str
    use_4bit: bool = True
    bnb_quant_type: str = "nf4"
    bnb_compute_dtype: str = "float16"


class Probe:
    """Wraps an autoregressive LM for perplexity computation."""

    def __init__(self, config: ProbeConfig):
        self.config = config
        self.model = None
        self.tokenizer = None
        self.device = None

    def load(self):
        """Load model and tokenizer."""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.tokenizer = AutoTokenizer.from_pretrained(self.config.hf_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        if self.config.use_4bit and self.device == "cuda":
            dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16}
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type=self.config.bnb_quant_type,
                bnb_4bit_compute_dtype=dtype_map.get(
                    self.config.bnb_compute_dtype, torch.float16
                ),
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                self.config.hf_name,
                quantization_config=bnb_config,
                device_map="auto",
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.config.hf_name,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            )
            self.model = self.model.to(self.device)

        self.model.eval()

    def unload(self):
        """Free model from memory."""
        if self.model is not None:
            del self.model
            self.model = None
        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        import gc
        gc.collect()

    def encode(self, text: str) -> list[int]:
        """Tokenize text to token IDs (no special tokens)."""
        return self.tokenizer.encode(text, add_special_tokens=False)

    @torch.no_grad()
    def compute_ppl(self, token_ids: list[int], target_start: int, target_end: int) -> float:
        """Compute perplexity over target region given preceding context.

        Args:
            token_ids: Full sequence [context_tokens + target_tokens].
            target_start: Index of first target token.
            target_end: Index past last target token.

        Returns:
            Perplexity (exp of mean cross-entropy over target tokens).
            Returns inf if target region is too short.

        Matches the existing notebook implementation exactly:
        no KV caching, full forward pass from scratch each call.
        """
        if target_start >= target_end - 1:
            return float("inf")

        input_ids = torch.tensor([token_ids], device=self.model.device)
        outputs = self.model(input_ids)
        logits = outputs.logits[0]

        total_loss = 0.0
        count = 0
        for i in range(target_start, target_end - 1):
            log_probs = torch.log_softmax(logits[i], dim=-1)
            total_loss += -log_probs[token_ids[i + 1]].item()
            count += 1

        del outputs, logits
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return math.exp(total_loss / count) if count > 0 else float("inf")
