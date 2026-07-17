"""HuggingFace Transformers model backend."""

from typing import Literal

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizer

from lrtia.model.base import ModelBackend, ModelOutput


class HuggingFaceBackend(ModelBackend):
    """Model backend using HuggingFace Transformers."""

    def __init__(
        self,
        model_name: str,
        device: str = "auto",
        dtype: Literal["float32", "float16", "bfloat16", "auto"] = "auto",
        trust_remote_code: bool = False,
    ):
        """Initialize the HuggingFace backend.

        Args:
            model_name: Model name or path (e.g., "gpt2", "meta-llama/Llama-2-7b").
            device: Device to run on ("auto", "cuda", "cpu", "mps", or specific like "cuda:0").
            dtype: Data type for model weights.
            trust_remote_code: Whether to trust remote code for custom models.
        """
        self.model_name = model_name
        self.device = self._resolve_device(device)
        self.dtype_str = dtype
        self.trust_remote_code = trust_remote_code

        self._model: PreTrainedModel | None = None
        self._tokenizer: PreTrainedTokenizer | None = None
        self._dtype: torch.dtype | None = None

    @staticmethod
    def _resolve_device(device: str) -> str:
        """Resolve 'auto' device to the best available device."""
        if device != "auto":
            return device

        if torch.cuda.is_available():
            return "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"

    def _resolve_dtype(self) -> torch.dtype:
        """Resolve the dtype string to a torch dtype."""
        if self.dtype_str == "float32":
            return torch.float32
        elif self.dtype_str == "float16":
            return torch.float16
        elif self.dtype_str == "bfloat16":
            return torch.bfloat16
        elif self.dtype_str == "auto":
            # Use float16 for CUDA/MPS, float32 for CPU
            if self.device in ("cuda", "mps") or "cuda" in self.device:
                return torch.float16
            else:
                return torch.float32
        else:
            raise ValueError(f"Unknown dtype: {self.dtype_str}")

    def load(self) -> None:
        """Load the model and tokenizer."""
        if self._model is not None:
            return

        self._dtype = self._resolve_dtype()

        # Load tokenizer
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=self.trust_remote_code,
        )

        # Ensure tokenizer has pad token
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        # Load model - device_map only works with CUDA, not MPS
        if self.device == "cuda" or self.device.startswith("cuda:"):
            self._model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                dtype=self._dtype,
                trust_remote_code=self.trust_remote_code,
                device_map=self.device,
            )
        else:
            # For CPU and MPS, load then move
            self._model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                dtype=self._dtype,
                trust_remote_code=self.trust_remote_code,
            )
            self._model = self._model.to(self.device)

        self._model.eval()

    def unload(self) -> None:
        """Unload the model from memory."""
        if self._model is not None:
            del self._model
            self._model = None

        if self._tokenizer is not None:
            del self._tokenizer
            self._tokenizer = None

        # Clear CUDA cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    @property
    def vocab_size(self) -> int:
        """Get vocabulary size."""
        if self._tokenizer is None:
            raise RuntimeError("Model not loaded. Call load() first.")
        return self._tokenizer.vocab_size

    @property
    def max_length(self) -> int:
        """Get maximum sequence length."""
        if self._model is None:
            raise RuntimeError("Model not loaded. Call load() first.")
        # Try different attributes for max length
        if hasattr(self._model.config, "max_position_embeddings"):
            return self._model.config.max_position_embeddings
        elif hasattr(self._model.config, "n_positions"):
            return self._model.config.n_positions
        else:
            return 2048  # Default fallback

    @property
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self._model is not None

    @property
    def tokenizer(self) -> PreTrainedTokenizer:
        """Get the tokenizer."""
        if self._tokenizer is None:
            raise RuntimeError("Model not loaded. Call load() first.")
        return self._tokenizer

    def tokenize(self, text: str) -> list[int]:
        """Tokenize text to token IDs."""
        if self._tokenizer is None:
            raise RuntimeError("Model not loaded. Call load() first.")
        return self._tokenizer.encode(text, add_special_tokens=False)

    def decode(self, token_ids: list[int]) -> str:
        """Decode token IDs to text."""
        if self._tokenizer is None:
            raise RuntimeError("Model not loaded. Call load() first.")
        return self._tokenizer.decode(token_ids)

    def get_mask_token_id(self) -> int:
        """Get the mask/pad token ID for interventions.

        For causal LMs, we typically use the EOS token since they don't have
        a true mask token.
        """
        if self._tokenizer is None:
            raise RuntimeError("Model not loaded. Call load() first.")

        # Prefer pad token, fall back to EOS
        if self._tokenizer.pad_token_id is not None:
            return self._tokenizer.pad_token_id
        elif self._tokenizer.eos_token_id is not None:
            return self._tokenizer.eos_token_id
        else:
            # Last resort: use 0
            return 0

    @torch.no_grad()
    def forward(
        self,
        token_ids: list[int] | list[list[int]],
        positions: list[int] | None = None,
        return_full_distribution: bool = False,
        top_k: int | None = None,
    ) -> ModelOutput:
        """Run forward pass and get model outputs.

        Args:
            token_ids: Token IDs (single sequence or batch).
            positions: If provided, only return outputs at these positions.
            return_full_distribution: If True, return full log probability distribution.
            top_k: If provided, return top-k tokens and their log probabilities.

        Returns:
            ModelOutput with requested outputs.
        """
        if self._model is None:
            raise RuntimeError("Model not loaded. Call load() first.")

        # Handle single sequence vs batch
        if isinstance(token_ids[0], int):
            token_ids = [token_ids]
            single_input = True
        else:
            single_input = False

        # Convert to tensor
        max_len = max(len(seq) for seq in token_ids)
        padded = [seq + [self.get_mask_token_id()] * (max_len - len(seq)) for seq in token_ids]
        input_ids = torch.tensor(padded, device=self.device)

        # Create attention mask
        attention_mask = torch.tensor(
            [[1] * len(seq) + [0] * (max_len - len(seq)) for seq in token_ids],
            device=self.device,
        )

        # Forward pass
        outputs = self._model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=False,
        )

        logits = outputs.logits  # [batch, seq_len, vocab_size]

        # If positions specified, slice to those positions
        if positions is not None:
            position_indices = torch.tensor(positions, device=self.device)
            logits = logits[:, position_indices, :]
            positions_evaluated = positions
        else:
            positions_evaluated = list(range(logits.shape[1]))

        # Compute log probabilities
        log_probs = torch.log_softmax(logits, dim=-1)

        # Prepare output
        output = ModelOutput(
            logits=logits if not return_full_distribution else None,
            log_probs=log_probs if return_full_distribution else None,
            positions_evaluated=positions_evaluated,
        )

        # Get top-k if requested
        if top_k is not None:
            top_k_log_probs, top_k_tokens = torch.topk(log_probs, k=top_k, dim=-1)
            output.top_k_tokens = top_k_tokens
            output.top_k_log_probs = top_k_log_probs

        # Compute token log probs (for actual next tokens)
        # This gives the log prob of the actual token at each position
        if not return_full_distribution:
            # Store logits for later use
            output.logits = logits

        if single_input:
            # Remove batch dimension for convenience
            if output.logits is not None:
                output.logits = output.logits[0:1]  # Keep as [1, seq, vocab]
            if output.log_probs is not None:
                output.log_probs = output.log_probs[0:1]
            if output.top_k_tokens is not None:
                output.top_k_tokens = output.top_k_tokens[0:1]
            if output.top_k_log_probs is not None:
                output.top_k_log_probs = output.top_k_log_probs[0:1]

        return output

    def get_token_log_probs(
        self,
        token_ids: list[int],
        positions: list[int] | None = None,
    ) -> np.ndarray:
        """Get log probabilities of actual next tokens.

        Args:
            token_ids: Full token sequence.
            positions: Positions to evaluate (default: all except last).

        Returns:
            Array of log probabilities for the actual next tokens.
        """
        if positions is None:
            positions = list(range(len(token_ids) - 1))

        output = self.forward(token_ids, positions=positions)

        # Get log prob of actual next token at each position
        log_probs_list = []
        logits = output.logits[0]  # Remove batch dim

        for i, pos in enumerate(positions):
            next_token = token_ids[pos + 1]
            log_prob = torch.log_softmax(logits[i], dim=-1)[next_token].item()
            log_probs_list.append(log_prob)

        return np.array(log_probs_list)

    def get_distributions_at_positions(
        self,
        token_ids: list[int],
        positions: list[int],
        top_k: int | None = None,
    ) -> dict:
        """Get probability distributions at specified positions.

        Args:
            token_ids: Full token sequence.
            positions: Positions to evaluate.
            top_k: If provided, only return top-k tokens.

        Returns:
            Dictionary with 'log_probs' and optionally 'top_k_tokens', 'top_k_log_probs'.
        """
        output = self.forward(
            token_ids,
            positions=positions,
            return_full_distribution=(top_k is None),
            top_k=top_k,
        )

        result = {}
        if output.log_probs is not None:
            result["log_probs"] = output.log_probs[0].cpu().numpy()
        if output.top_k_tokens is not None:
            result["top_k_tokens"] = output.top_k_tokens[0].cpu().numpy()
            result["top_k_log_probs"] = output.top_k_log_probs[0].cpu().numpy()

        return result
