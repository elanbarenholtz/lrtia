"""Abstract base class for model backends."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np
import torch


@dataclass
class ModelOutput:
    """Output from model inference."""

    logits: torch.Tensor | None = None  # Shape: [batch, seq_len, vocab_size]
    log_probs: torch.Tensor | None = None  # Shape: [batch, seq_len, vocab_size]
    token_log_probs: torch.Tensor | None = None  # Shape: [batch, seq_len] (log prob of actual tokens)
    top_k_tokens: torch.Tensor | None = None  # Shape: [batch, seq_len, k]
    top_k_log_probs: torch.Tensor | None = None  # Shape: [batch, seq_len, k]
    positions_evaluated: list[int] = field(default_factory=list)

    def get_log_probs_at_positions(
        self,
        positions: list[int],
        token_ids: list[int] | None = None,
    ) -> np.ndarray:
        """Get log probabilities at specific positions.

        Args:
            positions: Token positions to get log probs for.
            token_ids: If provided, get log prob of these specific tokens.
                       Otherwise, return full distribution.

        Returns:
            Log probabilities array.
        """
        if self.log_probs is None and self.logits is not None:
            # Compute log probs from logits
            log_probs = torch.log_softmax(self.logits, dim=-1)
        elif self.log_probs is not None:
            log_probs = self.log_probs
        else:
            raise ValueError("No logits or log_probs available")

        # Handle batch dimension
        if log_probs.dim() == 3:
            log_probs = log_probs[0]  # Assume batch size 1

        result = []
        for i, pos in enumerate(positions):
            if token_ids is not None:
                # Get log prob of specific token
                result.append(log_probs[pos, token_ids[i]].item())
            else:
                # Get full distribution
                result.append(log_probs[pos].cpu().numpy())

        return np.array(result)

    def get_distribution_at_position(self, position: int) -> np.ndarray:
        """Get full probability distribution at a position.

        Args:
            position: Token position.

        Returns:
            Probability distribution (normalized).
        """
        if self.log_probs is None and self.logits is not None:
            probs = torch.softmax(self.logits, dim=-1)
        elif self.log_probs is not None:
            probs = torch.exp(self.log_probs)
        else:
            raise ValueError("No logits or log_probs available")

        if probs.dim() == 3:
            probs = probs[0]

        return probs[position].cpu().numpy()


class ModelBackend(ABC):
    """Abstract base class for model backends."""

    @abstractmethod
    def load(self) -> None:
        """Load the model into memory."""
        pass

    @abstractmethod
    def unload(self) -> None:
        """Unload the model from memory."""
        pass

    @property
    @abstractmethod
    def vocab_size(self) -> int:
        """Get vocabulary size."""
        pass

    @property
    @abstractmethod
    def max_length(self) -> int:
        """Get maximum sequence length."""
        pass

    @property
    @abstractmethod
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        pass

    @abstractmethod
    def tokenize(self, text: str) -> list[int]:
        """Tokenize text to token IDs."""
        pass

    @abstractmethod
    def decode(self, token_ids: list[int]) -> str:
        """Decode token IDs to text."""
        pass

    @abstractmethod
    def get_mask_token_id(self) -> int:
        """Get the mask/pad token ID for interventions."""
        pass

    @abstractmethod
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
            positions: If provided, only compute outputs at these positions.
            return_full_distribution: If True, return full probability distribution.
            top_k: If provided, return top-k tokens and their probabilities.

        Returns:
            ModelOutput with requested outputs.
        """
        pass

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
        next_tokens = [token_ids[p + 1] for p in positions]
        return output.get_log_probs_at_positions(positions, next_tokens)

    def __enter__(self) -> "ModelBackend":
        """Context manager entry."""
        self.load()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit."""
        self.unload()
