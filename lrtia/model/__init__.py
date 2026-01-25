"""Model backends for language model inference."""

from lrtia.model.base import ModelBackend, ModelOutput
from lrtia.model.hf_backend import HuggingFaceBackend

__all__ = ["ModelBackend", "ModelOutput", "HuggingFaceBackend"]
