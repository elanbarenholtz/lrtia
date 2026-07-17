"""Configuration schema for LRTIA using Pydantic."""

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator


class ModelConfig(BaseModel):
    """Configuration for the language model."""

    name: str = Field(default="gpt2", description="Model name or path")
    backend: Literal["huggingface"] = Field(
        default="huggingface", description="Model backend to use"
    )
    device: str = Field(default="auto", description="Device to run model on (auto, cuda, cpu, mps)")
    dtype: Literal["float32", "float16", "bfloat16", "auto"] = Field(
        default="auto", description="Data type for model weights"
    )


class WindowingConfig(BaseModel):
    """Configuration for text windowing."""

    window_tokens: int = Field(default=2048, ge=128, description="Size of each evaluation window")
    stride_tokens: int = Field(default=512, ge=1, description="Stride between windows")

    @model_validator(mode="after")
    def validate_stride(self) -> "WindowingConfig":
        if self.stride_tokens > self.window_tokens:
            raise ValueError("stride_tokens cannot be greater than window_tokens")
        return self


class TargetsConfig(BaseModel):
    """Configuration for target position selection."""

    method: Literal["interval", "sentence"] = Field(
        default="interval", description="Method for selecting target positions"
    )
    interval: int = Field(
        default=100, ge=1, description="Interval between targets (for interval method)"
    )
    region_length: int = Field(
        default=30, ge=1, description="Number of tokens to aggregate over at each target"
    )
    min_context: int = Field(
        default=64, ge=1, description="Minimum context tokens before first target"
    )


class SpansConfig(BaseModel):
    """Configuration for source span selection."""

    distances: list[int] = Field(
        default=[32, 64, 128, 256, 512, 1024, 2048],
        description="Distance buckets (tokens from span center to target)",
    )
    widths: list[int] = Field(default=[32], description="Span widths to use")
    spans_per_distance: int = Field(
        default=1, ge=1, description="Number of spans to sample per distance bucket"
    )

    @field_validator("distances")
    @classmethod
    def validate_distances(cls, v: list[int]) -> list[int]:
        if not v:
            raise ValueError("distances cannot be empty")
        if any(d <= 0 for d in v):
            raise ValueError("all distances must be positive")
        return sorted(v)

    @field_validator("widths")
    @classmethod
    def validate_widths(cls, v: list[int]) -> list[int]:
        if not v:
            raise ValueError("widths cannot be empty")
        if any(w <= 0 for w in v):
            raise ValueError("all widths must be positive")
        return sorted(v)


class InterventionsConfig(BaseModel):
    """Configuration for intervention types."""

    types: list[Literal["mask", "shuffle", "delete"]] = Field(
        default=["mask", "shuffle"], description="Intervention types to apply"
    )
    mask_token: str | None = Field(
        default=None, description="Token to use for masking (None = model's pad/eos token)"
    )

    @field_validator("types")
    @classmethod
    def validate_types(cls, v: list[str]) -> list[str]:
        if not v:
            raise ValueError("types cannot be empty")
        return v


class ComputationConfig(BaseModel):
    """Configuration for computation settings."""

    batch_size: int = Field(default=8, ge=1, description="Batch size for model inference")
    seed: int = Field(default=42, description="Random seed for reproducibility")
    top_k_for_kl: int = Field(
        default=1000, ge=1, description="Number of top tokens for KL approximation"
    )
    top_k_for_stability: int = Field(
        default=10, ge=1, description="K for top-k stability metric"
    )
    num_workers: int = Field(default=0, ge=0, description="Number of data loading workers")


class OutputConfig(BaseModel):
    """Configuration for output settings."""

    results_dir: Path = Field(default=Path("./results"), description="Directory for output files")
    format: Literal["parquet", "csv", "json"] = Field(
        default="parquet", description="Format for detailed results"
    )
    save_distributions: bool = Field(
        default=False, description="Whether to save full probability distributions"
    )


class LRTIAConfig(BaseModel):
    """Main configuration for LRTIA."""

    model: ModelConfig = Field(default_factory=ModelConfig)
    windowing: WindowingConfig = Field(default_factory=WindowingConfig)
    targets: TargetsConfig = Field(default_factory=TargetsConfig)
    spans: SpansConfig = Field(default_factory=SpansConfig)
    interventions: InterventionsConfig = Field(default_factory=InterventionsConfig)
    computation: ComputationConfig = Field(default_factory=ComputationConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "LRTIAConfig":
        """Load configuration from a YAML file."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Configuration file not found: {path}")
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls.model_validate(data or {})

    @classmethod
    def from_dict(cls, data: dict) -> "LRTIAConfig":
        """Load configuration from a dictionary."""
        return cls.model_validate(data)

    def to_yaml(self, path: str | Path) -> None:
        """Save configuration to a YAML file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            # Use mode="json" to convert Path objects to strings
            yaml.dump(self.model_dump(mode="json"), f, default_flow_style=False, sort_keys=False)

    def to_dict(self) -> dict:
        """Convert configuration to a dictionary."""
        return self.model_dump()
