"""Tests for configuration module."""

import pytest
import tempfile
from pathlib import Path

from lrtia.config import (
    LRTIAConfig,
    ModelConfig,
    WindowingConfig,
    SpansConfig,
    InterventionsConfig,
)


class TestModelConfig:
    def test_defaults(self):
        config = ModelConfig()
        assert config.name == "gpt2"
        assert config.backend == "huggingface"
        assert config.device == "cuda"

    def test_custom_values(self):
        config = ModelConfig(name="meta-llama/Llama-2-7b", device="cpu")
        assert config.name == "meta-llama/Llama-2-7b"
        assert config.device == "cpu"


class TestWindowingConfig:
    def test_defaults(self):
        config = WindowingConfig()
        assert config.window_tokens == 2048
        assert config.stride_tokens == 512

    def test_stride_validation(self):
        # Stride can be <= window
        config = WindowingConfig(window_tokens=1024, stride_tokens=512)
        assert config.stride_tokens == 512

        # Stride > window should fail
        with pytest.raises(ValueError):
            WindowingConfig(window_tokens=512, stride_tokens=1024)


class TestSpansConfig:
    def test_defaults(self):
        config = SpansConfig()
        assert config.distances == [32, 64, 128, 256, 512, 1024, 2048]
        assert config.widths == [32]
        assert config.spans_per_distance == 1

    def test_distances_sorted(self):
        config = SpansConfig(distances=[256, 32, 128])
        assert config.distances == [32, 128, 256]

    def test_empty_distances_fails(self):
        with pytest.raises(ValueError):
            SpansConfig(distances=[])

    def test_negative_distances_fails(self):
        with pytest.raises(ValueError):
            SpansConfig(distances=[32, -1, 64])


class TestInterventionsConfig:
    def test_defaults(self):
        config = InterventionsConfig()
        assert config.types == ["mask", "shuffle"]
        assert config.mask_token is None

    def test_custom_types(self):
        config = InterventionsConfig(types=["mask", "delete"])
        assert "mask" in config.types
        assert "delete" in config.types

    def test_empty_types_fails(self):
        with pytest.raises(ValueError):
            InterventionsConfig(types=[])


class TestLRTIAConfig:
    def test_defaults(self):
        config = LRTIAConfig()
        assert config.model.name == "gpt2"
        assert config.windowing.window_tokens == 2048
        assert config.spans.distances[0] == 32

    def test_from_yaml(self):
        yaml_content = """
model:
  name: "gpt2-medium"
  device: "cpu"

windowing:
  window_tokens: 1024
  stride_tokens: 256
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            f.flush()

            config = LRTIAConfig.from_yaml(f.name)
            assert config.model.name == "gpt2-medium"
            assert config.model.device == "cpu"
            assert config.windowing.window_tokens == 1024

    def test_to_yaml(self):
        config = LRTIAConfig()
        config.model.name = "test-model"

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "config.yaml"
            config.to_yaml(path)

            loaded = LRTIAConfig.from_yaml(path)
            assert loaded.model.name == "test-model"

    def test_to_dict(self):
        config = LRTIAConfig()
        d = config.to_dict()

        assert isinstance(d, dict)
        assert "model" in d
        assert d["model"]["name"] == "gpt2"
