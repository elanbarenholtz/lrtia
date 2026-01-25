"""Tests for aggregation module."""

import numpy as np
import pandas as pd
import pytest

from lrtia.aggregation.curves import (
    CurvePoint,
    MemoryCurve,
    build_memory_curve,
    build_curves_by_group,
)
from lrtia.aggregation.summary import (
    compute_auc,
    compute_half_life,
    compute_tail_mass,
    compute_summary_metrics,
)


class TestCurvePoint:
    def test_basic(self):
        point = CurvePoint(
            distance=32,
            mean=0.5,
            std=0.1,
            ci_lower=0.4,
            ci_upper=0.6,
            count=10,
            values=[0.4, 0.5, 0.6],
        )
        assert point.distance == 32
        assert point.mean == 0.5
        assert point.count == 10


class TestMemoryCurve:
    def test_basic_properties(self):
        points = [
            CurvePoint(32, 0.5, 0.1, 0.4, 0.6, 10),
            CurvePoint(64, 0.3, 0.1, 0.2, 0.4, 10),
            CurvePoint(128, 0.1, 0.05, 0.05, 0.15, 10),
        ]
        curve = MemoryCurve(
            metric="delta_nll",
            group_by="population",
            group_id="test",
            points=points,
        )

        assert curve.distances == [32, 64, 128]
        assert curve.means == [0.5, 0.3, 0.1]
        assert curve.stds == [0.1, 0.1, 0.05]

    def test_to_dataframe(self):
        points = [
            CurvePoint(32, 0.5, 0.1, 0.4, 0.6, 10),
            CurvePoint(64, 0.3, 0.1, 0.2, 0.4, 10),
        ]
        curve = MemoryCurve("delta_nll", "population", "test", points)

        df = curve.to_dataframe()
        assert len(df) == 2
        assert "distance" in df.columns
        assert "mean" in df.columns
        assert df["metric"].iloc[0] == "delta_nll"

    def test_to_dict(self):
        points = [CurvePoint(32, 0.5, 0.1, 0.4, 0.6, 10)]
        curve = MemoryCurve("delta_nll", "population", "test", points)

        d = curve.to_dict()
        assert d["metric"] == "delta_nll"
        assert d["distances"] == [32]
        assert d["means"] == [0.5]


class TestBuildMemoryCurve:
    def test_basic(self):
        # Create sample data
        data = {
            "distance": [32, 32, 32, 64, 64, 64, 128, 128, 128],
            "delta_nll": [0.5, 0.6, 0.4, 0.3, 0.35, 0.25, 0.1, 0.15, 0.05],
            "population": ["a"] * 9,
        }
        df = pd.DataFrame(data)

        curve = build_memory_curve(df, metric="delta_nll")

        assert len(curve.points) == 3
        assert curve.distances == [32, 64, 128]

        # Check means are correct
        assert curve.points[0].mean == pytest.approx(0.5, abs=0.01)
        assert curve.points[1].mean == pytest.approx(0.3, abs=0.01)
        assert curve.points[2].mean == pytest.approx(0.1, abs=0.01)

    def test_with_group(self):
        data = {
            "distance": [32, 32, 64, 64],
            "delta_nll": [0.5, 0.6, 0.3, 0.4],
            "population": ["a", "a", "a", "a"],
        }
        df = pd.DataFrame(data)

        curve = build_memory_curve(
            df, metric="delta_nll", group_by="population", group_id="a"
        )

        assert curve.group_by == "population"
        assert curve.group_id == "a"


class TestBuildCurvesByGroup:
    def test_multiple_groups(self):
        data = {
            "distance": [32, 32, 64, 64, 32, 32, 64, 64],
            "delta_nll": [0.5, 0.6, 0.3, 0.4, 0.8, 0.9, 0.6, 0.7],
            "population": ["a", "a", "a", "a", "b", "b", "b", "b"],
        }
        df = pd.DataFrame(data)

        curves = build_curves_by_group(df, metric="delta_nll", group_by="population")

        assert "a" in curves
        assert "b" in curves
        assert curves["b"].points[0].mean > curves["a"].points[0].mean


class TestComputeAUC:
    def test_basic(self):
        points = [
            CurvePoint(0, 1.0, 0.1, 0.9, 1.1, 10),
            CurvePoint(100, 0.0, 0.1, -0.1, 0.1, 10),
        ]
        curve = MemoryCurve("delta_nll", "all", "all", points)

        auc = compute_auc(curve)
        # Triangle with base 100 and height 1.0 = 50
        assert auc == pytest.approx(50.0)

    def test_normalized(self):
        points = [
            CurvePoint(0, 1.0, 0.1, 0.9, 1.1, 10),
            CurvePoint(100, 0.0, 0.1, -0.1, 0.1, 10),
        ]
        curve = MemoryCurve("delta_nll", "all", "all", points)

        auc_norm = compute_auc(curve, normalize=True)
        # Normalized by range (100), so 50/100 = 0.5
        assert auc_norm == pytest.approx(0.5)


class TestComputeHalfLife:
    def test_basic(self):
        # Linear decay from 1.0 to 0.0 over distance 0-100
        points = [
            CurvePoint(0, 1.0, 0.1, 0.9, 1.1, 10),
            CurvePoint(50, 0.5, 0.1, 0.4, 0.6, 10),
            CurvePoint(100, 0.0, 0.1, -0.1, 0.1, 10),
        ]
        curve = MemoryCurve("delta_nll", "all", "all", points)

        half_life = compute_half_life(curve)
        assert half_life == pytest.approx(50.0)

    def test_no_decay(self):
        # Constant effect
        points = [
            CurvePoint(0, 1.0, 0.1, 0.9, 1.1, 10),
            CurvePoint(100, 1.0, 0.1, 0.9, 1.1, 10),
        ]
        curve = MemoryCurve("delta_nll", "all", "all", points)

        half_life = compute_half_life(curve)
        # Effect never drops to 50%, so returns None
        assert half_life is None


class TestComputeTailMass:
    def test_basic(self):
        # Effect concentrated at short distances
        points = [
            CurvePoint(0, 1.0, 0.1, 0.9, 1.1, 10),
            CurvePoint(50, 0.5, 0.1, 0.4, 0.6, 10),
            CurvePoint(100, 0.0, 0.1, -0.1, 0.1, 10),
        ]
        curve = MemoryCurve("delta_nll", "all", "all", points)

        tail_mass = compute_tail_mass(curve, threshold_distance=50)
        # Tail is from 50-100, which has lower values
        assert 0.0 <= tail_mass <= 1.0


class TestComputeSummaryMetrics:
    def test_all_metrics(self):
        points = [
            CurvePoint(32, 0.5, 0.1, 0.4, 0.6, 10, [0.4, 0.5, 0.6]),
            CurvePoint(64, 0.3, 0.1, 0.2, 0.4, 10, [0.25, 0.3, 0.35]),
            CurvePoint(128, 0.1, 0.05, 0.05, 0.15, 10, [0.08, 0.1, 0.12]),
        ]
        curve = MemoryCurve("delta_nll", "all", "all", points)

        summary = compute_summary_metrics(curve)

        assert summary.auc > 0
        assert summary.auc_normalized > 0
        assert summary.peak_effect == 0.5
        assert summary.peak_distance == 32

    def test_empty_curve(self):
        curve = MemoryCurve("delta_nll", "all", "all", [])
        summary = compute_summary_metrics(curve)

        assert summary.auc == 0.0
        assert summary.peak_effect == 0.0
