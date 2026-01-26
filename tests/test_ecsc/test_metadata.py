"""Tests for ECSC metadata handling."""

from pathlib import Path

import pytest

from lrtia.data.ecsc.metadata import ECScMetadataLoader
from lrtia.data.ecsc.models import ECScSubjectMetadata


class TestECScSubjectMetadata:
    """Tests for the ECScSubjectMetadata model."""

    def test_basic_creation(self):
        """Test creating a basic metadata object."""
        meta = ECScSubjectMetadata(
            subject_id="1001",
            age_months=42,
            sex="M",
            study_year=1,
            recruitment_year=1,
        )

        assert meta.subject_id == "1001"
        assert meta.age_months == 42
        assert meta.sex == "M"
        assert meta.study_year == 1
        assert meta.recruitment_year == 1
        assert meta.dataset == "ECSC"

    def test_with_ppvt_scores(self):
        """Test metadata with PPVT scores."""
        meta = ECScSubjectMetadata(
            subject_id="1001",
            age_months=42,
            sex="M",
            study_year=1,
            recruitment_year=1,
            ppvt_raw=85.0,
            ppvt_standard=105.0,
            ppvt_percentile=63.0,
        )

        assert meta.ppvt_raw == 85.0
        assert meta.ppvt_standard == 105.0
        assert meta.ppvt_percentile == 63.0

    def test_to_population_string(self):
        """Test serialization to JSON string."""
        meta = ECScSubjectMetadata(
            subject_id="1001",
            age_months=42,
            sex="M",
            study_year=1,
            recruitment_year=1,
            ppvt_standard=105.0,
        )

        pop_str = meta.to_population_string()

        assert isinstance(pop_str, str)
        assert "ECSC" in pop_str
        assert "1001" in pop_str
        assert "105" in pop_str

    def test_roundtrip_serialization(self):
        """Test roundtrip serialization to/from population string."""
        original = ECScSubjectMetadata(
            subject_id="2001",
            age_months=54,
            sex="F",
            study_year=2,
            recruitment_year=2,
            ppvt_standard=98.0,
            ethnicity="Hispanic",
        )

        pop_str = original.to_population_string()
        recovered = ECScSubjectMetadata.from_population_string(pop_str)

        assert recovered.subject_id == original.subject_id
        assert recovered.age_months == original.age_months
        assert recovered.sex == original.sex
        assert recovered.ppvt_standard == original.ppvt_standard
        assert recovered.ethnicity == original.ethnicity

    def test_none_values_excluded(self):
        """Test that None values are excluded from serialization."""
        meta = ECScSubjectMetadata(
            subject_id="1001",
            age_months=42,
            sex="M",
            study_year=1,
            recruitment_year=1,
            ppvt_raw=None,  # Explicit None
        )

        pop_str = meta.to_population_string()

        assert "ppvt_raw" not in pop_str

    def test_other_meta_dict(self):
        """Test that other_meta stores extra fields."""
        meta = ECScSubjectMetadata(
            subject_id="1001",
            age_months=42,
            sex="M",
            study_year=1,
            recruitment_year=1,
            other_meta={"custom_field": "value", "another": "data"},
        )

        assert meta.other_meta["custom_field"] == "value"
        assert meta.other_meta["another"] == "data"


class TestECScMetadataLoader:
    """Tests for the ECScMetadataLoader class."""

    def test_load_csv(self, sample_metadata_csv: Path):
        """Test loading metadata from CSV."""
        loader = ECScMetadataLoader(sample_metadata_csv)
        loader.load()

        # Should have loaded data
        assert loader._metadata_df is not None
        assert len(loader._metadata_df) == 3

    def test_get_subject_with_table_data(self, sample_metadata_csv: Path):
        """Test getting subject metadata with table lookup."""
        loader = ECScMetadataLoader(sample_metadata_csv)
        loader.load()

        meta = loader.get_subject_metadata(
            subject_id="1001",
            age_months=50,
            sex="M",
            study_year=1,
        )

        # From filename
        assert meta.subject_id == "1001"
        assert meta.age_months == 50
        assert meta.sex == "M"
        assert meta.study_year == 1

        # From table
        assert meta.ppvt_standard == 105.0
        assert meta.ppvt_raw == 85.0
        assert meta.ethnicity == "White"

    def test_get_subject_without_table(self):
        """Test getting subject metadata without loading table."""
        loader = ECScMetadataLoader()

        meta = loader.get_subject_metadata(
            subject_id="1001",
            age_months=50,
            sex="M",
            study_year=1,
        )

        # Only file-derived values
        assert meta.subject_id == "1001"
        assert meta.age_months == 50
        assert meta.ppvt_standard is None  # Not available

    def test_recruitment_year_derived(self, sample_metadata_csv: Path):
        """Test that recruitment year is derived from subject_id."""
        loader = ECScMetadataLoader(sample_metadata_csv)
        loader.load()

        meta = loader.get_subject_metadata(
            subject_id="2001",
            age_months=54,
            sex="F",
            study_year=1,
        )

        # First digit of 2001 is 2
        assert meta.recruitment_year == 2

    def test_join_success_rate(self, sample_metadata_csv: Path):
        """Test join success rate tracking."""
        loader = ECScMetadataLoader(sample_metadata_csv)
        loader.load()

        # Query existing subjects
        loader.get_subject_metadata("1001", 50, "M", 1)
        loader.get_subject_metadata("2001", 54, "F", 1)

        # Query non-existing subject
        loader.get_subject_metadata("9999", 60, "M", 1)

        # 2 out of 3 should be found
        assert loader.join_success_rate == pytest.approx(2 / 3)

    def test_missing_subjects_tracked(self, sample_metadata_csv: Path):
        """Test that missing subjects are tracked."""
        loader = ECScMetadataLoader(sample_metadata_csv)
        loader.load()

        loader.get_subject_metadata("1001", 50, "M", 1)
        loader.get_subject_metadata("9999", 60, "M", 1)

        missing = loader.subjects_missing_metadata
        assert "9999" in missing
        assert "1001" not in missing

    def test_get_stats(self, sample_metadata_csv: Path):
        """Test getting loader statistics."""
        loader = ECScMetadataLoader(sample_metadata_csv)
        loader.load()

        loader.get_subject_metadata("1001", 50, "M", 1)
        loader.get_subject_metadata("9999", 60, "M", 1)

        stats = loader.get_stats()

        assert stats["metadata_loaded"] is True
        assert stats["total_subjects_in_table"] == 3
        assert stats["subjects_queried"] == 2
        assert stats["subjects_found"] == 1

    def test_file_not_found_raises(self, tmp_path: Path):
        """Test that missing file raises FileNotFoundError."""
        loader = ECScMetadataLoader(tmp_path / "nonexistent.csv")

        with pytest.raises(FileNotFoundError):
            loader.load()

    def test_unsupported_format_raises(self, tmp_path: Path):
        """Test that unsupported format raises ValueError."""
        bad_file = tmp_path / "data.json"
        bad_file.write_text("{}")

        loader = ECScMetadataLoader(bad_file)

        with pytest.raises(ValueError):
            loader.load()
