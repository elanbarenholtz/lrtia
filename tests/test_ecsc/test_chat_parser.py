"""Tests for CHAT parser."""

from pathlib import Path

import pytest

from lrtia.data.ecsc.chat_parser import ECScCHATParser, CHATParseError


class TestFilenameMetadata:
    """Tests for filename metadata parsing."""

    def test_parse_standard_filename(self):
        """Test parsing a standard ECSC filename."""
        parser = ECScCHATParser()
        meta = parser._parse_filename_metadata("1001-42-M-1")

        assert meta["subject_id"] == "1001"
        assert meta["age_months"] == 42
        assert meta["sex"] == "M"
        assert meta["study_year"] == 1

    def test_parse_female_subject(self):
        """Test parsing filename with female subject."""
        parser = ECScCHATParser()
        meta = parser._parse_filename_metadata("2001-54-F-2")

        assert meta["subject_id"] == "2001"
        assert meta["age_months"] == 54
        assert meta["sex"] == "F"
        assert meta["study_year"] == 2

    def test_parse_older_child(self):
        """Test parsing filename with older child (3-digit age)."""
        parser = ECScCHATParser()
        meta = parser._parse_filename_metadata("3001-108-M-3")

        assert meta["subject_id"] == "3001"
        assert meta["age_months"] == 108
        assert meta["sex"] == "M"
        assert meta["study_year"] == 3

    def test_invalid_filename_raises(self):
        """Test that invalid filename raises ValueError."""
        parser = ECScCHATParser()

        with pytest.raises(ValueError):
            parser._parse_filename_metadata("invalid")

        with pytest.raises(ValueError):
            parser._parse_filename_metadata("1001-42-X-1")  # Invalid sex

        with pytest.raises(ValueError):
            parser._parse_filename_metadata("100-42-M-1")  # 3-digit subject ID


class TestSpeakerIdentification:
    """Tests for child speaker identification."""

    def test_identify_target_child_role(self):
        """Test identifying child by Target_Child role."""
        parser = ECScCHATParser()
        participants = {
            "CHI": {"role": "Target_Child", "name": "John"},
            "MOT": {"role": "Mother", "name": "Jane"},
        }
        assert parser._identify_child_speaker(participants) == "CHI"

    def test_identify_child_role(self):
        """Test identifying child by Child role."""
        parser = ECScCHATParser()
        participants = {
            "TAR": {"role": "Child", "name": "Jackie"},
            "MOT": {"role": "Mother", "name": "Jane"},
        }
        assert parser._identify_child_speaker(participants) == "TAR"

    def test_identify_nonstandard_code(self):
        """Test identifying child with non-CHI speaker code."""
        parser = ECScCHATParser()
        participants = {
            "JAC": {"role": "Target_Child", "name": "Jackie"},
            "MOM": {"role": "Mother", "name": "Mary"},
        }
        assert parser._identify_child_speaker(participants) == "JAC"

    def test_case_insensitive_role_match(self):
        """Test case-insensitive role matching."""
        parser = ECScCHATParser()
        participants = {
            "CHI": {"role": "target_child", "name": "John"},
            "MOT": {"role": "mother", "name": "Jane"},
        }
        assert parser._identify_child_speaker(participants) == "CHI"

    def test_fallback_to_common_codes(self):
        """Test fallback to common child speaker codes."""
        parser = ECScCHATParser()
        participants = {
            "CHI": {"role": "", "name": "Unknown"},
            "MOT": {"role": "Mother", "name": "Jane"},
        }
        assert parser._identify_child_speaker(participants) == "CHI"

    def test_no_child_returns_none(self):
        """Test that no child speaker returns None."""
        parser = ECScCHATParser()
        participants = {
            "INV": {"role": "Investigator", "name": "Dr. Smith"},
            "MOT": {"role": "Mother", "name": "Jane"},
        }
        assert parser._identify_child_speaker(participants) is None


class TestCHATFileParsing:
    """Tests for parsing actual CHAT files."""

    def test_parse_sample_file(self, sample_chat_file: Path):
        """Test parsing a sample CHAT file."""
        parser = ECScCHATParser()
        transcript = parser.parse_file(sample_chat_file)

        assert transcript.subject_id == "1001"
        assert transcript.age_months == 50
        assert transcript.sex == "M"
        assert transcript.study_year == 1
        assert transcript.child_speaker_code == "CHI"

    def test_parse_child_utterances(self, sample_chat_file: Path):
        """Test that child utterances are extracted correctly."""
        parser = ECScCHATParser()
        transcript = parser.parse_file(sample_chat_file)

        child_utts = transcript.child_utterances
        assert len(child_utts) >= 4  # At least 4 child utterances

        # Check that text is cleaned
        for utt in child_utts:
            assert utt.speaker == "CHI"
            assert utt.clean_text  # Should have some cleaned text

    def test_parse_nonstandard_speaker(self, sample_chat_nonstandard_file: Path):
        """Test parsing file with non-standard speaker code."""
        parser = ECScCHATParser()
        transcript = parser.parse_file(sample_chat_nonstandard_file)

        # Should identify JAC as the child speaker, not hardcode to CHI
        assert transcript.child_speaker_code == "JAC"
        assert len(transcript.child_utterances) >= 3

    def test_child_text_extraction(self, sample_chat_file: Path):
        """Test extraction of concatenated child text."""
        parser = ECScCHATParser()
        transcript = parser.parse_file(sample_chat_file)

        clean_text = transcript.clean_child_text
        assert "frog" in clean_text
        assert "boy" in clean_text
        assert "jar" in clean_text

    def test_markup_cleaned_from_text(self, sample_chat_file: Path):
        """Test that CHAT markup is cleaned from text."""
        parser = ECScCHATParser()
        transcript = parser.parse_file(sample_chat_file)

        clean_text = transcript.clean_child_text

        # Markup should be removed
        assert "[" not in clean_text
        assert "]" not in clean_text
        assert "&=" not in clean_text
        assert "xxx" not in clean_text

    def test_file_not_found_raises(self, tmp_path: Path):
        """Test that non-existent file raises error."""
        parser = ECScCHATParser()

        with pytest.raises(CHATParseError) as exc_info:
            parser.parse_file(tmp_path / "nonexistent.cha")

        assert "does not exist" in str(exc_info.value)

    def test_wrong_extension_raises(self, tmp_path: Path):
        """Test that wrong file extension raises error."""
        parser = ECScCHATParser()
        txt_file = tmp_path / "test.txt"
        txt_file.write_text("not a chat file")

        with pytest.raises(CHATParseError) as exc_info:
            parser.parse_file(txt_file)

        assert ".cha" in str(exc_info.value)


class TestDirectoryParsing:
    """Tests for parsing directories of CHAT files."""

    def test_parse_directory(
        self, tmp_path: Path, sample_chat_content: str, sample_chat_nonstandard_speaker: str
    ):
        """Test parsing a directory of CHAT files."""
        # Create multiple files
        (tmp_path / "1001-50-M-1.cha").write_text(sample_chat_content)
        (tmp_path / "2001-54-F-1.cha").write_text(sample_chat_nonstandard_speaker)

        parser = ECScCHATParser()
        results = list(parser.parse_directory(tmp_path, skip_errors=True))

        assert len(results) == 2

        # All should be successful
        for path, result in results:
            assert not isinstance(result, CHATParseError)

    def test_parse_directory_with_errors(self, tmp_path: Path, sample_chat_content: str):
        """Test parsing directory with some invalid files."""
        # Create valid file
        (tmp_path / "1001-50-M-1.cha").write_text(sample_chat_content)
        # Create invalid file (bad filename)
        (tmp_path / "invalid.cha").write_text(sample_chat_content)

        parser = ECScCHATParser()
        results = list(parser.parse_directory(tmp_path, skip_errors=True))

        assert len(results) == 2

        # One success, one error
        errors = [r for _, r in results if isinstance(r, CHATParseError)]
        successes = [r for _, r in results if not isinstance(r, CHATParseError)]

        assert len(errors) == 1
        assert len(successes) == 1
