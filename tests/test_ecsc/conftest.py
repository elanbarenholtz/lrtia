"""Pytest fixtures for ECSC tests."""

from pathlib import Path

import pytest


@pytest.fixture
def sample_chat_content() -> str:
    """Return sample CHAT file content for testing."""
    return """@UTF8
@Begin
@Languages:\teng
@Participants:\tCHI Jackie Target_Child, MOT Mother Mother
@ID:\teng|ECSC|CHI|4;2|male|||Target_Child|||
@ID:\teng|ECSC|MOT|35;0|female|||Mother|||
@Media:\t1001-50-M-1, audio
@Comment:\tThis is a sample frog story narration.
*CHI:\tthe frog was in the jar .
%mor:\tdet|the n|frog cop|be&PAST prep|in det|the n|jar .
*MOT:\tyes it was .
*CHI:\tthen the frog [= pet] jumped out &=laughs .
%mor:\tadv|then det|the n|frog v|jump-PAST adv|out .
*CHI:\tthe boy was sleeping .
*MOT:\twhat happened next ?
*CHI:\tthe dog looked in the jar xxx .
*CHI:\tuh the frog escaped .
@End
"""


@pytest.fixture
def sample_chat_file(tmp_path: Path, sample_chat_content: str) -> Path:
    """Create a sample CHAT file for testing."""
    file_path = tmp_path / "1001-50-M-1.cha"
    file_path.write_text(sample_chat_content, encoding="utf-8")
    return file_path


@pytest.fixture
def sample_chat_nonstandard_speaker() -> str:
    """Return sample CHAT content with non-standard speaker code."""
    return """@UTF8
@Begin
@Languages:\teng
@Participants:\tJAC Jackie Target_Child, MOM Mother Mother
@ID:\teng|ECSC|JAC|4;6|female|||Target_Child|||
@ID:\teng|ECSC|MOM|32;0|female|||Mother|||
*JAC:\tonce there was a boy .
*MOM:\tvery good .
*JAC:\the had a frog and a dog .
*JAC:\tthe frog went away .
@End
"""


@pytest.fixture
def sample_chat_nonstandard_file(
    tmp_path: Path, sample_chat_nonstandard_speaker: str
) -> Path:
    """Create a CHAT file with non-standard speaker code."""
    file_path = tmp_path / "2001-54-F-1.cha"
    file_path.write_text(sample_chat_nonstandard_speaker, encoding="utf-8")
    return file_path


@pytest.fixture
def sample_metadata_csv(tmp_path: Path) -> Path:
    """Create a sample metadata CSV file."""
    content = """subject_id,ppvt_standard,ppvt_raw,ethnicity,maternal_education
1001,105,85,White,College
2001,98,72,Hispanic,High School
3001,112,91,Asian,Graduate
"""
    file_path = tmp_path / "metadata.csv"
    file_path.write_text(content, encoding="utf-8")
    return file_path
