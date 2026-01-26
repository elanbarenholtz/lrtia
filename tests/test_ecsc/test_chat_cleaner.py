"""Tests for CHAT text cleaner."""

import pytest

from lrtia.data.ecsc.chat_cleaner import CHATCleaner, FILLER_WORDS


class TestCHATCleaner:
    """Tests for the CHATCleaner class."""

    def test_default_initialization(self):
        """Test that cleaner initializes with sensible defaults."""
        cleaner = CHATCleaner()
        assert cleaner.remove_brackets is True
        assert cleaner.remove_angles is True
        assert cleaner.remove_ampersand_codes is True
        assert cleaner.remove_unintelligible is True
        assert cleaner.preserve_fillers is True
        assert cleaner.collapse_whitespace is True

    def test_remove_brackets(self):
        """Test removal of bracketed content."""
        cleaner = CHATCleaner()
        assert cleaner.clean("hello [laugh] world") == "hello world"
        assert cleaner.clean("the frog [= pet] jumped") == "the frog jumped"
        assert cleaner.clean("go [/] go away") == "go go away"

    def test_preserve_brackets_when_disabled(self):
        """Test that brackets are preserved when disabled."""
        cleaner = CHATCleaner(remove_brackets=False)
        assert cleaner.clean("hello [laugh] world") == "hello [laugh] world"

    def test_remove_angles(self):
        """Test removal of angle-bracketed content."""
        cleaner = CHATCleaner()
        assert cleaner.clean("hello <overlap> world") == "hello world"
        assert cleaner.clean("<before> text <after>") == "text"

    def test_preserve_angles_when_disabled(self):
        """Test that angles are preserved when disabled."""
        cleaner = CHATCleaner(remove_angles=False)
        assert cleaner.clean("hello <overlap> world") == "hello <overlap> world"

    def test_remove_ampersand_eq_codes(self):
        """Test removal of &=codes."""
        cleaner = CHATCleaner()
        assert cleaner.clean("hello &=laughs world") == "hello world"
        assert cleaner.clean("&=coughs the boy") == "the boy"
        assert cleaner.clean("sleep &=snores &=breathes deeply") == "sleep deeply"

    def test_remove_unintelligible_xxx(self):
        """Test removal of xxx markers."""
        cleaner = CHATCleaner()
        assert cleaner.clean("hello xxx world") == "hello world"
        assert cleaner.clean("xxx the boy") == "the boy"

    def test_remove_unintelligible_yyy(self):
        """Test removal of yyy markers."""
        cleaner = CHATCleaner()
        assert cleaner.clean("hello yyy world") == "hello world"

    def test_remove_unintelligible_www(self):
        """Test removal of www markers."""
        cleaner = CHATCleaner()
        assert cleaner.clean("hello www world") == "hello world"

    def test_preserve_unintelligible_when_disabled(self):
        """Test that xxx/yyy/www are preserved when disabled."""
        cleaner = CHATCleaner(remove_unintelligible=False)
        assert cleaner.clean("hello xxx world") == "hello xxx world"

    def test_remove_pauses(self):
        """Test removal of pause markers."""
        cleaner = CHATCleaner()
        assert cleaner.clean("hello (.) world") == "hello world"
        assert cleaner.clean("wait (..) here") == "wait here"
        assert cleaner.clean("long (...) pause") == "long pause"

    def test_remove_special_chars(self):
        """Test removal of CHAT special characters."""
        cleaner = CHATCleaner()
        assert cleaner.clean("hello # world") == "hello world"
        assert cleaner.clean("the @ boy") == "the boy"

    def test_preserve_punctuation(self):
        """Test that sentence punctuation is preserved."""
        cleaner = CHATCleaner()
        assert cleaner.clean("hello. goodbye!") == "hello. goodbye!"
        assert cleaner.clean("what? yes, okay.") == "what? yes, okay."

    def test_collapse_whitespace(self):
        """Test collapsing of multiple spaces."""
        cleaner = CHATCleaner()
        assert cleaner.clean("hello    world") == "hello world"
        assert cleaner.clean("  leading") == "leading"
        assert cleaner.clean("trailing  ") == "trailing"

    def test_preserve_whitespace_when_disabled(self):
        """Test that whitespace is preserved when disabled."""
        cleaner = CHATCleaner(collapse_whitespace=False)
        result = cleaner.clean("hello    world")
        assert "    " in result

    def test_preserve_fillers(self):
        """Test that filler words are preserved."""
        cleaner = CHATCleaner(preserve_fillers=True)
        # &um should become just "um"
        assert cleaner.clean("&um the boy") == "um the boy"
        assert cleaner.clean("he &uh went") == "he uh went"

    def test_remove_fillers_when_disabled(self):
        """Test that fillers are removed when disabled."""
        cleaner = CHATCleaner(preserve_fillers=False)
        assert cleaner.clean("&um the boy") == "the boy"

    def test_complex_example(self):
        """Test cleaning of a complex CHAT utterance."""
        cleaner = CHATCleaner()
        raw = "the frog [= pet] jumped &=laughs out xxx ."
        expected = "the frog jumped out ."
        assert cleaner.clean(raw) == expected

    def test_clean_with_stats(self):
        """Test cleaning with statistics."""
        cleaner = CHATCleaner()
        text = "hello [comment] &=laughs xxx ."
        cleaned, stats = cleaner.clean_with_stats(text)

        assert cleaned == "hello ."
        assert stats.brackets_removed >= 1
        assert stats.ampersand_codes_removed >= 1
        assert stats.unintelligible_removed >= 1

    def test_empty_string(self):
        """Test cleaning empty string."""
        cleaner = CHATCleaner()
        assert cleaner.clean("") == ""

    def test_no_markup(self):
        """Test cleaning text with no CHAT markup."""
        cleaner = CHATCleaner()
        text = "the boy found his frog."
        assert cleaner.clean(text) == text


class TestFillerWords:
    """Tests for filler word handling."""

    def test_common_fillers_in_set(self):
        """Test that common fillers are in the FILLER_WORDS set."""
        assert "um" in FILLER_WORDS
        assert "uh" in FILLER_WORDS
        assert "like" in FILLER_WORDS
        assert "well" in FILLER_WORDS
