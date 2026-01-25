"""Text normalization utilities."""

import re
import unicodedata


def normalize_text(
    text: str,
    *,
    normalize_unicode: bool = True,
    normalize_whitespace: bool = True,
    strip_control_chars: bool = True,
    lowercase: bool = False,
) -> str:
    """Normalize text for consistent processing.

    Args:
        text: Input text to normalize.
        normalize_unicode: If True, apply NFC unicode normalization.
        normalize_whitespace: If True, normalize whitespace (collapse multiple
            spaces, normalize line endings).
        strip_control_chars: If True, remove control characters except newlines
            and tabs.
        lowercase: If True, convert to lowercase.

    Returns:
        Normalized text.
    """
    if not text:
        return ""

    # Unicode normalization (NFC form)
    if normalize_unicode:
        text = unicodedata.normalize("NFC", text)

    # Strip control characters (except newlines and tabs)
    if strip_control_chars:
        text = _strip_control_chars(text)

    # Normalize whitespace
    if normalize_whitespace:
        text = _normalize_whitespace(text)

    # Lowercase
    if lowercase:
        text = text.lower()

    return text


def _strip_control_chars(text: str) -> str:
    """Remove control characters except newlines and tabs."""
    # Keep: \t (0x09), \n (0x0A), \r (0x0D)
    # Remove other control chars (0x00-0x08, 0x0B-0x0C, 0x0E-0x1F, 0x7F)
    return "".join(
        char
        for char in text
        if char in "\t\n\r" or (ord(char) >= 0x20 and ord(char) != 0x7F)
    )


def _normalize_whitespace(text: str) -> str:
    """Normalize whitespace in text."""
    # Normalize line endings to \n
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # Replace tabs with spaces (optional, configurable)
    # text = text.replace("\t", " ")

    # Collapse multiple spaces into one (but preserve newlines)
    text = re.sub(r"[^\S\n]+", " ", text)

    # Remove trailing whitespace from lines
    text = re.sub(r" +\n", "\n", text)

    # Remove leading whitespace from lines
    text = re.sub(r"\n +", "\n", text)

    # Collapse multiple newlines into at most two
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Strip leading/trailing whitespace
    text = text.strip()

    return text


def normalize_for_tokenization(text: str) -> str:
    """Normalize text specifically for tokenization.

    This is a lighter normalization that preserves more structure
    while ensuring consistent tokenization.

    Args:
        text: Input text.

    Returns:
        Normalized text suitable for tokenization.
    """
    if not text:
        return ""

    # Unicode normalization
    text = unicodedata.normalize("NFC", text)

    # Remove null bytes and other problematic characters
    text = text.replace("\x00", "")

    # Normalize line endings
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    return text
