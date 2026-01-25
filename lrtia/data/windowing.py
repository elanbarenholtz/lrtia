"""Text windowing for evaluation."""

from dataclasses import dataclass, field

from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast

from lrtia.data.ingestion import Document
from lrtia.data.normalization import normalize_for_tokenization


@dataclass
class Window:
    """A tokenized window of text for evaluation."""

    doc_id: str
    author_id: str
    population: str
    window_id: int
    token_ids: list[int]
    token_offset: int  # Offset of first token in original document
    text_span: tuple[int, int] = field(default=(0, 0))  # Character span in original text

    @property
    def num_tokens(self) -> int:
        """Number of tokens in this window."""
        return len(self.token_ids)


def create_windows(
    document: Document,
    tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast,
    window_tokens: int = 2048,
    stride_tokens: int = 512,
    normalize: bool = True,
) -> list[Window]:
    """Create evaluation windows from a document.

    Args:
        document: The document to window.
        tokenizer: Tokenizer to use.
        window_tokens: Maximum tokens per window.
        stride_tokens: Stride between windows.
        normalize: Whether to normalize text before tokenization.

    Returns:
        List of Window objects.
    """
    text = document.text
    if normalize:
        text = normalize_for_tokenization(text)

    # Tokenize the full document
    encoding = tokenizer(
        text,
        add_special_tokens=False,
        return_offsets_mapping=True,
        truncation=False,
    )

    token_ids = encoding["input_ids"]
    offset_mapping = encoding.get("offset_mapping", [])

    if not token_ids:
        return []

    windows = []
    window_id = 0
    start = 0

    while start < len(token_ids):
        end = min(start + window_tokens, len(token_ids))
        window_token_ids = token_ids[start:end]

        # Get character span if offset mapping is available
        if offset_mapping:
            char_start = offset_mapping[start][0] if start < len(offset_mapping) else 0
            char_end = offset_mapping[end - 1][1] if end - 1 < len(offset_mapping) else len(text)
            text_span = (char_start, char_end)
        else:
            text_span = (0, 0)

        window = Window(
            doc_id=document.doc_id,
            author_id=document.author_id,
            population=document.population,
            window_id=window_id,
            token_ids=window_token_ids,
            token_offset=start,
            text_span=text_span,
        )
        windows.append(window)

        # Move to next window
        start += stride_tokens
        window_id += 1

        # If we've reached the end, stop
        if end >= len(token_ids):
            break

    return windows


def create_windows_batch(
    documents: list[Document],
    tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast,
    window_tokens: int = 2048,
    stride_tokens: int = 512,
    normalize: bool = True,
) -> list[Window]:
    """Create evaluation windows from multiple documents.

    Args:
        documents: List of documents to window.
        tokenizer: Tokenizer to use.
        window_tokens: Maximum tokens per window.
        stride_tokens: Stride between windows.
        normalize: Whether to normalize text before tokenization.

    Returns:
        List of Window objects from all documents.
    """
    all_windows = []
    for doc in documents:
        windows = create_windows(
            doc,
            tokenizer,
            window_tokens=window_tokens,
            stride_tokens=stride_tokens,
            normalize=normalize,
        )
        all_windows.extend(windows)
    return all_windows


def get_target_positions(
    window: Window,
    method: str = "interval",
    interval: int = 100,
    min_context: int = 64,
    region_length: int = 30,
) -> list[tuple[int, int]]:
    """Get target positions within a window.

    Args:
        window: The window to get targets from.
        method: Selection method ("interval" or "sentence").
        interval: Interval between targets (for interval method).
        min_context: Minimum context tokens before first target.
        region_length: Length of the evaluation region at each target.

    Returns:
        List of (start, end) tuples for each target region.
    """
    num_tokens = window.num_tokens
    targets = []

    if method == "interval":
        # Start after min_context, place targets at regular intervals
        pos = min_context
        while pos + region_length <= num_tokens:
            targets.append((pos, pos + region_length))
            pos += interval
    elif method == "sentence":
        # Sentence boundary detection would require additional processing
        # For now, fall back to interval method
        # TODO: Implement sentence boundary detection
        pos = min_context
        while pos + region_length <= num_tokens:
            targets.append((pos, pos + region_length))
            pos += interval
    else:
        raise ValueError(f"Unknown target selection method: {method}")

    return targets
