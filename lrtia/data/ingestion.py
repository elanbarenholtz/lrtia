"""Data ingestion for loading corpora from various formats."""

import json
from pathlib import Path
from typing import Iterator

import pandas as pd
from pydantic import BaseModel, Field


class Document(BaseModel):
    """A single document in the corpus."""

    doc_id: str = Field(description="Unique document identifier")
    author_id: str = Field(default="unknown", description="Author identifier")
    population: str = Field(default="default", description="Population/group identifier")
    text: str = Field(description="Document text content")

    @classmethod
    def from_dict(cls, data: dict) -> "Document":
        """Create a Document from a dictionary, handling missing fields."""
        return cls(
            doc_id=str(data.get("doc_id", data.get("id", ""))),
            author_id=str(data.get("author_id", data.get("author", "unknown"))),
            population=str(data.get("population", data.get("group", "default"))),
            text=str(data.get("text", data.get("content", ""))),
        )


class CorpusLoader:
    """Loader for corpus files in various formats."""

    SUPPORTED_FORMATS = {".jsonl", ".json", ".csv", ".parquet"}

    def __init__(self, path: str | Path):
        """Initialize the loader with a file path.

        Args:
            path: Path to the corpus file.
        """
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(f"Corpus file not found: {self.path}")

        self.suffix = self.path.suffix.lower()
        if self.suffix not in self.SUPPORTED_FORMATS:
            raise ValueError(
                f"Unsupported format: {self.suffix}. "
                f"Supported: {self.SUPPORTED_FORMATS}"
            )

    def load(self) -> list[Document]:
        """Load all documents from the corpus file.

        Returns:
            List of Document objects.
        """
        return list(self.iter_documents())

    def iter_documents(self) -> Iterator[Document]:
        """Iterate over documents in the corpus file.

        Yields:
            Document objects.
        """
        if self.suffix == ".jsonl":
            yield from self._load_jsonl()
        elif self.suffix == ".json":
            yield from self._load_json()
        elif self.suffix == ".csv":
            yield from self._load_csv()
        elif self.suffix == ".parquet":
            yield from self._load_parquet()

    def _load_jsonl(self) -> Iterator[Document]:
        """Load documents from a JSONL file."""
        with open(self.path, encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    doc = Document.from_dict(data)
                    if not doc.doc_id:
                        doc.doc_id = f"doc_{line_num}"
                    yield doc
                except json.JSONDecodeError as e:
                    raise ValueError(f"Invalid JSON at line {line_num}: {e}") from e

    def _load_json(self) -> Iterator[Document]:
        """Load documents from a JSON file (array of objects)."""
        with open(self.path, encoding="utf-8") as f:
            data = json.load(f)

        if isinstance(data, list):
            for i, item in enumerate(data):
                doc = Document.from_dict(item)
                if not doc.doc_id:
                    doc.doc_id = f"doc_{i + 1}"
                yield doc
        elif isinstance(data, dict):
            # Single document
            doc = Document.from_dict(data)
            if not doc.doc_id:
                doc.doc_id = "doc_1"
            yield doc
        else:
            raise ValueError("JSON file must contain an array or object")

    def _load_csv(self) -> Iterator[Document]:
        """Load documents from a CSV file."""
        df = pd.read_csv(self.path)
        yield from self._dataframe_to_documents(df)

    def _load_parquet(self) -> Iterator[Document]:
        """Load documents from a Parquet file."""
        df = pd.read_parquet(self.path)
        yield from self._dataframe_to_documents(df)

    def _dataframe_to_documents(self, df: pd.DataFrame) -> Iterator[Document]:
        """Convert a DataFrame to Document objects."""
        # Map common column names
        column_mapping = {
            "id": "doc_id",
            "author": "author_id",
            "group": "population",
            "content": "text",
        }

        for old_name, new_name in column_mapping.items():
            if old_name in df.columns and new_name not in df.columns:
                df = df.rename(columns={old_name: new_name})

        # Check for required text column
        text_col = None
        for col in ["text", "content"]:
            if col in df.columns:
                text_col = col
                break

        if text_col is None:
            raise ValueError("CSV/Parquet must have a 'text' or 'content' column")

        for i, row in df.iterrows():
            doc_data = row.to_dict()
            doc = Document.from_dict(doc_data)
            if not doc.doc_id:
                doc.doc_id = f"doc_{i + 1}"
            yield doc


def load_corpus(path: str | Path) -> list[Document]:
    """Load a corpus from a file.

    Args:
        path: Path to the corpus file (JSONL, JSON, CSV, or Parquet).

    Returns:
        List of Document objects.
    """
    loader = CorpusLoader(path)
    return loader.load()
