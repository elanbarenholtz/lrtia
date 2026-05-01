"""Topic-matching infrastructure for D5 foreign-block insertion.

Builds per-corpus indices of paragraph-level embeddings to support:
  - D5a: same-topic foreign (cosine sim >= 0.6, different document)
  - D5b: different-topic foreign (cosine sim <= 0.2, different document)
  - D5c: document-specific nonlocal (same document, distant, opposite half)
  - D5d: shuffled/nonsense (D5a tokens randomly shuffled)

Uses paraphrase-multilingual-mpnet-base-v2 for multilingual embedding.
Only runs on the 6 Wikipedia corpora.
"""

from __future__ import annotations

import json
import logging
import random
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ParagraphInfo:
    """A paragraph with its embedding and position metadata."""
    doc_id: str
    doc_idx: int
    para_idx: int
    token_start: int  # token offset within document
    token_end: int
    tokens: list[int]
    embedding: np.ndarray | None = None


@dataclass
class DonorBlock:
    """A K-token block selected as a D5 donor."""
    tokens: list[int]
    source_doc_id: str
    source_para_idx: int
    variant: str  # 'd5a', 'd5b', 'd5c', 'd5d'
    cosine_sim: float | None = None


@dataclass
class CorpusIndex:
    """Per-corpus index for D5 donor selection."""
    corpus_name: str
    language: str
    paragraphs: list[ParagraphInfo] = field(default_factory=list)
    doc_embeddings: dict[str, np.ndarray] = field(default_factory=dict)
    _embeddings_matrix: np.ndarray | None = None
    _doc_ids: list[str] = field(default_factory=list)


class TopicMatcher:
    """Builds paragraph-level embedding indices and selects D5 donors."""

    def __init__(
        self,
        model_name: str = "paraphrase-multilingual-mpnet-base-v2",
        d5a_threshold: float = 0.6,
        d5b_threshold: float = 0.2,
        d5c_min_distance: int = 200,
        block_size: int = 20,
        seed: int = 42,
    ):
        self.model_name = model_name
        self.d5a_threshold = d5a_threshold
        self.d5b_threshold = d5b_threshold
        self.d5c_min_distance = d5c_min_distance
        self.block_size = block_size
        self.rng = random.Random(seed)
        self._model = None

    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def build_index(
        self,
        corpus_path: Path,
        tokenizer,
        corpus_name: str,
        language: str,
    ) -> CorpusIndex:
        """Build a paragraph-level embedding index for a Wikipedia corpus.

        Args:
            corpus_path: Path to the JSONL corpus file.
            tokenizer: HuggingFace tokenizer (for token-level positions).
            corpus_name: Human-readable corpus name.
            language: Language code.

        Returns:
            CorpusIndex with paragraph embeddings.
        """
        index = CorpusIndex(corpus_name=corpus_name, language=language)

        # Load documents
        docs = []
        with open(corpus_path) as f:
            for line in f:
                docs.append(json.loads(line))

        all_para_texts = []
        para_infos = []

        for doc_idx, doc in enumerate(docs):
            doc_id = doc.get('doc_id', f'doc_{doc_idx}')
            text = doc['text']
            full_tokens = tokenizer.encode(text, add_special_tokens=False)

            # Split into paragraphs (by sentence boundaries approximated
            # by splitting on periods followed by space, or every ~100 tokens)
            # Simple approach: chunk into ~100-token paragraphs
            chunk_size = 100
            token_pos = 0
            para_idx = 0

            while token_pos < len(full_tokens):
                end_pos = min(token_pos + chunk_size, len(full_tokens))
                para_tokens = full_tokens[token_pos:end_pos]

                if len(para_tokens) >= 20:  # minimum paragraph size
                    para_text = tokenizer.decode(para_tokens)
                    all_para_texts.append(para_text)

                    info = ParagraphInfo(
                        doc_id=doc_id,
                        doc_idx=doc_idx,
                        para_idx=para_idx,
                        token_start=token_pos,
                        token_end=end_pos,
                        tokens=para_tokens,
                    )
                    para_infos.append(info)
                    para_idx += 1

                token_pos = end_pos

        # Compute embeddings
        logger.info(f"  Computing embeddings for {len(all_para_texts)} paragraphs...")
        embeddings = self.model.encode(
            all_para_texts, show_progress_bar=False, batch_size=64
        )

        for info, emb in zip(para_infos, embeddings):
            info.embedding = emb

        index.paragraphs = para_infos

        # Compute per-document mean embeddings
        doc_embeds: dict[str, list[np.ndarray]] = {}
        for info in para_infos:
            doc_embeds.setdefault(info.doc_id, []).append(info.embedding)
        for doc_id, embs in doc_embeds.items():
            index.doc_embeddings[doc_id] = np.mean(embs, axis=0)

        logger.info(f"  Index built: {len(para_infos)} paragraphs, "
                     f"{len(doc_embeds)} documents")
        return index

    def _cosine_sim(self, a: np.ndarray, b: np.ndarray) -> float:
        """Cosine similarity between two vectors."""
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))

    def _extract_block(self, tokens: list[int]) -> list[int]:
        """Extract a contiguous K-token block from a token sequence.

        If len(tokens) >= K, take a random contiguous slice.
        If len(tokens) < K, return None (caller must handle).
        """
        K = self.block_size
        if len(tokens) < K:
            return None
        start = self.rng.randint(0, len(tokens) - K)
        return tokens[start:start + K]

    def find_d5a_donor(
        self, index: CorpusIndex, target_doc_id: str
    ) -> DonorBlock | None:
        """Find a same-topic foreign donor (different doc, cosine >= threshold).

        Returns None if no suitable donor found.
        """
        target_emb = index.doc_embeddings.get(target_doc_id)
        if target_emb is None:
            return None

        # Collect candidate paragraphs from other documents
        candidates = []
        for para in index.paragraphs:
            if para.doc_id == target_doc_id:
                continue
            sim = self._cosine_sim(target_emb, para.embedding)
            if sim >= self.d5a_threshold and len(para.tokens) >= self.block_size:
                candidates.append((para, sim))

        if not candidates:
            return None

        # Pick one at random from qualifying candidates
        para, sim = self.rng.choice(candidates)
        block = self._extract_block(para.tokens)
        if block is None:
            return None

        return DonorBlock(
            tokens=block,
            source_doc_id=para.doc_id,
            source_para_idx=para.para_idx,
            variant='d5a',
            cosine_sim=sim,
        )

    def find_d5b_donor(
        self, index: CorpusIndex, target_doc_id: str
    ) -> DonorBlock | None:
        """Find a different-topic foreign donor (cosine <= threshold)."""
        target_emb = index.doc_embeddings.get(target_doc_id)
        if target_emb is None:
            return None

        candidates = []
        for para in index.paragraphs:
            if para.doc_id == target_doc_id:
                continue
            sim = self._cosine_sim(target_emb, para.embedding)
            if sim <= self.d5b_threshold and len(para.tokens) >= self.block_size:
                candidates.append((para, sim))

        if not candidates:
            return None

        para, sim = self.rng.choice(candidates)
        block = self._extract_block(para.tokens)
        if block is None:
            return None

        return DonorBlock(
            tokens=block,
            source_doc_id=para.doc_id,
            source_para_idx=para.para_idx,
            variant='d5b',
            cosine_sim=sim,
        )

    def find_d5c_donor(
        self,
        index: CorpusIndex,
        target_doc_id: str,
        target_token_start: int,
        target_token_end: int,
        context_token_start: int,
        doc_total_tokens: int,
    ) -> DonorBlock | None:
        """Find a document-specific nonlocal donor.

        Constraints (per protocol §5, v3):
        - Same document as target
        - No overlap with the 100-token original context
        - >= 200 tokens from target region (in either direction)
        - From the opposite half of the document
        - Not adjacent to the target paragraph
        """
        doc_mid = doc_total_tokens // 2
        target_in_first_half = target_token_start < doc_mid

        candidates = []
        for para in index.paragraphs:
            if para.doc_id != target_doc_id:
                continue
            if len(para.tokens) < self.block_size:
                continue

            # No overlap with original 100-token context
            if para.token_end > context_token_start and para.token_start < target_token_end:
                continue

            # >= 200 tokens from target region
            dist_before = target_token_start - para.token_end
            dist_after = para.token_start - target_token_end
            min_dist = max(dist_before, dist_after, 0)
            if min_dist < self.d5c_min_distance:
                continue

            # Opposite half of document
            para_in_first_half = para.token_start < doc_mid
            if para_in_first_half == target_in_first_half:
                continue

            # Not adjacent to target paragraph (already covered by >= 200 distance)
            candidates.append(para)

        if not candidates:
            return None

        para = self.rng.choice(candidates)
        block = self._extract_block(para.tokens)
        if block is None:
            return None

        return DonorBlock(
            tokens=block,
            source_doc_id=para.doc_id,
            source_para_idx=para.para_idx,
            variant='d5c',
        )

    def make_d5d_block(self, d5a_donor: DonorBlock) -> DonorBlock:
        """Create a shuffled/nonsense block from a D5a donor's tokens.

        Same K tokens as D5a, randomly shuffled within the block.
        """
        shuffled_tokens = list(d5a_donor.tokens)
        self.rng.shuffle(shuffled_tokens)
        return DonorBlock(
            tokens=shuffled_tokens,
            source_doc_id=d5a_donor.source_doc_id,
            source_para_idx=d5a_donor.source_para_idx,
            variant='d5d',
        )
