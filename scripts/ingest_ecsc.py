#!/usr/bin/env python3
"""ECSC Data Ingestion Pipeline.

Downloads, parses, and processes the Eugene Children's Story Corpus (ECSC)
from TalkBank/CHILDES for use with LRTIA analysis.

Usage:
    python scripts/ingest_ecsc.py
    python scripts/ingest_ecsc.py --config configs/ecsc_ingest.yaml
    python scripts/ingest_ecsc.py --skip-download  # Use existing data

For research use only. Please cite the ECSC corpus:
https://talkbank.org/childes/access/Frogs/English-ECSC.html
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
import zipfile
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
from urllib.request import urlretrieve

import pandas as pd

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from lrtia.data.ingestion import Document
from lrtia.data.ecsc.config import ECScPipelineConfig
from lrtia.data.ecsc.chat_parser import ECScCHATParser, CHATParseError
from lrtia.data.ecsc.chat_cleaner import CHATCleaner
from lrtia.data.ecsc.metadata import ECScMetadataLoader
from lrtia.data.ecsc.models import ECScTranscript, ECScSubjectMetadata


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class ECScPipeline:
    """ECSC data ingestion pipeline.

    Downloads, parses, and processes the ECSC corpus to produce
    LRTIA-compatible JSONL documents.
    """

    def __init__(self, config: ECScPipelineConfig):
        """Initialize pipeline.

        Args:
            config: Pipeline configuration.
        """
        self.config = config

        # Initialize components
        self.cleaner = CHATCleaner(
            remove_brackets=config.cleaning.remove_brackets,
            remove_angles=config.cleaning.remove_angles,
            remove_ampersand_codes=config.cleaning.remove_ampersand_codes,
            remove_unintelligible=config.cleaning.remove_unintelligible,
            remove_pauses=config.cleaning.remove_pauses,
            remove_special_chars=config.cleaning.remove_special_chars,
            preserve_fillers=config.cleaning.preserve_fillers,
            collapse_whitespace=config.cleaning.collapse_whitespace,
        )
        self.parser = ECScCHATParser(
            child_roles=config.parsing.child_roles,
            cleaner=self.cleaner,
            require_child_speaker=True,
        )
        self.metadata_loader = ECScMetadataLoader(config.metadata_path)

        # State tracking
        self._parsed_transcripts: list[ECScTranscript] = []
        self._failed_files: list[tuple[Path, str]] = []
        self._transcript_documents: list[Document] = []
        self._concatenated_documents: list[Document] = []
        self._zip_sha256: Optional[str] = None

    def run(self, skip_download: bool = False) -> None:
        """Run the complete pipeline.

        Args:
            skip_download: Skip downloading, use existing data.
        """
        logger.info("Starting ECSC ingestion pipeline")
        start_time = datetime.now(timezone.utc)

        # Step 1: Download and extract
        if not skip_download:
            self._download_and_extract()
        else:
            logger.info("Skipping download (--skip-download flag)")
            # Still compute checksum if file exists
            zip_path = self.config.download.output_dir / "English-ECSC.zip"
            if zip_path.exists():
                self._zip_sha256 = self._compute_sha256(zip_path)

        # Step 2: Load metadata if configured
        if self.config.metadata_path:
            logger.info(f"Loading metadata from {self.config.metadata_path}")
            self.metadata_loader.load()
        else:
            # Try to find metadata file in downloaded data
            self._auto_discover_metadata()

        # Step 3: Parse CHAT files
        self._parse_all_files()

        # Step 4: Generate outputs
        self._generate_outputs()

        # Step 5: Generate QA report and manifest
        if self.config.qa.generate_qa_report or self.config.qa.generate_manifest:
            self._generate_qa_artifacts(start_time)

        # Summary
        logger.info("=" * 60)
        logger.info("Pipeline complete!")
        logger.info(f"  Transcripts parsed: {len(self._parsed_transcripts)}")
        logger.info(f"  Files failed: {len(self._failed_files)}")
        logger.info(f"  Transcript documents: {len(self._transcript_documents)}")
        logger.info(f"  Concatenated documents: {len(self._concatenated_documents)}")
        logger.info(f"  Output directory: {self.config.output.output_dir}")
        logger.info("=" * 60)

    def _download_and_extract(self) -> None:
        """Download ZIP and extract contents."""
        output_dir = self.config.download.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)

        zip_path = output_dir / "English-ECSC.zip"
        extract_dir = output_dir / "English-ECSC"

        # Check if already downloaded
        if self.config.download.cache_download and zip_path.exists():
            logger.info(f"ZIP already exists at {zip_path}, skipping download")
        else:
            logger.info(f"Downloading from {self.config.download.url}")

            def progress_hook(count, block_size, total_size):
                if total_size > 0:
                    percent = min(100, count * block_size * 100 // total_size)
                    print(f"\rDownloading: {percent}%", end="", flush=True)

            urlretrieve(self.config.download.url, zip_path, progress_hook)
            print()  # New line after progress
            logger.info(f"Downloaded to {zip_path}")

        # Compute checksum
        self._zip_sha256 = self._compute_sha256(zip_path)
        logger.info(f"ZIP SHA256: {self._zip_sha256}")

        # Verify checksum if expected
        if (
            self.config.download.verify_checksum
            and self.config.download.expected_sha256
        ):
            if self._zip_sha256 != self.config.download.expected_sha256:
                raise ValueError(
                    f"Checksum mismatch! Expected {self.config.download.expected_sha256}, "
                    f"got {self._zip_sha256}"
                )
            logger.info("Checksum verified")

        # Extract
        if extract_dir.exists():
            logger.info(f"Extract directory exists at {extract_dir}, skipping extraction")
        else:
            logger.info(f"Extracting to {extract_dir}")
            with zipfile.ZipFile(zip_path, "r") as zf:
                zf.extractall(output_dir)
            logger.info("Extraction complete")

    def _auto_discover_metadata(self) -> None:
        """Try to automatically discover metadata file in downloaded data."""
        extract_dir = self.config.download.output_dir / "English-ECSC"
        if not extract_dir.exists():
            return

        # Look for common metadata file patterns
        patterns = ["*.csv", "*.tsv", "*.xlsx", "*metadata*", "*subjects*"]
        for pattern in patterns:
            files = list(extract_dir.rglob(pattern))
            for f in files:
                if f.is_file() and f.suffix.lower() in {".csv", ".tsv", ".xlsx", ".xls"}:
                    logger.info(f"Auto-discovered metadata file: {f}")
                    self.metadata_loader = ECScMetadataLoader(f)
                    try:
                        self.metadata_loader.load()
                        return
                    except Exception as e:
                        logger.warning(f"Could not load {f}: {e}")

    def _parse_all_files(self) -> None:
        """Parse all CHAT files in extracted directory."""
        extract_dir = self.config.download.output_dir / "English-ECSC"

        if not extract_dir.exists():
            raise FileNotFoundError(f"Extract directory not found: {extract_dir}")

        logger.info(f"Parsing CHAT files from {extract_dir}")

        for file_path, result in self.parser.parse_directory(
            extract_dir,
            pattern="*.cha",
            recursive=True,
            skip_errors=self.config.parsing.skip_malformed,
        ):
            if isinstance(result, CHATParseError):
                logger.warning(f"Failed to parse {file_path}: {result.reason}")
                self._failed_files.append((file_path, result.reason))
            else:
                # Check minimum utterances
                child_utts = len(result.child_utterances)
                if child_utts < self.config.parsing.min_utterances:
                    logger.debug(
                        f"Skipping {file_path}: only {child_utts} child utterances"
                    )
                    continue

                self._parsed_transcripts.append(result)

                # Log speaker coverage for QA
                total_utts = len(result.utterances)
                child_ratio = child_utts / total_utts if total_utts > 0 else 0
                if child_ratio < self.config.qa.warn_child_ratio_threshold:
                    logger.warning(
                        f"{file_path}: child utterance ratio {child_ratio:.1%} "
                        f"is below threshold {self.config.qa.warn_child_ratio_threshold:.1%}"
                    )

        logger.info(
            f"Parsed {len(self._parsed_transcripts)} transcripts, "
            f"{len(self._failed_files)} failed"
        )

    def _generate_outputs(self) -> None:
        """Generate all output files."""
        output_dir = self.config.output.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)

        if self.config.output.generate_transcripts:
            self._generate_transcript_documents()

        if self.config.output.generate_concatenated:
            self._generate_concatenated_documents()

        if self.config.output.generate_utterances:
            self._generate_utterances_parquet()

    def _generate_transcript_documents(self) -> None:
        """Generate transcript-level JSONL."""
        output_path = self.config.output.output_dir / "transcripts.jsonl"
        logger.info(f"Generating transcript documents to {output_path}")

        self._transcript_documents = []
        skipped_short = 0

        for transcript in self._parsed_transcripts:
            doc = self._create_document(transcript)

            # Check minimum length
            word_count = len(doc.text.split())
            if word_count < self.config.output.min_tokens:
                skipped_short += 1
                continue

            self._transcript_documents.append(doc)

        # Write JSONL
        with open(output_path, "w", encoding="utf-8") as f:
            for doc in self._transcript_documents:
                f.write(
                    json.dumps(
                        {
                            "doc_id": doc.doc_id,
                            "author_id": doc.author_id,
                            "population": doc.population,
                            "text": doc.text,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )

        logger.info(
            f"Wrote {len(self._transcript_documents)} transcript documents "
            f"({skipped_short} skipped as too short)"
        )

    def _generate_concatenated_documents(self) -> None:
        """Generate concatenated documents grouped by subject."""
        output_path = self.config.output.output_dir / "concatenated_docs.jsonl"
        logger.info(f"Generating concatenated documents to {output_path}")

        # Group transcripts by subject
        by_subject: dict[str, list[ECScTranscript]] = defaultdict(list)
        for transcript in self._parsed_transcripts:
            by_subject[transcript.subject_id].append(transcript)

        self._concatenated_documents = []
        skipped_short = 0

        for subject_id, transcripts in by_subject.items():
            # Sort by configured fields
            def sort_key(t: ECScTranscript):
                return tuple(
                    getattr(t, field) for field in self.config.output.concatenation_sort
                )

            transcripts = sorted(transcripts, key=sort_key)

            # Get text from each transcript
            texts = []
            for t in transcripts:
                if self.config.output.text_field == "clean":
                    texts.append(t.clean_child_text)
                else:
                    texts.append(t.raw_child_text)

            # Concatenate
            concatenated_text = self.config.output.concatenation_separator.join(texts)

            # Check minimum length
            word_count = len(concatenated_text.split())
            if word_count < self.config.output.min_tokens:
                skipped_short += 1
                continue

            # Get metadata from first transcript (or combine?)
            first = transcripts[0]
            metadata = self.metadata_loader.get_subject_metadata(
                subject_id=subject_id,
                age_months=first.age_months,
                sex=first.sex,
                study_year=first.study_year,
            )

            doc = Document(
                doc_id=f"ECSC_{subject_id}_concatenated",
                author_id=subject_id,
                population=metadata.to_population_string(),
                text=concatenated_text,
            )
            self._concatenated_documents.append(doc)

        # Write JSONL
        with open(output_path, "w", encoding="utf-8") as f:
            for doc in self._concatenated_documents:
                f.write(
                    json.dumps(
                        {
                            "doc_id": doc.doc_id,
                            "author_id": doc.author_id,
                            "population": doc.population,
                            "text": doc.text,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )

        logger.info(
            f"Wrote {len(self._concatenated_documents)} concatenated documents "
            f"({skipped_short} skipped as too short)"
        )

    def _generate_utterances_parquet(self) -> None:
        """Generate utterance-level parquet for auditability."""
        output_path = self.config.output.output_dir / "utterances.parquet"
        logger.info(f"Generating utterances parquet to {output_path}")

        rows = []
        for transcript in self._parsed_transcripts:
            doc_id = f"ECSC_{transcript.subject_id}_{transcript.study_year}_{transcript.file_stem}"

            # Get metadata
            metadata = self.metadata_loader.get_subject_metadata(
                subject_id=transcript.subject_id,
                age_months=transcript.age_months,
                sex=transcript.sex,
                study_year=transcript.study_year,
            )

            for i, utt in enumerate(transcript.utterances):
                rows.append(
                    {
                        "subject_id": transcript.subject_id,
                        "doc_id": doc_id,
                        "utterance_idx": i,
                        "speaker_code": utt.speaker,
                        "speaker_role": utt.speaker_role,
                        "is_child": utt.speaker == transcript.child_speaker_code,
                        "age_months": transcript.age_months,
                        "sex": transcript.sex,
                        "study_year": transcript.study_year,
                        "ppvt_standard": metadata.ppvt_standard,
                        "ppvt_raw": metadata.ppvt_raw,
                        "utterance_text_raw": utt.raw_text,
                        "utterance_text_clean": utt.clean_text,
                        "line_number": utt.line_number,
                        "source_file": transcript.file_stem,
                    }
                )

        df = pd.DataFrame(rows)
        df.to_parquet(output_path, index=False)
        logger.info(f"Wrote {len(df)} utterances")

    def _generate_qa_artifacts(self, start_time: datetime) -> None:
        """Generate QA report and reproducibility manifest."""
        output_dir = self.config.output.output_dir

        # QA Report
        if self.config.qa.generate_qa_report:
            qa_path = output_dir / "qa_report.json"
            logger.info(f"Generating QA report to {qa_path}")

            # Collect statistics
            speaker_coverage = []
            text_lengths = []

            for transcript in self._parsed_transcripts:
                total_utts = len(transcript.utterances)
                child_utts = len(transcript.child_utterances)
                speaker_coverage.append(
                    {
                        "file": transcript.file_stem,
                        "total_utterances": total_utts,
                        "child_utterances": child_utts,
                        "child_ratio": child_utts / total_utts if total_utts > 0 else 0,
                    }
                )
                text_lengths.append(
                    {
                        "file": transcript.file_stem,
                        "word_count": len(transcript.clean_child_text.split()),
                        "char_count": len(transcript.clean_child_text),
                    }
                )

            qa_report = {
                "file_counts": {
                    "total_attempted": len(self._parsed_transcripts)
                    + len(self._failed_files),
                    "parsed": len(self._parsed_transcripts),
                    "failed": len(self._failed_files),
                },
                "failed_files": [
                    {"path": str(p), "reason": r} for p, r in self._failed_files
                ],
                "speaker_coverage": {
                    "per_file": speaker_coverage,
                    "mean_child_ratio": (
                        sum(s["child_ratio"] for s in speaker_coverage)
                        / len(speaker_coverage)
                        if speaker_coverage
                        else 0
                    ),
                },
                "text_lengths": {
                    "per_file": text_lengths,
                    "mean_word_count": (
                        sum(t["word_count"] for t in text_lengths) / len(text_lengths)
                        if text_lengths
                        else 0
                    ),
                    "min_word_count": (
                        min(t["word_count"] for t in text_lengths) if text_lengths else 0
                    ),
                    "max_word_count": (
                        max(t["word_count"] for t in text_lengths) if text_lengths else 0
                    ),
                },
                "metadata": self.metadata_loader.get_stats(),
                "output_counts": {
                    "transcript_documents": len(self._transcript_documents),
                    "concatenated_documents": len(self._concatenated_documents),
                    "unique_subjects": len(
                        set(t.subject_id for t in self._parsed_transcripts)
                    ),
                },
            }

            with open(qa_path, "w", encoding="utf-8") as f:
                json.dump(qa_report, f, indent=2, ensure_ascii=False)

        # Manifest
        if self.config.qa.generate_manifest:
            manifest_path = output_dir / "manifest.json"
            logger.info(f"Generating manifest to {manifest_path}")

            manifest = {
                "corpus": {
                    "name": "Eugene Children's Story Corpus (ECSC)",
                    "source": "TalkBank CHILDES Frogs Collection",
                    "url": "https://talkbank.org/childes/access/Frogs/English-ECSC.html",
                    "citation": "See corpus page for citation requirements",
                    "license": "Research use only",
                },
                "download": {
                    "url": self.config.download.url,
                    "zip_sha256": self._zip_sha256,
                },
                "processing": {
                    "timestamp": start_time.isoformat(),
                    "config": self.config.to_dict(),
                },
                "outputs": {
                    "transcripts_jsonl": str(output_dir / "transcripts.jsonl"),
                    "concatenated_jsonl": str(output_dir / "concatenated_docs.jsonl"),
                    "utterances_parquet": str(output_dir / "utterances.parquet"),
                },
                "counts": {
                    "files_processed": len(self._parsed_transcripts),
                    "files_failed": len(self._failed_files),
                    "subjects": len(
                        set(t.subject_id for t in self._parsed_transcripts)
                    ),
                    "transcript_documents": len(self._transcript_documents),
                    "concatenated_documents": len(self._concatenated_documents),
                },
            }

            with open(manifest_path, "w", encoding="utf-8") as f:
                json.dump(manifest, f, indent=2, ensure_ascii=False)

    def _create_document(self, transcript: ECScTranscript) -> Document:
        """Create an LRTIA Document from a transcript.

        Args:
            transcript: Parsed transcript.

        Returns:
            LRTIA Document.
        """
        # Get metadata
        metadata = self.metadata_loader.get_subject_metadata(
            subject_id=transcript.subject_id,
            age_months=transcript.age_months,
            sex=transcript.sex,
            study_year=transcript.study_year,
        )

        # Select text version
        if self.config.output.text_field == "clean":
            text = transcript.clean_child_text
        else:
            text = transcript.raw_child_text

        # Build document ID
        doc_id = f"ECSC_{transcript.subject_id}_{transcript.study_year}_{transcript.file_stem}"

        return Document(
            doc_id=doc_id,
            author_id=transcript.subject_id,
            population=metadata.to_population_string(),
            text=text,
        )

    @staticmethod
    def _compute_sha256(path: Path) -> str:
        """Compute SHA256 hash of a file."""
        sha256 = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                sha256.update(chunk)
        return sha256.hexdigest()


def main():
    parser = argparse.ArgumentParser(
        description="ECSC Data Ingestion Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
For research use only. Please cite the ECSC corpus:
https://talkbank.org/childes/access/Frogs/English-ECSC.html
        """,
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to configuration YAML file",
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Skip downloading, use existing data",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Override output directory",
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        default=None,
        help="Path to subject metadata file (CSV/TSV/Excel)",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose (debug) logging",
    )
    args = parser.parse_args()

    # Set log level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Load configuration
    if args.config:
        logger.info(f"Loading configuration from {args.config}")
        config = ECScPipelineConfig.from_yaml(args.config)
    else:
        config = ECScPipelineConfig()

    # Apply overrides
    if args.output_dir:
        config.output.output_dir = args.output_dir
    if args.metadata:
        config.metadata_path = args.metadata

    # Run pipeline
    pipeline = ECScPipeline(config)
    pipeline.run(skip_download=args.skip_download)


if __name__ == "__main__":
    main()
