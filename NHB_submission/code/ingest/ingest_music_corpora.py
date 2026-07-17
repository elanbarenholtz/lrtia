#!/usr/bin/env python3
"""
ingest_music_corpora.py

Convert MIDI files from each music corpus into the jsonl format consumed by
Music_Finegrain_v1.ipynb: one record per recording with pre-tokenized event
stream, so the notebook can load tokens directly without re-tokenizing on GPU.

Tokenization: Anticipatory Music Transformer (Stanford CRFM). MIDI → flat list
of integer tokens, 3 tokens per musical event (time, duration, note).

Usage:
    python scripts/ingest_music_corpora.py \\
        --midi-dir data/music_raw/weimar_jazz \\
        --corpus improvised \\
        --output data/music_processed/improvised.jsonl

    python scripts/ingest_music_corpora.py \\
        --midi-dir data/music_raw/maestro \\
        --corpus composed \\
        --output data/music_processed/composed.jsonl \\
        --max-files 300
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Iterator


def iter_midi_files(root: Path) -> Iterator[Path]:
    """Walk a directory for .mid / .midi files."""
    for ext in ("*.mid", "*.midi", "*.MID", "*.MIDI"):
        yield from root.rglob(ext)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--midi-dir", type=Path, required=True,
                   help="Directory of MIDI files (searched recursively)")
    p.add_argument("--corpus", required=True,
                   choices=["improvised", "composed", "ai_generated"],
                   help="Corpus label for the output records")
    p.add_argument("--output", type=Path, required=True,
                   help="Output jsonl path")
    p.add_argument("--min-tokens", type=int, default=300,
                   help="Drop recordings with fewer tokens than this "
                        "(need >= 2*MIN_CONTEXT_BEFORE_TARGET in notebook)")
    p.add_argument("--max-files", type=int, default=None,
                   help="Cap number of files processed")
    p.add_argument("--max-tokens", type=int, default=8192,
                   help="Truncate per-recording token streams to this length "
                        "(avoids outlier recordings dominating)")
    args = p.parse_args()

    try:
        from anticipation.convert import midi_to_events
    except ImportError:
        sys.exit("anticipation package not installed. "
                 "Run: pip install git+https://github.com/jthickstun/anticipation.git")

    if not args.midi_dir.exists():
        sys.exit(f"MIDI dir not found: {args.midi_dir}")

    args.output.parent.mkdir(parents=True, exist_ok=True)

    files = sorted(iter_midi_files(args.midi_dir))
    if args.max_files is not None:
        files = files[: args.max_files]
    if not files:
        sys.exit(f"No MIDI files found under {args.midi_dir}")

    print(f"Found {len(files)} MIDI files under {args.midi_dir}")
    n_kept, n_skipped, n_failed = 0, 0, 0

    with open(args.output, "w", encoding="utf-8") as out:
        for i, path in enumerate(files):
            try:
                tokens = midi_to_events(str(path))
            except Exception as e:
                n_failed += 1
                print(f"  [{i:>4}] FAIL  {path.name}: {e}", file=sys.stderr)
                continue

            n_tokens = len(tokens)
            if n_tokens < args.min_tokens:
                n_skipped += 1
                continue
            if n_tokens > args.max_tokens:
                tokens = tokens[: args.max_tokens]
                n_tokens = len(tokens)

            rec = {
                "doc_id": f"{args.corpus}::{path.stem}",
                "corpus": args.corpus,
                "source_path": str(path.relative_to(args.midi_dir)),
                "tokens": tokens,
                "n_tokens": n_tokens,
                "n_events": n_tokens // 3,
            }
            out.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n_kept += 1
            if n_kept % 25 == 0:
                print(f"  [{i:>4}] kept {n_kept}, skipped {n_skipped}, failed {n_failed}")

    print(f"\nDone. Wrote {n_kept} records to {args.output}")
    print(f"  skipped (too short): {n_skipped}")
    print(f"  failed to tokenize:  {n_failed}")


if __name__ == "__main__":
    main()
