#!/usr/bin/env python3
"""Generate synthetic corpora for sanity check with controlled coherence.

All three conditions use the SAME underlying content/vocabulary.
Only the coherence structure differs:

- Narrative: Long-range dependencies (pronouns reference distant antecedents)
- Paragraph: Medium-range dependencies (pronouns only within paragraphs)
- Shuffled: No dependencies (same sentences as narrative, but randomly ordered)

Expected results:
- Narrative: LONGEST half-life (distant context matters for pronoun resolution)
- Paragraph: MEDIUM half-life (only local context matters)
- Shuffled: SHORTEST half-life (no predictable structure)
"""

import json
import random
from pathlib import Path
from typing import List, Tuple


# Common vocabulary used across ALL conditions
CHARACTERS = [
    ("John", "he", "his", "him"),
    ("Sarah", "she", "her", "her"),
    ("Michael", "he", "his", "him"),
    ("Emma", "she", "her", "her"),
    ("David", "he", "his", "him"),
    ("Lisa", "she", "her", "her"),
]

OBJECTS = [
    "the old book", "the letter", "the key", "the photograph",
    "the journal", "the map", "the box", "the note",
]

LOCATIONS = [
    "the house", "the garden", "the library", "the study",
    "the attic", "the kitchen", "the hallway", "the room",
]

# Sentence templates - these create the base content
# {char} = character name, {obj} = object, {loc} = location
# {subj}/{poss}/{obj_pron} = pronouns for the current character

INTRO_TEMPLATES = [
    "{char} walked into {loc} and noticed {obj} on the table.",
    "{char} found {obj} while searching through {loc}.",
    "In {loc}, {char} discovered {obj} hidden away.",
    "{char} entered {loc} and immediately saw {obj}.",
]

CONTINUATION_TEMPLATES = [
    "{subj} picked it up and examined it closely.",
    "{subj} wondered where it had come from.",
    "{subj} remembered seeing something similar before.",
    "{subj} turned it over in {poss} hands.",
    "{subj} felt a sense of recognition.",
    "{subj} decided to investigate further.",
    "{subj} looked around {loc} for more clues.",
    "{subj} thought about what this could mean.",
    "{subj} set it down and considered {poss} options.",
    "{subj} knew this was important somehow.",
    "This reminded {obj_pron} of something from the past.",
    "{subj} took a deep breath and focused.",
    "{subj} noticed details {subj} had missed before.",
    "{subj} began to understand the significance.",
    "{poss} mind raced with possibilities.",
]

ACTION_TEMPLATES = [
    "{subj} moved to another part of {loc}.",
    "{subj} sat down to think about {obj}.",
    "{subj} returned {poss} attention to {obj}.",
    "{subj} went to find more information.",
    "{subj} checked {obj} one more time.",
]


def generate_sentence_sequence(
    n_sentences: int,
    char_name: str,
    pronouns: Tuple[str, str, str],
    obj: str,
    loc: str,
    rng: random.Random,
) -> List[str]:
    """Generate a coherent sequence of sentences about a character."""
    subj, poss, obj_pron = pronouns
    sentences = []

    # Start with an intro
    intro = rng.choice(INTRO_TEMPLATES).format(
        char=char_name, obj=obj, loc=loc,
        subj=subj, poss=poss, obj_pron=obj_pron
    )
    sentences.append(intro)

    # Continue with pronoun-based sentences
    for i in range(n_sentences - 1):
        if rng.random() < 0.15:
            # Occasionally use an action template
            template = rng.choice(ACTION_TEMPLATES)
        else:
            template = rng.choice(CONTINUATION_TEMPLATES)

        # Capitalize pronoun at start of sentence
        sentence = template.format(
            char=char_name, obj=obj, loc=loc,
            subj=subj.capitalize(), poss=poss, obj_pron=obj_pron
        )
        sentences.append(sentence)

    return sentences


def generate_narrative_document(doc_id: int, n_sentences: int = 40, seed: int = None) -> dict:
    """Generate a document with LONG-RANGE narrative coherence.

    A single character is introduced at the start and referenced with pronouns
    throughout the entire document. Understanding the pronouns requires
    remembering the character from the beginning.
    """
    rng = random.Random(seed + doc_id if seed else None)

    # Pick one character for the whole document
    char_name, subj, poss, obj_pron = rng.choice(CHARACTERS)
    obj = rng.choice(OBJECTS)
    loc = rng.choice(LOCATIONS)

    sentences = generate_sentence_sequence(
        n_sentences, char_name, (subj, poss, obj_pron), obj, loc, rng
    )

    text = " ".join(sentences)

    return {
        "doc_id": f"narrative_{doc_id:03d}",
        "author_id": f"author_{doc_id % 5}",
        "population": "narrative",
        "text": text
    }


def generate_paragraph_document(doc_id: int, n_paragraphs: int = 6, sentences_per_para: int = 7, seed: int = None) -> dict:
    """Generate a document with PARAGRAPH-LEVEL coherence only.

    Each paragraph introduces a NEW character with their name, then uses pronouns.
    Pronouns only make sense within the current paragraph.
    Cross-paragraph context should NOT help prediction.
    """
    rng = random.Random(seed + doc_id + 500 if seed else None)

    all_sentences = []

    for para_idx in range(n_paragraphs):
        # Each paragraph gets a DIFFERENT character, object, location
        char_name, subj, poss, obj_pron = rng.choice(CHARACTERS)
        obj = rng.choice(OBJECTS)
        loc = rng.choice(LOCATIONS)

        para_sentences = generate_sentence_sequence(
            sentences_per_para, char_name, (subj, poss, obj_pron), obj, loc, rng
        )
        all_sentences.extend(para_sentences)

    text = " ".join(all_sentences)

    return {
        "doc_id": f"paragraph_{doc_id:03d}",
        "author_id": f"author_{doc_id % 5}",
        "population": "paragraph",
        "text": text
    }


def generate_shuffled_document(doc_id: int, n_sentences: int = 40, seed: int = None) -> dict:
    """Generate a document with NO coherence (shuffled narrative).

    Takes the same sentence content as a narrative document but shuffles
    the order randomly. Pronouns now have no clear antecedent.
    Context at any distance should NOT help prediction.
    """
    rng = random.Random(seed + doc_id + 1000 if seed else None)

    # Generate a narrative first (same content)
    char_name, subj, poss, obj_pron = rng.choice(CHARACTERS)
    obj = rng.choice(OBJECTS)
    loc = rng.choice(LOCATIONS)

    sentences = generate_sentence_sequence(
        n_sentences, char_name, (subj, poss, obj_pron), obj, loc, rng
    )

    # Shuffle the sentences randomly
    rng.shuffle(sentences)

    text = " ".join(sentences)

    return {
        "doc_id": f"shuffled_{doc_id:03d}",
        "author_id": f"author_{doc_id % 5}",
        "population": "shuffled",
        "text": text
    }


def main():
    output_dir = Path(__file__).parent.parent / "sanity_check_data"
    output_dir.mkdir(exist_ok=True)

    print("Generating sanity check corpora (controlled coherence)...")
    print()
    print("All conditions use the SAME vocabulary and sentence templates.")
    print("Only the COHERENCE STRUCTURE differs.")
    print()

    # Generate documents (interleaved so small samples include all types)
    n_docs_per_type = 25
    documents = []

    print(f"Generating {n_docs_per_type} narrative documents (long-range coherence)...")
    narrative_docs = [generate_narrative_document(i, n_sentences=45, seed=42) for i in range(n_docs_per_type)]

    print(f"Generating {n_docs_per_type} paragraph documents (paragraph-level coherence)...")
    paragraph_docs = [generate_paragraph_document(i, n_paragraphs=6, sentences_per_para=7, seed=42) for i in range(n_docs_per_type)]

    print(f"Generating {n_docs_per_type} shuffled documents (no coherence)...")
    shuffled_docs = [generate_shuffled_document(i, n_sentences=45, seed=42) for i in range(n_docs_per_type)]

    # Interleave documents so any subset includes all three populations
    for i in range(n_docs_per_type):
        documents.append(narrative_docs[i])
        documents.append(paragraph_docs[i])
        documents.append(shuffled_docs[i])

    # Save corpus
    corpus_path = output_dir / "sanity_check_corpus.jsonl"
    with open(corpus_path, "w") as f:
        for doc in documents:
            f.write(json.dumps(doc) + "\n")

    print()
    print(f"Saved {len(documents)} documents to: {corpus_path}")
    print()

    # Show samples
    print("=" * 70)
    print("SAMPLE: Narrative (single character, pronouns throughout)")
    print("=" * 70)
    print(documents[0]["text"][:600] + "...")
    print()

    print("=" * 70)
    print("SAMPLE: Paragraph (new character each paragraph)")
    print("=" * 70)
    print(documents[1]["text"][:600] + "...")
    print()

    print("=" * 70)
    print("SAMPLE: Shuffled (same content as narrative, random order)")
    print("=" * 70)
    print(documents[2]["text"][:600] + "...")
    print()

    print("=" * 70)
    print("Expected results:")
    print("  - Narrative: LONGEST half-life")
    print("      (pronouns require remembering character from document start)")
    print("  - Paragraph: MEDIUM half-life")
    print("      (pronouns only require local paragraph context)")
    print("  - Shuffled: SHORTEST half-life")
    print("      (no coherent structure, context doesn't help)")
    print()
    print("Key insight: Same vocabulary means similar baseline predictability.")
    print("Differences in half-life reflect coherence structure, not content.")
    print("=" * 70)


if __name__ == "__main__":
    main()
