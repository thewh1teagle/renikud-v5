"""
Prepare tokenized Arrow dataset from aligned JSONL for classifier training.

Reads train_alignment.jsonl produced by align_data.py and produces an Arrow
dataset with per-character consonant, vowel, and stress labels aligned to
BERT token positions.

Usage:
    uv run src/prepare_classifier_tokens.py dataset/train_alignment.jsonl dataset/.cache/classifier-train
    uv run src/prepare_classifier_tokens.py dataset/val_alignment.jsonl dataset/.cache/classifier-val
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import datasets
from tqdm import tqdm

from constants_classifier import (
    CONSONANT_TO_ID,
    VOWEL_TO_ID,
    STRESS_YES,
    STRESS_NONE,
    IGNORE_INDEX,
    is_hebrew_letter,
)
from tokenization import load_encoder_tokenizer

STRESS_MARK = "ˈ"
VOWELS_SET = set("aeiou")


def parse_ipa_chunk(chunk: str) -> tuple[str, str, int]:
    """
    Parse an IPA chunk into (consonant, vowel, stress).

    Chunk format: [ˈ][consonant][vowel]  e.g. "ʃa", "lˈo", "∅", "ˈa", "bi"
    Returns (consonant_str, vowel_str, stress_int) where:
      - consonant_str is the consonant or "∅" if silent/none
      - vowel_str is the vowel or "∅" if none
      - stress_int is STRESS_YES or STRESS_NONE
    """
    if not chunk or chunk == " ":
        return ("∅", "∅", STRESS_NONE)

    pos = 0
    stress = STRESS_NONE

    # Stress mark comes before the vowel in the chunk
    if STRESS_MARK in chunk:
        stress = STRESS_YES
        chunk = chunk.replace(STRESS_MARK, "")

    # Try to match known multi-char consonants first (ts, tʃ, dʒ)
    consonant = "∅"
    for multi in ("tʃ", "dʒ", "ts"):
        if chunk.startswith(multi):
            consonant = multi
            pos = len(multi)
            break
    else:
        # Single char consonant or vowel-only
        if pos < len(chunk) and chunk[pos] not in VOWELS_SET:
            consonant = chunk[pos]
            pos += 1

    # Remaining is the vowel
    vowel = chunk[pos:] if pos < len(chunk) else "∅"
    if not vowel:
        vowel = "∅"

    # Special case: [vowel]aχ pattern (word-final ח) e.g. "uaχ", "oaχ", "eaχ"
    # The aligner assigns the whole diphthong to ח — aχ encodes the full coda, consonant is ∅
    if vowel not in VOWEL_TO_ID and vowel.endswith("aχ"):
        consonant = "∅"
        vowel = "aχ"
    # Plain "aχ" chunk (ח -> "aχ"): consonant is already embedded in vowel token
    if vowel == "aχ":
        consonant = "∅"

    # Validate — fall back to ∅ if unknown
    if consonant not in CONSONANT_TO_ID:
        consonant = "∅"
    if vowel not in VOWEL_TO_ID:
        vowel = "∅"

    return (consonant, vowel, stress)


def process_sentence(
    hebrew: str,
    alignment: list[list[str]],
    tokenizer,
) -> dict | None:
    """
    Tokenize the Hebrew sentence and align per-character labels to token positions.
    Returns None if tokenization produces unexpected token count.
    """
    encoding = tokenizer(
        hebrew,
        truncation=True,
        max_length=512,
        return_offsets_mapping=True,
        return_tensors=None,
    )

    input_ids = encoding["input_ids"]
    attention_mask = encoding["attention_mask"]
    offset_mapping = encoding["offset_mapping"]

    seq_len = len(input_ids)
    consonant_labels = [IGNORE_INDEX] * seq_len
    vowel_labels = [IGNORE_INDEX] * seq_len
    stress_labels = [IGNORE_INDEX] * seq_len

    # Build char_index -> (consonant, vowel, stress) from alignment.
    # The alignment pairs only contain Hebrew letters and spaces (punctuation,
    # digits, Latin chars were stripped by the aligner), so we walk the original
    # sentence to get correct offsets for the tokenizer's offset_mapping.
    char_labels: dict[int, tuple[str, str, int]] = {}
    align_iter = iter(alignment)
    for char_pos, orig_char in enumerate(hebrew):
        if not is_hebrew_letter(orig_char) and orig_char != " ":
            continue  # punctuation/digit/Latin — not in alignment, skip
        try:
            align_char, chunk = next(align_iter)
        except StopIteration:
            break
        if is_hebrew_letter(orig_char):
            char_labels[char_pos] = parse_ipa_chunk(chunk)

    # Map token positions to char positions using offset_mapping
    for tok_idx, (start, end) in enumerate(offset_mapping):
        if end - start != 1:
            # CLS, SEP, or multi-char token — ignore
            continue
        char_idx = start
        if char_idx in char_labels:
            consonant, vowel, stress = char_labels[char_idx]
            consonant_labels[tok_idx] = CONSONANT_TO_ID.get(consonant, IGNORE_INDEX)
            vowel_labels[tok_idx] = VOWEL_TO_ID.get(vowel, IGNORE_INDEX)
            stress_labels[tok_idx] = stress

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "consonant_labels": consonant_labels,
        "vowel_labels": vowel_labels,
        "stress_labels": stress_labels,
    }


def main():
    parser = argparse.ArgumentParser(description="Prepare classifier training tokens from aligned JSONL")
    parser.add_argument("input", help="Input JSONL file (from align_data.py)")
    parser.add_argument("output", help="Output Arrow dataset directory")
    args = parser.parse_args()

    tokenizer = load_encoder_tokenizer()

    records = []
    skipped = 0

    with open(args.input, encoding="utf-8") as f:
        lines = f.readlines()

    for line in tqdm(lines, desc="Tokenizing"):
        line = line.strip()
        if not line:
            continue
        obj = json.loads(line)
        hebrew, alignment = next(iter(obj.items()))

        record = process_sentence(hebrew, alignment, tokenizer)
        if record is None:
            skipped += 1
            continue
        records.append(record)

    print(f"\nProcessed: {len(records):,}")
    print(f"Skipped:   {skipped:,}")

    dataset = datasets.Dataset.from_list(records)
    dataset.save_to_disk(args.output)
    print(f"Saved to:  {args.output}")


if __name__ == "__main__":
    main()
