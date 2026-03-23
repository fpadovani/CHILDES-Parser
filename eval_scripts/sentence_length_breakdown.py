"""
Split CoNLL-U files by sentence length into 4 bins:
  - le3:   length <= 3
  - 4to6:  4 <= length <= 6
  - 7to10: 7 <= length <= 10
  - gt10:  length > 10

Sentence length = number of regular token lines with UPOS != PUNCT
(excludes comments, multiword tokens like 1-2, empty nodes like 1.1,
and punctuation tokens — so e.g. "What are you ?" counts as length 3).
"""

import os
import re

FILES = [
    # Gold files
    "/Users/frapadovani/Desktop/CHILDES-Parser/UD_English-CHILDES/en_childes-ud-test_adults.conllu",
    "/Users/frapadovani/Desktop/CHILDES-Parser/UD_English-CHILDES/en_childes-ud-test_childes.conllu",
    # Off-the-shelf Stanza predictions
    "/Users/frapadovani/Desktop/CHILDES-Parser/eval_scripts/prediction_files_by_speaker/off_the_shelf_test_adults.conllu",
    "/Users/frapadovani/Desktop/CHILDES-Parser/eval_scripts/prediction_files_by_speaker/off_the_shelf_test_childes.conllu",
    # SuPar predictions
    "/Users/frapadovani/Desktop/CHILDES-Parser/eval_scripts/prediction_files_by_speaker/supar_childes_roberta_adults.conllu",
    "/Users/frapadovani/Desktop/CHILDES-Parser/eval_scripts/prediction_files_by_speaker/supar_childes_roberta_childes.conllu",
]

BINS = [
    ("le3",   lambda n: n <= 3),
    ("4to6",  lambda n: 4 <= n <= 6),
    ("7to10", lambda n: 7 <= n <= 10),
    ("gt10",  lambda n: n > 10),
]


def is_token_line(line):
    """True for regular token lines (not comments, not multiword, not empty nodes,
    and not punctuation — UPOS == PUNCT)."""
    if line.startswith("#") or line.strip() == "":
        return False
    parts = line.split("\t")
    idx = parts[0]
    if "-" in idx or "." in idx:
        return False
    # CoNLL-U column 4 (0-indexed) is UPOS
    if len(parts) > 3 and parts[3].strip() == "PUNCT":
        return False
    return True


def read_sentences(path):
    """Return list of (length, raw_lines_including_trailing_blank) tuples."""
    sentences = []
    current = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            current.append(line)
            if line.strip() == "":
                if current:
                    length = sum(1 for l in current if is_token_line(l))
                    sentences.append((length, current))
                    current = []
        # Handle file with no trailing newline
        if current:
            length = sum(1 for l in current if is_token_line(l))
            sentences.append((length, current))
    return sentences


def split_file(path):
    sentences = read_sentences(path)
    base, ext = os.path.splitext(path)

    counts = {name: 0 for name, _ in BINS}
    handles = {}
    for name, _ in BINS:
        out_path = f"{base}_{name}{ext}"
        handles[name] = open(out_path, "w", encoding="utf-8")

    for length, lines in sentences:
        for name, condition in BINS:
            if condition(length):
                handles[name].writelines(lines)
                # Ensure trailing blank line
                if lines[-1].strip() != "":
                    handles[name].write("\n")
                counts[name] += 1
                break

    for h in handles.values():
        h.close()

    print(f"\n{os.path.basename(path)}")
    total = sum(counts.values())
    for name, _ in BINS:
        print(f"  [{name}] {counts[name]} sentences -> {base}_{name}{ext}")
    print(f"  Total: {total} sentences")


if __name__ == "__main__":
    for f in FILES:
        if not os.path.exists(f):
            print(f"WARNING: File not found: {f}")
            continue
        split_file(f)
    print("\nDone!")
