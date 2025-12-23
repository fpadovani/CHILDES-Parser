import re
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict, Counter

def extract_sentences_with_errors(input_paths, output_dir="error_sentences"):
    """
    Extract sentences that contain at least one token where Gold and Pred head or label mismatch.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    for input_path in input_paths:
        input_path = Path(input_path)
        output_path = output_dir / f"{input_path.stem}_errors.conllu"

        with open(input_path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        # Split file into sentences
        sentences = []
        current_sentence = []
        for line in lines:
            if line.strip() == "------------------------------------------------------------":
                if current_sentence:
                    sentences.append(current_sentence)
                    current_sentence = []
            else:
                current_sentence.append(line.rstrip("\n"))
        if current_sentence:
            sentences.append(current_sentence)

        # Process sentences
        error_sentences = []

        for sent in tqdm(sentences, desc=f"Processing {input_path.name}"):
            has_error = False
            # Iterate lines with index
            for i, line in enumerate(sent):
                if line.startswith("Gold: token="):
                    gold_line = line
                    pred_line = sent[i + 1] if i + 1 < len(sent) else ""
                    # Extract head and label
                    gold_match = re.search(r"head=(\d+), label=([^\s]+)", gold_line)
                    pred_match = re.search(r"head=(\d+), label=([^\s]+)", pred_line)
                    if gold_match and pred_match:
                        gold_head, gold_label = int(gold_match.group(1)), gold_match.group(2)
                        pred_head, pred_label = int(pred_match.group(1)), pred_match.group(2)
                        if gold_head != pred_head or gold_label != pred_label:
                            has_error = True
                            break  # one error is enough to keep the sentence

            if has_error:
                error_sentences.append(sent + ["------------------------------------------------------------"])

        # Write output
        with open(output_path, "w", encoding="utf-8") as f:
            for sent in error_sentences:
                for line in sent:
                    f.write(line + "\n")

        print(f"Saved {len(error_sentences)} error sentences to {output_path}")


# Example usage
if __name__ == "__main__":
    parser_files = [
        "/Users/frapadovani/Desktop/CHILDES-Parser/parser/parser_predictions_Roberta_CDS_biaffine.conllu",
        "/Users/frapadovani/Desktop/CHILDES-Parser/parser/parser_predictions_Roberta_eng_biaffine.conllu",
        "/Users/frapadovani/Desktop/CHILDES-Parser/parser/parser_predictions_Stanza_off_the_shelf_old.conllu"
    ]
    extract_sentences_with_errors(parser_files, output_dir="/Users/frapadovani/Desktop/CHILDES-Parser/parser/error_analysis")



def analyze_label_errors(error_files, top_n=10):
    """
    Analyze label errors in parser error files.

    For each file:
    - Counts how many times each Gold label is predicted incorrectly.
    - Counts the frequency of each specific Gold->Pred label error.
    - Prints the top N most frequent errors.
    """
    for file_path in error_files:
        file_path = Path(file_path)
        gold_label_errors = defaultdict(int)  # counts of gold labels that are predicted incorrectly
        specific_errors = defaultdict(int)    # counts of (gold_label, pred_label) pairs

        with open(file_path, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f if line.strip()]

        # Iterate lines in pairs: Gold line followed by Pred line
        i = 0
        while i < len(lines):
            line = lines[i]
            if line.startswith("Gold: token="):
                gold_line = line
                pred_line = lines[i + 1] if i + 1 < len(lines) else ""
                
                # Extract head and label
                gold_match = re.search(r"label=([^\s]+)", gold_line)
                pred_match = re.search(r"label=([^\s]+)", pred_line)
                
                if gold_match and pred_match:
                    gold_label = gold_match.group(1)
                    pred_label = pred_match.group(1)
                    if gold_label != pred_label:
                        gold_label_errors[gold_label] += 1
                        specific_errors[(gold_label, pred_label)] += 1
                i += 2
            else:
                i += 1

        print(f"\n=== Analysis for {file_path.name} ===")
        print("\nGold labels predicted incorrectly (frequency):")
        for label, count in sorted(gold_label_errors.items(), key=lambda x: x[1], reverse=True):
            print(f"{label}: {count}")

        print(f"\nTop {top_n} most frequent specific Gold->Pred label errors:")
        for (gold, pred), count in Counter(specific_errors).most_common(top_n):
            print(f"{gold} -> {pred}: {count}")


# Example usage
if __name__ == "__main__":
    error_files = [
        "/Users/frapadovani/Desktop/CHILDES-Parser/parser/error_analysis/parser_predictions_Roberta_CDS_biaffine_errors.conllu",
        "/Users/frapadovani/Desktop/CHILDES-Parser/parser/error_analysis/parser_predictions_Roberta_eng_biaffine_errors.conllu",
        "/Users/frapadovani/Desktop/CHILDES-Parser/parser/error_analysis/parser_predictions_Stanza_off_the_shelf_old_errors.conllu"
    ]
    analyze_label_errors(error_files, top_n=10)


INPUT_FILE = Path(
    "/Users/frapadovani/Desktop/CHILDES-Parser/parser/error_analysis/"
    "parser_predictions_Roberta_eng_biaffine_errors.conllu"
)

OUTPUT_DIR = INPUT_FILE.parent / "fine_grained_errors_ENG_Roberta"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TOP_ERRORS = [
    ("nsubj", "root"),
    ("root", "nsubj"),
    ("conj", "discourse"),
    ("root", "discourse"),
    ("root", "compound"),
    ("mark", "advmod"),
    ("conj", "root"),
    ("parataxis","root"),
    ("advmod", "compound:prt"),
    ("discourse", "root"),
]

# Prepare output files
out_files = {}
for g, p in TOP_ERRORS:
    fname = f"gold_{g}__pred_{p}.conllu"
    out_files[(g, p)] = open(OUTPUT_DIR / fname, "w", encoding="utf-8")

def flush_sentence(sentence_lines, sentence_errors):
    """
    Write sentence to all error-specific files it belongs to
    """
    for err in sentence_errors:
        out_files[err].write("".join(sentence_lines))
        out_files[err].write("\n")

with open(INPUT_FILE, encoding="utf-8") as f:
    sentence_lines = []
    sentence_errors = set()
    last_gold = None

    for line in f:
        if line.startswith("# sentence:"):
            # New sentence begins — flush previous
            if sentence_lines and sentence_errors:
                flush_sentence(sentence_lines, sentence_errors)

            sentence_lines = [line]
            sentence_errors = set()
            last_gold = None
            continue

        sentence_lines.append(line)

        # Capture Gold line
        if line.startswith("Gold: token="):
            gold_match = re.search(r"label=([^\s]+)", line)
            if gold_match:
                last_gold = gold_match.group(1)

        # Capture Pred line (must follow Gold)
        elif line.startswith("Pred: token=") and last_gold is not None:
            pred_match = re.search(r"label=([^\s]+)", line)
            if pred_match:
                pred_label = pred_match.group(1)

                for g, p in TOP_ERRORS:
                    if last_gold == g and pred_label == p:
                        sentence_errors.add((g, p))

            last_gold = None

    # Flush final sentence
    if sentence_lines and sentence_errors:
        flush_sentence(sentence_lines, sentence_errors)

# Close files
for f in out_files.values():
    f.close()

print(f"Fine-grained error files written to: {OUTPUT_DIR}")
