import os
import re
from collections import defaultdict

# ==== CONFIG ====
input_file = "/Users/frapadovani/Desktop/CHILDES-Parser/parser/error_analysis/supar_eng_roberta_errors.conllu"

# Define the label transitions you want
target_errors = {
    ("advmod", "discourse"),
    ("nsubj", "root"),
    ("nsubj", "vocative"),
}

# Output directory (created automatically)
output_dir = os.path.join(os.path.dirname(input_file), "filtered_errors")
os.makedirs(output_dir, exist_ok=True)

# =================

def extract_label(line):
    """Extract label from Gold/Pred line."""
    match = re.search(r"label=([^\s]+)", line)
    return match.group(1) if match else None


with open(input_file, "r", encoding="utf-8") as f:
    content = f.read()

# Split by sentence blocks
blocks = content.split("------------------------------------------------------------")

# Dictionary to collect sentences per error type
collected = defaultdict(list)

for block in blocks:
    lines = block.strip().split("\n")
    if not lines:
        continue

    sent_id_line = None
    gold_lines = []
    pred_lines = []

    for line in lines:
        if line.startswith("# sent_id"):
            sent_id_line = line
        elif line.startswith("Gold:"):
            gold_lines.append(line)
        elif line.startswith("Pred:"):
            pred_lines.append(line)

    # Pair Gold and Pred lines
    for gold, pred in zip(gold_lines, pred_lines):
        gold_label = extract_label(gold)
        pred_label = extract_label(pred)

        if (gold_label, pred_label) in target_errors:
            key = f"{gold_label}_TO_{pred_label}"
            collected[key].append(block.strip())
            break  # Save sentence only once


# Write separate files
for error_type, sentences in collected.items():
    output_path = os.path.join(output_dir, f"{error_type}.conllu")
    with open(output_path, "w", encoding="utf-8") as out:
        for sent in sentences:
            out.write(sent + "\n")
            out.write("------------------------------------------------------------\n")

print("Done. Files created in:", output_dir)
