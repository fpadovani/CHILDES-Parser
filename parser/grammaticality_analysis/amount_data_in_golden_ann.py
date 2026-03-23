from pathlib import Path
import pandas as pd
import re

def normalize_sentence(sent):
    sent = sent.lower()
    sent = sent.replace("’", "'")
    sent = re.sub(r"[^\w\s']", "", sent)   # remove punctuation except apostrophe
    sent = re.sub(r"\s+", " ", sent)
    return sent.strip()


def read_conllu_sentences(conllu_path):
    sentences = []

    with open(conllu_path, "r", encoding="utf-8") as f:
        block = []
        tokens = []

        for line in f:
            line = line.rstrip("\n")

            if line == "":
                if tokens:
                    sent_text = " ".join(tokens)
                    norm_text = normalize_sentence(sent_text)

                    sentences.append({
                        "text": sent_text,
                        "norm": norm_text,
                        "block": "\n".join(block) + "\n"
                    })

                block = []
                tokens = []
                continue

            block.append(line)

            if not line.startswith("#"):
                cols = line.split("\t")
                if "-" not in cols[0] and "." not in cols[0]:
                    tokens.append(cols[1])

    return sentences


manual_path = "/Users/frapadovani/Desktop/CHILDES-Parser/parser/grammaticality_analysis/data/manually_annotated_full.csv"

manual_df = pd.read_csv(manual_path)

manual_df["norm"] = manual_df["transcript_clean"].astype(str).apply(normalize_sentence)

manual_set = set(manual_df["norm"])



train_path = "/Users/frapadovani/Desktop/CHILDES-Parser/UD_English-CHILDES/en_childes-ud-train.conllu"
dev_path   = "/Users/frapadovani/Desktop/CHILDES-Parser/UD_English-CHILDES/en_childes-ud-dev.conllu"
test_path  = "/Users/frapadovani/Desktop/CHILDES-Parser/UD_English-CHILDES/en_childes-ud-test.conllu"

train_sents = read_conllu_sentences(train_path)
dev_sents   = read_conllu_sentences(dev_path)
test_sents  = read_conllu_sentences(test_path)

all_gold = (
    [("train", s) for s in train_sents] +
    [("dev", s)   for s in dev_sents] +
    [("test", s)  for s in test_sents]
)


matched = []

for split, sent in all_gold:
    if sent["norm"] in manual_set:
        matched.append({
            "split": split,
            "text": sent["text"],
            "block": sent["block"]
        })


output_path = "/Users/frapadovani/Desktop/CHILDES-Parser/parser/grammaticality_analysis/matched_manual_sentences.conllu"

with open(output_path, "w", encoding="utf-8") as f:
    for item in matched:
        f.write(f"# source_split = {item['split']}\n")
        f.write(item["block"])
        f.write("\n")


matched_df = pd.DataFrame(matched)

print("Total matched sentences:", len(matched_df))
print(matched_df["split"].value_counts())
