import stanza
import spacy
from supar import Parser
from conllu import parse_incr
from collections import defaultdict
from pathlib import Path
from tqdm import tqdm
import re
import pandas as pd

def parse_and_save_conllu(data_file, output_folder, parsers):

    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(data_file)

    categories = {
        "grammatical": df[df['is_grammatical'] == 1]['transcript_clean'].tolist(),
        "ambiguous": df[df['is_grammatical'] == 0]['transcript_clean'].tolist(),
        "ungrammatical": df[df['is_grammatical'] == -1]['transcript_clean'].tolist()
    }

    for parser_name, nlp in parsers.items():
        for cat_name, sentences in categories.items():
            conllu_lines = []

            for sent_id, sent_text in enumerate(sentences, start=1):

                # --- SuPar/Biaffine parsers ---
                if parser_name in ["Roberta_CDS_biaffine", "Roberta_eng_biaffine"]:
                    dataset = nlp.predict([sent_text], lang='en', prob=True, verbose=False)
                    sent = dataset[0]

                    for word, arc, rel in zip(sent.words, sent.arcs, sent.rels):
                        head = arc
                        conllu_lines.append(
                            f"{word}\t_\t_\t_\t_\t_\t{head}\t{rel}\t_\t_"
                        )

                # --- Stanza parser ---
                elif parser_name == "Stanza_off_the_shelf":
                    doc = nlp(sent_text)
                    for sent in doc.sentences:
                        for token in sent.tokens:
                            for word in token.words:
                                head = 0 if word.head == word.id else word.head
                                conllu_lines.append(
                                    f"{word.text}\t_\t{word.upos}\t{word.xpos}\t_\t_\t{head}\t{word.deprel}\t_\t_"
                                )

                # 🔹 ADD EOS MARKER AFTER EACH SENTENCE
                conllu_lines.append("##")

            out_file = output_folder / f"{cat_name}_{parser_name}.conllu"
            with open(out_file, 'w', encoding='utf-8') as f:
                for line in conllu_lines:
                    f.write(line + "\n")

            print(f"Saved output to {out_file}")

# --- Example usage ---
parsers = {
    "Roberta_CDS_biaffine": Parser.load('/Users/frapadovani/Desktop/CHILDES-Parser/parser/biaffine_roberta_large_childes_10/brlc'),
    "Roberta_eng_biaffine": Parser.load('/Users/frapadovani/Desktop/CHILDES-Parser/parser/biaffine_roberta_large_eng_10/brlcomb'),
    "Stanza_off_the_shelf": stanza.Pipeline(lang='en', processors='tokenize,pos,lemma,depparse', use_gpu=True)
}

data_file = "/Users/frapadovani/Desktop/CHILDES-Parser/parser/grammaticality_analysis/data/manually_annotated_full.csv"
output_folder = "/Users/frapadovani/Desktop/CHILDES-Parser/parser/grammaticality_analysis/data/conllu_outputs"

parse_and_save_conllu(data_file, output_folder, parsers)