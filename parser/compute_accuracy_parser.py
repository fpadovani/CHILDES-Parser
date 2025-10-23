import stanza
from supar import Parser
from conllu import parse_incr
from collections import defaultdict
from pathlib import Path
from tqdm import tqdm
import re

def _normalize_form(s):
    """Normalize surface forms for loose matching (lowercase, unify quotes)."""
    if s is None:
        return ""
    s = s.replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"')
    s = re.sub(r"\s+", " ", s)
    return s.strip().lower()


def load_conllu_gold(path):
    """
    Load gold annotations from a CONLLU file.
    Returns list of sentences, each as a list of dicts:
    {
        'form': str, 
        'upos': str, 
        'xpos': str, 
        'head': int, 
        'deprel': str
    }
    """
    sentences = []
    with open(path, "r", encoding="utf-8") as f:
        for tokenlist in parse_incr(f):
            tokens = []
            for token in tokenlist:
                if isinstance(token["id"], tuple):  # skip multi-word tokens like "2-3"
                    continue
                tokens.append({
                    "form": token["form"],
                    "upos": token["upostag"],
                    "xpos": token["xpostag"],
                    "head": token["head"],
                    "deprel": token["deprel"]
                })
            if tokens:
                sentences.append(tokens)
    return sentences


def evaluate_parsers_overall(test_path, nlp_stanza_childes, log_predictions=False, log_path="parser_predictions"):
    results = {}

    parsers = {
        "Supar_Childes_biaffine": Parser.load('/Users/frapadovani/Desktop/CHILDES-Parser/parser/biaffine_childes/b_childes_new'),
        "Supar_Childes_crf": Parser.load('/Users/frapadovani/Desktop/CHILDES-Parser/parser/crf_childes/model_crf_childes'),
        "Supar_Combined_eng_biaffine": Parser.load('/Users/frapadovani/Desktop/CHILDES-Parser/parser/biaffine_combined/biaffine_combined'),
        "Supar_Combined_eng_crf": Parser.load('/Users/frapadovani/Desktop/CHILDES-Parser/parser/crf_combined/crf_combined'),
        "Supar_finetuned_onlygolden_crf": Parser.load('/Users/frapadovani/Desktop/CHILDES-Parser/parser/crf_combined_finetuned_only_golden/crf_combined_finetuned'),
        "Supar_finetuned_complete_biaffine": Parser.load('/Users/frapadovani/Desktop/CHILDES-Parser/parser/biaffine_combined_finetuned/model_biaffine_combined_finetuned'),
        #"Supar_finetuned_complete_crf": Parser.load('/Users/frapadovani/Desktop/stanza/parser/crf_combined_finetuned/model_crf_combined_finetuned'),
        #"Stanza_off_the_shelf": stanza.Pipeline(lang='en', processors='tokenize,pos,lemma,depparse', use_gpu=True),
        #"Stanza_Custom": nlp_stanza_childes,

    }

    gold_sentences = load_conllu_gold(test_path)

    log_file_handles = {}
    if log_predictions:
        base = Path(log_path).stem
        out_dir = Path(log_path).parent if Path(log_path).parent.exists() else Path(".")
        out_dir.mkdir(parents=True, exist_ok=True)

    for parser_name, nlp in parsers.items():
        if log_predictions:
            safe_name = re.sub(r"[^0-9A-Za-z_\-]", "_", parser_name)
            log_fp = out_dir / f"{base}_{safe_name}.conllu"
            lf = open(log_fp, "w", encoding="utf-8")
            log_file_handles[parser_name] = lf
            lf.write(f"# Parser: {parser_name}\n")

        correct_upos = 0
        correct_xpos = 0
        correct_heads = 0
        correct_labels = 0
        total_tokens = 0

        if log_predictions:
            lf.write("\n# === Start sentences ===\n\n")

        for gold_tokens in tqdm(gold_sentences, desc=f"Parsing with {parser_name}"):
            gold_text = " ".join(t["form"] for t in gold_tokens)

            if parser_name in ["Stanza_Custom", "Stanza_off_the_shelf"]:
                doc = nlp(gold_text)
                pred_tokens = []
                for sent in doc.sentences:
                    for word in sent.words:
                        pred_tokens.append({
                            "form": word.text,
                            "upos": word.upos,
                            "xpos": word.xpos,
                            "head": int(word.head) if getattr(word, "head", None) is not None else 0,
                            "deprel": word.deprel
                        })
            elif parser_name in ["Supar_Childes_biaffine", "Supar_Childes_crf", "Supar_Combined_eng_biaffine", "Supar_Combined_eng_crf", "Supar_finetuned_biaffine", "Supar_finetuned_crf"]:
                pred_tokens = []
          
                dataset = nlp.predict(gold_text, lang= 'en', prob = False, verbose = False)
                sent = dataset[0]
                for word, arc, rel in zip(sent.words, sent.arcs, sent.rels):  # iterate tokens
                    pred_tokens.append({
                        "form": word,
                        "upos": '',
                        "xpos": '',
                        "head": arc,
                        "deprel": rel
                    })
            else:
                doc = nlp(gold_text)
                pred_tokens = []
                for token in doc:
                    head = 0 if token.head.i == token.i else token.head.i + 1
                    pred_tokens.append({
                        "form": token.text,
                        "upos": token.pos_,
                        "xpos": token.tag_,
                        "head": head,
                        "deprel": token.dep_
                    })

            if log_predictions:
                lf.write(f"# sentence: {gold_text}\n")

            pairs = []
            if len(gold_tokens) == len(pred_tokens):
                pairs = list(zip(gold_tokens, pred_tokens))

            else:
                ### THIS PART IS ONLY FOR SPACY MODELS ###
                pred_idx = 0
                for g in gold_tokens:
                    g_norm = _normalize_form(g["form"])
                    found = None
                    for j in range(pred_idx, min(pred_idx + 6, len(pred_tokens))):
                        if _normalize_form(pred_tokens[j]["form"]) == g_norm:
                            found = j
                            break
                    if found is not None:
                        pairs.append((g, pred_tokens[found]))
                        pred_idx = found + 1
                    else:
                        if pred_idx < len(pred_tokens):
                            pairs.append((g, pred_tokens[pred_idx]))
                            pred_idx += 1
                        else:
                            pairs.append((g, None))

            for g, p in pairs:
                total_tokens += 1
                gold_upos = g.get("upos", "")
                gold_xpos = g.get("xpos", "")
                gold_head = int(g.get("head", 0))
                gold_label = g.get("deprel", "")

                if p is None:
                    if log_predictions:
                        lf.write(f"\nGold: {g}\nPred: <NO PREDICTION>\n")
                    continue

                pred_upos = p.get("upos", "")
                pred_xpos = p.get("xpos", "")
                pred_head = int(p.get("head", 0))
                pred_label = p.get("deprel", "")

                if gold_upos == 'INTJ' and gold_label in ['discourse', 'reparandum']:
                    gold_label = 'intj'

                if gold_label == 'acl:relcl':
                    gold_label = 'relcl'

                if pred_label == 'acl:relcl':
                    pred_label = 'relcl'
                
                if gold_label == 'compound:prt':
                    gold_label = 'prt'

                if pred_label == 'compound:prt':
                    pred_label = 'prt' 
                
                if gold_label == 'cc:preconj':
                    gold_label = 'preconj'

                if pred_label == 'cc:preconj':
                    pred_label = 'preconj' 
                
                if gold_label == 'nmod:poss':
                    gold_label = 'poss'

                if pred_label == 'nmod:poss':
                    pred_label = 'poss' 
                
                if gold_label == 'obl:npmod':
                    gold_label = 'npadvmod'

                if pred_label == 'obl:npmod':
                    pred_label = 'npadvmod'

                if pred_label in ['dobj', 'pobj'] and gold_label == 'obj':
                    pred_label = 'obj'

                if pred_label == 'neg':
                    pred_label = 'advmod'

                if pred_label == 'ROOT':
                    pred_label = 'root'
                
                # UPOS
                if pred_upos and pred_upos.lower() == gold_upos.lower():
                    correct_upos += 1

                # XPOS
                if pred_xpos and pred_xpos == gold_xpos:
                    correct_xpos += 1

                # Heads
                if pred_head == gold_head:
                    correct_heads += 1
                
                
                if pred_head == gold_head and pred_label == gold_label:
                    correct_labels += 1

                if log_predictions:                    lf.write(
                        f"\nGold: token={g['form']}, upos={gold_upos}, xpos={gold_xpos}, head={gold_head}, label={gold_label}\n"
                        f"Pred: token={p['form']}, upos={pred_upos}, xpos={pred_xpos}, head={pred_head}, label={pred_label}\n"
                    )
            if log_predictions:
                lf.write("\n" + ("-" * 60) + "\n\n")

        if log_predictions:
            lf.close()

        results[parser_name] = {
            "UPOS Accuracy": correct_upos / total_tokens if total_tokens > 0 else 0.0,
            "XPOS Accuracy": correct_xpos / total_tokens if total_tokens > 0 else 0.0,
            "UAS": correct_heads / total_tokens if total_tokens > 0 else 0.0,
            "LAS": correct_labels / total_tokens if total_tokens > 0 else 0.0,
            "Total Tokens": total_tokens
        }

    return results

test_path = "UD_English-CHILDES/en_childes-ud-test.conllu"
pos_tagger_model = "./saved_models/pos/en_childes_charlm_tagger.pt"
parser_model = "./saved_models/depparse/en_childes_charlm_parser.pt"

nlp_childes = stanza.Pipeline(
        lang='en',
        processors='tokenize,pos,lemma,depparse',
        use_gpu=True,
        pos_model_path=pos_tagger_model,
        depparse_model_path=parser_model
    )

metrics = evaluate_parsers_overall(test_path, nlp_childes, log_predictions=True, log_path="parser_predictions.txt")


for parser, scores in metrics.items():
    print(f"\n=== {parser} ===")
    for metric, value in scores.items():
        if isinstance(value, (int, float)):
            print(f"{metric}: {value:.4f}")
        else:
            print(f"{metric}: {value}")