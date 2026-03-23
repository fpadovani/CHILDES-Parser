import stanza
from stanza.utils.conll import CoNLL
from stanza.models.pos import scorer

# Load POS-only pipeline
nlp = stanza.Pipeline(
    lang="en",
    processors="tokenize,pos,lemma,depparse",
    tokenize_pretokenized=True,
    pos_model_path="/Users/frapadovani/Desktop/CHILDES-Parser/saved_models/pos/en_childes_transformer_tagger.pt"
)


pred_file = "stanza_pos_childes_charlm_dev.pred.conllu"
# Load gold test set
gold_doc = CoNLL.conll2doc(
    "/Users/frapadovani/Desktop/CHILDES-Parser/UD_English-CHILDES/en_childes-ud-dev.conllu"
)
print(f"Number of sentences in dev set: {len(gold_doc.sentences)}")

# Run POS tagging
pred_doc = nlp(gold_doc)

# Save predictions to file
CoNLL.write_doc2conll(pred_doc, pred_file)

# Evaluate (FILE PATHS, not Documents!)
results = scorer.score(pred_file, '/Users/frapadovani/Desktop/CHILDES-Parser/UD_English-CHILDES/en_childes-ud-dev.conllu')
print(results)