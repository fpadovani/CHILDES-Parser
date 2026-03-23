import stanza
from stanza.utils.conll import CoNLL
from stanza.utils.conll18_ud_eval import load_conllu_file, evaluate


# Load pretrained English pipeline
nlp = stanza.Pipeline(
    lang="en",
    processors="tokenize,pos,lemma,depparse",
    tokenize_pretokenized=True,
    pos_model_path="/Users/frapadovani/Desktop/CHILDES-Parser/saved_models/pos/en_childes_charlm_tagger.pt",
    depparse_model_path="/Users/frapadovani/Desktop/CHILDES-Parser/saved_models/depparse/cds_charlm/charlm_cds_nopretrain_charlmpos_charlm_parser.pt"
)


# Load UD-CHILDES test set
doc = CoNLL.conll2doc("/Users/frapadovani/Desktop/CHILDES-Parser/UD_English-CHILDES/en_childes-ud-test_childes.conllu")

doc = nlp(doc)

# Save predictions
CoNLL.write_doc2conll(doc, "/Users/frapadovani/Desktop/CHILDES-Parser/eval_scripts/prediction_files_by_speaker/stanza_charlm_childes.conllu")

gold_ud = load_conllu_file("/Users/frapadovani/Desktop/CHILDES-Parser/UD_English-CHILDES/en_childes-ud-test.conllu")
pred_ud = load_conllu_file("/Users/frapadovani/Desktop/CHILDES-Parser/eval_scripts/prediction_files_integral/from_server/roberta_large_no_finetuned_test.conllu")
scores = evaluate(gold_ud, pred_ud)

print("UAS: {:.2f}".format(100*scores["UAS"].f1))
print("LAS: {:.2f}".format(100*scores["LAS"].f1))
print("MLAS: {:.2f}".format(100*scores["MLAS"].f1))
print("BLEX: {:.2f}".format(100*scores["BLEX"].f1))

# Print total words and sentences for table
print("Total words in gold:", scores["Words"].gold_total)
print("Total words in system:", scores["Words"].system_total)
print("Total sentences in gold:", scores["Sentences"].gold_total)
print("Total sentences in system:", scores["Sentences"].system_total)


