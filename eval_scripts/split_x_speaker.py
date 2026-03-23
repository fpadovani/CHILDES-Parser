import os

def split_by_speaker(input_path):
    # Determine output file names
    base_dir = os.path.dirname(input_path)
    base_name = os.path.basename(input_path).replace(".conllu", "")
    
    child_path = os.path.join(base_dir, f"{base_name}_childes.conllu")
    adult_path = os.path.join(base_dir, f"{base_name}_adults.conllu")
    
    with open(input_path, "r", encoding="utf-8") as f:
        content = f.read().strip()
    
    sentences = content.split("\n\n")
    
    child_sentences = []
    adult_sentences = []
    
    for sent in sentences:
        if "# speaker_role = Target_Child" in sent:
            child_sentences.append(sent)
        else:
            adult_sentences.append(sent)
    
    with open(child_path, "w", encoding="utf-8") as f:
        f.write("\n\n".join(child_sentences) + "\n\n")
    
    with open(adult_path, "w", encoding="utf-8") as f:
        f.write("\n\n".join(adult_sentences) + "\n\n")
    
    print(f"Created:\n  {child_path}\n  {adult_path}")


# Paths to your files
#dev_file = "/Users/frapadovani/Desktop/CHILDES-Parser/UD_English-CHILDES/en_childes-ud-dev.conllu"
test_file = "/Users/frapadovani/Desktop/CHILDES-Parser/eval_scripts/prediction_files_integral/supar_childes_roberta.conllu"

#split_by_speaker(dev_file)
split_by_speaker(test_file)
