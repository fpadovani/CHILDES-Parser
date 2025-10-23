input_file = "/Users/frapadovani/Desktop/stanza/UD_English-CHILDES/en_childes-ud-test.conllu"
output_file = "/Users/frapadovani/Desktop/stanza/UD_English-CHILDES/en_childes-ud-test.txt"

with open(input_file, "r", encoding="utf-8") as f_in, \
     open(output_file, "w", encoding="utf-8") as f_out:
    for line in f_in:
        line = line.strip()
        if line.startswith("# text ="):
            sentence = line[len("# text ="):].strip()
            f_out.write(sentence + "\n")

