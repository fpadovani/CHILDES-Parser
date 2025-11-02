# CHILDES-Parser
Training of a Dependency Parser on golden UD annotation on English CHILDES 

## STANZA 

1. Test the accuracy of a Stanza model trained on gold UD_English-CHILDES annotations taken from the dev branch of [Xiulin's paper repository](https://github.com/UniversalDependencies/UD_English-CHILDES), which I added as well in this repo.

### Instruction for the training of Stanza model [documentation](https://stanfordnlp.github.io/stanza/training_and_evaluation.html)
- First, clone the stanza repository,  create a virtual environment, and install all the dependencies. In ```script/config.sh```, please change the path of ```UD_BASE```.

```bash
git clone https://github.com/stanfordnlp/stanza.git
cd stanza
conda create -n parser python==3.10
conda activate parser
pip install -e .
cd script
source config.sh
```
- Then create a folder called **data** and move the UD-data-repository to **data** and prepare the data for different tasks.
```
# for tokenizer
# E.g., python -m stanza.utils.datasets.prepare_tokenizer_treebank UD_English-CHILDES
python -m stanza.utils.datasets.prepare_tokenizer_treebank ${corpus_name}

# for dependency parser
python -m stanza.utils.datasets.prepare_depparse_treebank ${corpus_name}

# for pos tagger
python -m stanza.utils.datasets.prepare_pos_treebank ${corpus_name}
```
- Start training!  
```bash
# To train the tokenizer you run this command:
python -m stanza.utils.training.run_tokenize ${corpus_name}

# To train the parser you run this command:
python -m stanza.utils.training.run_depparse ${corpus_name}
```
- Create also the output folder for your trained models **save_models**.

- In order to load the trained model and evaluate it, I use this:

```
stanza_childes_en = stanza.Pipeline(
        lang='en',
        processors='tokenize,pos,depparse',
        use_gpu=True,
        pos_model_path='./saved_models/pos/en_childes_charlm_tagger.pt',
        depparse_model_path='./saved_models/depparse/en_childes_charlm_parser.pt')
```


2. Test the accuracy of Stanza Dependency Parser off the shelf trained on combined UD. 

`stanza_en = stanza.Pipeline(lang='en', processors='tokenize,pos,lemma,depparse', use_gpu=True)`


## SUPAR 
The folder **parser** contains the code to train Supar parsers from the original repository + additional folders and notebooks that I added myself.

In *conllu_files_original* one can find the .conllu files of the different annotated resources [GUM, EWT, GumReddit, Pronouns, PUD] For the first three, the train/dev/test splits exixt, instead for Pronouns and PUD it's available only a test set. I use the code in  *merge_conllu_files.ipynb* to generate splits also for the two files (with a proportion of 80/10/10) and also to finally combine the data in a unique train/dev/test triplets, that you can find in *conllu_files_merged*. I then use these files as input for the training of the parser.

3. Test a Supar model trained on Combined English Dataset (to be compared with the Stanza off-the-shelf)

I try two models (crf and biaffine):

```
python3 -u -m supar.cmds.dep.crf2o train -b -d 0 -c dep-crf2o-en -p ./parser/model_crf_combined -f char  \
    --train ./conllu_files_merged/combined_train.conllu  \
    --dev ./conllu_files_merged/combined_dev.conllu  \
    --test ./conllu_files_merged/combined_test.conllu  \
    --embed glove-6b-100  \
    --mbr  \
    --proj
```
The trained model is saved in the **parser/crf_combined** folder.

```
python -u -m supar.cmds.dep.biaffine train -b -d 0 -c dep-biaffine-en -p 
    ./parser/biaffine_combined -f char  \
    --train ./UD_English-CHILDES/en_childes-ud-train.conllu \
    --dev ./UD_English-CHILDES/en_childes-ud-dev.conllu \
    --test ./UD_English-CHILDES/en_childes-ud-test.conllu  \
    --embed glove-6b-100  
```
The trained model is saved in the **parser/biaffine_combined** folder.


4. Test a Supar model trained on first Combined English + CHILDES(silver) + CHILDES(gold)

I take the trained model on teh Combined set of English UD and I use it as the input for the next phase of training.
I train on the CHILDES silver datasets (the original silver .connlu files can be found in the **UD_CHILDES_silver** folder); I generate the train/dev/test split, the code can be found in the */parser/merge_conllu_files.ipynb* notebook and the merged files in the **parser/silver_files_merged** folder.
I use the *--checkpoint* argument while running the command from the terminal. As indicated by the code snipped below:

```
python3 -u -m supar.cmds.dep.biaffine train -b -d 0 -c dep-biaffine-en -p ./parser/biaffine_combined/model_biaffine_combined -f char --checkpoint \
    --train ./parser/silver_files_merged/childes_silver_train_sampled.conllu \
    --dev ./parser/silver_files_merged/childes_silver_dev_sampled.conllu  \
    --test ./parser/silver_files_merged/childes_silver_test_sampled.conllu  \
    --embed glove-6b-100 

```

```
python3 -u -m supar.cmds.dep.crf2o train -b -d 0 -c dep-crf2o-en -p  -f char ./parser/crf_combined_finetuned_complete --checkpoint\
    --train ./parser/silver_files_merged/small_train.conllu  \
    --dev ./parser/conllu_files_merged/small_dev.conllu  \
    --test ./parser/conllu_files_merged/combined_test.conllu  \
    --embed glove-6b-100  \
    --mbr  \
    --proj
```


5. Test a Supar model trained on CHILDES (gold)

I try two models (crf and biaffine):

```
python3 -u -m supar.cmds.dep.crf2o train -b -d 0 -c dep-crf2o-en -p ./parser/crf_childes -f char  \
    --train UD_English-CHILDES/en_childes-ud-train.conllu  \
    --dev UD_English-CHILDES/en_childes-ud-dev.conllu  \
    --test UD_English-CHILDES/en_childes-ud-test.conllu  \
    --embed glove-6b-100  \
    --mbr  \
    --proj
```

```
python3 -u -m supar.cmds.dep.biaffine train -b -d 0 -c dep-biaffine-en \
-p ./parser/biaffine_childes -f char  \
    --train UD_English-CHILDES/en_childes-ud-train.conllu  \
    --dev UD_English-CHILDES/en_childes-ud-dev.conllu  \
    --test UD_English-CHILDES/en_childes-ud-test.conllu  \
    --embed glove-6b-100   
```



To evaluate the trained model you can use this command:

- biaffine
`python -u -m supar.cmds.dep.biaffine evaluate -d 0 -p model_path --data test_file.connlu --tree  --proj`
- crf 
`python -u -m supar.cmds.dep.crf2o evaluate -d 0 -p model_path --data test_file.connlu --mbr --tree --proj`


## CONSTRUCTION CLASSIFICATION 
based on the paper Do Construction Distributions Shape Formal Language Learning In German BabyLMs? Bunzeck et al., 2025

There is a function I developed (very first attempt) in the Notebook *parser/construction_distribution.ipynb*




