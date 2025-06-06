# GeNRe

GitHub Project for **GeNRe** (**Ge**nder-**N**eutral **Re**writer Using French Collective Nouns).

# Project Structure

- **/data**: contains the data of the project
    + **/collective_nouns**: contains the collective nouns dictionary
        * **/scripts**: contains Python scripts used to further populate the dictionary
            - **/output**: contains output of said scripts
    + **/dev**: contains development set of sentences used to build the RBS (rule-based system)
    + **/europarl**: contains Europarl corpus file used to extract sentences for fine-tuning and evaluation
    + **/eval**: contains evaluation data for both corpora (Wikipedia/Europarl) and for each component of the RBS (dependency detection/generation). See [Evaluation](#Evaluation)
    + **/ft**: contains train and valid sets used for fine-tuning
    + **/sentence_gathering**: contains Python scripts used to filter the Wikipedia and Europarl corpora, get sentences with member nouns in our dictionary and annotate them accordingly
        * **/output**: contains said sentences for both corpora
    + **/tests**: contains test sets for each component of the RBS (dependency detection/generation)
- **/instruction_models**: contains Python scripts used to communicate with Claude 3 Opus' and Mixtral 8x7B's APIs, for comparison with the RBS and fine-tuned models. Also contains a config.ini file for configuration purposes
- **/neural**: contains the IPYNB file used to fine-tune T5 and M2M100 models
- **/rbs**: contains Python scripts for the RBS (rule-based system) components
    + **/tests**: contains component testing Python scripts. See [Testing](#Testing)

# Usage

`pip install -r requirements.txt`

`python -m spacy download fr_core_news_sm`

## Rule-based system (RBS)

Run `python main.py` to instantiate the required spaCy and inflecteur models. From then, you can input as many sentences to convert as you want.

Before using the script, check the [Testing section](#Testing) to make sure everything works correctly in order to have the best usage experience.

### Testing

Run `python -m rbs.tests.tests` to execute a series of tests for the two components of the RBS: the dependency detection component and the generation component. Test data can be found in the **data/tests** folder.

If the default tests are failing, it probably means that there is something wrong with your spaCy installation. Please run `python -m spacy info` in your environment and make sure of the following:
- you ran all the commands in the [Usage](#Usage) section,
- your local spaCy version is equal to **3.7.4**,
- your local fr_core_news_sm spaCy pipeline version is equal to **3.7.0**.

If the issue persists, please [open an issue](https://github.com/spidersouris/GeNRe/issues) detailing your environment and the steps that you have taken so far.

## Fine-tuned models

### GeNRe-FT-T5

GeNRe-FT-T5 is made available on [HuggingFace](https://huggingface.co/spidersouris/genre-t5-small-60k).

For inference, use the following code:

```py
from transformers import T5ForConditionalGeneration, T5Tokenizer

model = T5ForConditionalGeneration.from_pretrained("spidersouris/genre-t5-small-60k")
tokenizer = T5Tokenizer.from_pretrained("spidersouris/genre-t5-small-60k")
```

### GeNRe-FT-M2M100

GeNRe-FT-M2M100 is made available on [HuggingFace](https://huggingface.co/spidersouris/genre-t5-small-60k).

For inference, use the following code:

```py
from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer

model = M2M100ForConditionalGeneration.from_pretrained("spidersouris/genre-m2m100_418M")
tokenizer = M2M100Tokenizer.from_pretrained("spidersouris/genre-m2m100_418M")
```

## Instruction models

`instruction_models/main.py` is nothing more than a Python wrapper around the Mistral and Claude APIs.

Prompts and few-shot examples can be modified in `instruction_models/utils.py`.

### Configuration

You can configure your model API keys as well as additional options by editing the included `config.ini` file.

The specified input file in `config.ini` should be a CSV file with one sentence per row (header excluded).

## Evaluation

Run `python eval.py` with arguments to run evaluation.

Usage: `python eval.py inp_file out_file [-h] [--component {gen,deps}] [--devset DEVSET] [--evalset EVALSET] [--print_error_types] [--autocol AUTOCOL] [--maxsents MAXSENTS] [--annotated] [--extract] [--corpus {wiki,eupr}] [--write_deps]`

### Arguments

Positional arguments correspond to the input file (inp_file) and the output file (out_file).

Optional arguments are listed below:
- -c, --component (str): The component to evaluate (either "gen" or "deps").
- -ds, --devset (str): The path to the development set file.
- -es, --evalset (str): The path to the evaluation set file.
- -p, --print_error_types: Print the error types.
- -ac, --autocol (int): The column number of the automatically generated sentences to evaluate. Default: 3.
- -ms, --maxsents (int): The maximum number of sentences to evaluate. Default: 250.
- -a, --annotated: Indicates whether the data is annotated with `<n>` tags.
- -e, --extract: Extract evaluation data from the input file.
- -cp, --corpus (str): The corpus type (either "wiki" or "eupr").
- -wd, --write_deps: Write syntactic dependencies to a file.

### Examples

Below are some examples of command usage to evaluate the two components ([dependency detection](#Dependency_Detection) and [generation](#Generation)).

#### Dependency Detection

Before evaluating dependency detection, you should make sure that you have a file with your dependencies. GeNRe has two of them: `data/eval/wiki/deps/wiki_deps_eval.csv` for the Wikipedia corpus, and `data/eval/eupr/deps/eupr_deps_eval.csv` for the Europarl corpus.

If you don't have one, you should run `eval.py` with the `-wd` flag like this. The input file should contain your input sentence (row 1). The output file is where your dependency file will be written.

`python eval.py -c "deps" -wd input.csv output.csv`

Once you have your file, here is an example of how to use the dependency detection evaluation system:

`python eval.py -c "deps" data/eval/eupr/deps/eupr_deps_eval.csv eupr_deps_out.csv`

Used as is, this command will generate the following files:
- **eupr_deps_out_detailed.csv**: CSV file containing the precision and recall for each sentence by model
- **eupr_deps_out_fscore.csv**: CSV file containing the average precision and recall, and computed F-score by model

#### Generation

Before evaluating dependency detection, you should make sure that you have a file with your sentences that's correctly formatted. See `data/eval/wiki/gen/wiki_gen_eval.csv` for an example.

Below is a basic example of how to use the generation evaluation system:

`python eval.py -c "gen" data/eval/wiki/gen/wiki_gen_eval.csv wiki_gen_out.csv`

This will generate a CSV file containing the baseline, golden (manual) and auto (RBS/LLM-generated) sentences, along with cosine similarity, WER and BLEU metric scores for baseline and auto sentences.

Use the `-ac [number]` flag to choose the column number that contains the "auto" sentences to be evaluated against golden sentences. Default column numbers are shown in the table below:

| ID |      Model      | Notes      |
|:--:|:---------------:|------------|
| 2  | RBS             |            |
| 3  | LLM-T5          |            |
| 4  | LLM-M2M100      |            |
| 5  | MIXTRAL-BASE    | *Shelved.* |
| 6  | MIXTRAL-CORR    | *Shelved.* |
| 7  | MIXTRAL-DICT    | *Shelved.* |
| 8  | MIXTRAL-BASE-FS | *Shelved.* |
| 9  | MIXTRAL-CORR-FS | *Shelved.* |
| 10 | MIXTRAL-DICT-FS | *Shelved.* |
| 11 | CLAUDE-BASE-FS  |            |
| 12 | CLAUDE-DICT-FS  |            |
| 13 | CLAUDE-CORR-FS  |            |

For example, the command below will evaluate sentences generated by CLAUDE-CORR-FS in the `wiki_gen_eval.csv` file. Results will both be printed and saved to `wiki_genèeval_out_13.csv`:

`python eval.py -c "gen" -ac 13 data/eval/wiki/gen/wiki_gen_eval.csv wiki_gen_eval_out_13.csv`

In addition, you can also use the `-p` flag to print error types for the selected auto sentences (annotated in columns with IDs 15, 16, 17 for RBS, T5 and M2M100, respectively).