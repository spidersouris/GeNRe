"""
Script used to extract member nouns in our collective nouns dictionary
from the French Wikipedia dataset: https://huggingface.co/datasets/graelo/wikipedia
"""

import csv
import os
import re
import nltk
import fr_core_news_md

from datasets import load_dataset
from utils import load_member_nouns, get_regex, get_member_nouns

MULT_N_PATTERN = r"<n-\d+><n-\d+>.*<\/n><\/n>"
WRITE_TO_FILE = True
OUTPUT_FILE = "wikipedia.csv"

print("Downloading NLTK punkt...")
nltk.download("punkt")
print("Successfully downloaded NLTK punkt")

member_nouns = load_member_nouns("../collective_nouns/collective_nouns.csv")
member_noun_regex = get_regex("../collective_nouns/collective_nouns.csv")
member_nouns_dict = {number: noun for number, noun in member_nouns}

count = 0

member_nouns_ids = []
member_nouns_text = []

print("Loading Wikipedia dataset")
wikipedia_fr = load_dataset("graelo/wikipedia", "20230601.fr")
print("Wikipedia dataset loaded!")

print("Loading spacy model...")
nlp = fr_core_news_md.load()

all_member_nouns = get_member_nouns("../collective_nouns/collective_nouns.csv")

def split_sentences_n(sent: str) -> set[str]:
    """
    Splits the input sentence into multiple sentences if it contains a member noun
    with several collective noun equivalents.
    (For instance, "soldats" has several collective
    noun equivalents: "armée", "troupe", "régiment", etc.)

    Args:
        sent (str): The input sentence to be split.

    Returns:
        set[str]: A set of sentences obtained after splitting the input sentence.
    """
    sentences = set()
    pattern = r"<n-\d+><n-\d+>.*<\/n><\/n>"
    matches = re.findall(pattern, sent)
    for match in matches:
        ids = [x.groups()[0] for x in re.finditer(r"<n-(\d+)>", match)]
        for id in ids:
            s = re.sub(r"<n-\d+><n-\d+>", "<n-" + id + ">", sent)
            s = re.sub(r"(<\/n>)+", "</n>", s)
            sentences.add(s)
    return sentences

def get_member_id(member_phrase: str) -> list[int]:
    """
    Retrieves the member ID associated with the given member noun in the member phrase.

    Args:
        member_phrase (str): The member phrase.

    Returns:
        list: A list of member IDs matching the given member noun.
    """
    member_noun = member_phrase.split()[1]
    return [k for k, v in member_nouns_dict.items() if v == member_noun]

def main():
    """
    Main function to extract member nouns from the French Wikipedia dataset.
    """
    if not os.path.exists(OUTPUT_FILE):
        print(f"{OUTPUT_FILE} does not exist, creating...")
        with open(OUTPUT_FILE, "w", newline="", encoding="utf8") as f:
            writer = csv.writer(f)
            writer.writerow(["id", "member_noun", "non_incl_sent"])

    for article in wikipedia_fr["train"]:
        for sent in nltk.sent_tokenize(article["text"], language="french"):
            f_sents = None
            ambiguous = False
            sent = sent.strip()
            pattern = member_noun_regex
            # remove previous <n> tags if any (for collective nouns with multiple member nouns)
            if all(x not in sent for x in ["\n", "  ", " ,", " ."]):
                if re.search(pattern, sent):
                    sent = re.sub(r"<n>", "", sent)
                    sent = re.sub(r"<n-\d+>", "", sent)
                    sent = re.sub(r"</n>", "", sent)

                    doc = nlp(sent)
                    for token in doc:
                        if token.text in all_member_nouns:
                            if token.pos_ != "NOUN":
                                ambiguous = True
                                print("Found ambigious sent", sent)
                                break

                    if not ambiguous:
                        for match in re.finditer(pattern, sent):
                            print(match, match.groups())
                            for i in range(len(match.groups())):
                                id = get_member_id(match.group(i).lower())

                                for j in id:
                                    sent = re.sub(fr"({match.group(i)})", fr"<n-{j}>\1</n>", sent)
                                    member_nouns_ids.append(j)
                                    member_nouns_text.append(match.group(i).lower().split()[1])

                                    if re.search(MULT_N_PATTERN, sent):
                                        f_sents = split_sentences_n(sent)

                        if WRITE_TO_FILE:
                            with open(OUTPUT_FILE, "a", newline="", encoding="utf8") as f:
                                writer = csv.writer(f)
                                if f_sents:
                                    for f_sent in f_sents:
                                        writer.writerow([f"{member_nouns_ids}",
                                                         f"{member_nouns_text}",
                                                         f"{f_sent}"])
                                else:
                                    writer.writerow([f"{member_nouns_ids}",
                                                     f"{member_nouns_text}",
                                                     f"{sent}"])


                        member_nouns_ids = []
                        member_nouns_text = []
                        count += 1

                        print(f"Processed {count} sentences")

if __name__ == "__main__":
    main()
