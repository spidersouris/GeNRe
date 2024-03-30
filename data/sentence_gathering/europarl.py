import csv
import os
import re
import time
import nltk
import spacy
import fr_core_news_md
from utils import load_member_nouns, get_regex, get_member_nouns

print("Downloading NLTK punkt...")
nltk.download("punkt")
print("Successfully downloaded NLTK punkt")

member_nouns = load_member_nouns("../collective_nouns/collective_nouns.csv")
member_noun_regex = get_regex("../collective_nouns/collective_nouns.csv")
member_nouns_dict = {number: noun for number, noun in member_nouns}
mult_N_pattern = r"<n-\d+><n-\d+>.*<\/n><\/n>"
write_to_file = True
count = 0

member_nouns_ids = []
member_nouns_text = []

OUTPUT_FILE = "europarl.csv"
EUROPARL_CORPUS = "../europarl/europarl-v7.fr-en.fr"

print("Loading spacy model...")
nlp = fr_core_news_md.load()

start_time = time.time()

all_member_nouns = get_member_nouns("../collective_nouns/collective_nouns.csv")

def split_sentences_n(sent):
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

def get_member_id(member_phrase):
    member_noun = member_phrase.split()[1]
    return [k for k, v in member_nouns_dict.items() if v == member_noun]

if not os.path.exists(OUTPUT_FILE):
    print(f"{OUTPUT_FILE} does not exist, creating...")
    with open(OUTPUT_FILE, "w", newline="", encoding="utf8") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "member_noun", "non_incl_sent"])

with open(EUROPARL_CORPUS, "r", encoding="utf8") as f:
    sents = f.readlines()
    for sent in sents:
        f_sents = None
        ambiguous = False
        sent = sent.strip()
        pattern = member_noun_regex
        # remove previous <n> tags if any (for collective nouns with multiple member nouns)
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

                            if re.search(mult_N_pattern, sent):
                                f_sents = split_sentences_n(sent)

                if write_to_file:
                    with open(OUTPUT_FILE, "a", newline="", encoding="utf8") as f:
                        writer = csv.writer(f)
                        if f_sents:
                            for f_sent in f_sents:
                                writer.writerow([f"{member_nouns_ids}", f"{member_nouns_text}", f"{f_sent}"])
                        else:
                            writer.writerow([f"{member_nouns_ids}", f"{member_nouns_text}", f"{sent}"])


                member_nouns_ids = []
                member_nouns_text = []
                count += 1

                print(f"Processed {count} sentences")
