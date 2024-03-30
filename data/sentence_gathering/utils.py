import csv
import re

def load_member_nouns(file_path):
    member_nouns = []
    with open(file_path, "rt", encoding="utf8") as f:
        reader = csv.reader(f, delimiter=",")
        next(reader)
        for row in reader:
            member_nouns.append((row[0], row[3]))
    return member_nouns

def get_regex(file_path):
    nouns = []
    mb_nouns_rx = ""
    with open(file_path, "r", encoding="utf8") as cn_file:
        reader = csv.reader(cn_file)
        next(reader)
        for row in reader:
            nouns.append(row[3])

    for i, noun in enumerate(nouns):
        mb_nouns_rx += re.escape(noun)
        if i < len(nouns) - 1:
            mb_nouns_rx += "|"

    return r"\b((?:[Aa]ux|[LlddCcSsMm]es)\s(?:" + mb_nouns_rx + r"))(?![\w-])\b"

def get_member_nouns(file_path):
    nouns = set()

    with open(file_path, "r", encoding="utf8") as cn_file:
        reader = csv.reader(cn_file)
        next(reader)
        for row in reader:
            nouns.add(row[3])

    return nouns