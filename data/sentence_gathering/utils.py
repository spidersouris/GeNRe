"""
Utility functions used for corpus sentence extraction.
"""

import csv
import re

def load_member_nouns(file_path: str) -> list[tuple[str, str]]:
    """
    Loads member nouns from the CSV dictionary.

    Args:
        file_path (str): The path to the CSV dictionary.

    Returns:
        list[tuple[str, str]]: A list of tuples containing the member nouns, where each tuple
              is composed of the ID of the member noun, and the member noun itself (plural form).
    """
    member_nouns = []
    with open(file_path, "rt", encoding="utf8") as f:
        reader = csv.reader(f, delimiter=",")
        next(reader)
        for row in reader:
            member_nouns.append((row[0], row[3]))
    return member_nouns

def get_regex(file_path: str) -> str:
    """
    Generates a regular expression pattern based on the member nouns
    extracted from the CSV dictionary, in order to catch member phrases
    in the corpora sentences.

    Args:
        file_path (str): The path to the CSV dictionary.

    Returns:
        str: The generated regular expression pattern.
    """
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
    """
    Retrieves a set of plural member nouns from the CSV dictionary.

    Args:
        file_path (str): The path to the CSV dictionary.

    Returns:
        set: A set containing the plural member nouns.
    """
    nouns = set()

    with open(file_path, "r", encoding="utf8") as cn_file:
        reader = csv.reader(cn_file)
        next(reader)
        for row in reader:
            nouns.add(row[3])

    return nouns
