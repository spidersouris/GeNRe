"""
This module contains functions that are used to fetch collective noun data
from the collective_nouns.csv file.
Those are used by the RBS.
"""
import csv
from collections import defaultdict

def get_collective_noun_data(member_noun: str,
                             return_type: str) -> str | list[tuple[str, str, str]] | None:
    """
    Retrieves data related to a member noun from a CSV file containing collective nouns.

    Args:
      member_noun (str): The member noun to search for in the CSV file.
      return_type (str): The type of data to return. Possible values are:
                - "coll_noun": Returns the collective noun.
                - "coll_phrase_gender": Returns the gender of the collective noun phrase.
                - "coll_phrase_number": Returns the number of the collective noun phrase.
                - "dups_data": Returns the duplicates data.

    Returns:
      str: The requested data based on the return_type parameter.

    """
    with open("data/collective_nouns/collective_nouns.csv", "r", encoding="utf8") as cn_file:
        reader = csv.reader(cn_file, delimiter=",")
        next(reader)
        for row in reader:
            if member_noun == row[3]:
                # main-d'oeuvre
                coll_noun = row[1].replace("’", "'")
                coll_phrase_gender = row[4]
                coll_phrase_number = row[5]

                if return_type == "coll_noun" and coll_noun:
                    return coll_noun
                if return_type == "coll_phrase_gender" and coll_phrase_gender:
                    return coll_phrase_gender
                if return_type == "coll_phrase_number" and coll_phrase_number:
                    return coll_phrase_number
                if return_type == "dups_data" and member_noun:
                    dups = search_duplicates(member_noun)
                    return dups

    return None

def get_duplicates():
    """
    Retrieves duplicate entries from a CSV file.

    Returns:
        dict: A dictionary containing duplicate entries, where the key is the value of the duplicate field
              and the value is a list of tuples representing the data of the duplicate entries.
    """
    dups = defaultdict(list)
    with open("data/collective_nouns/collective_nouns.csv", "r", encoding="utf8") as cn_file:
        reader = csv.reader(cn_file, delimiter=",")
        next(reader)
        for row in reader:
            dups[row[3]].append((row[1], row[4], row[5]))

        dups = {k:v for k,v in dups.items() if len(v) > 1}

        return dups

def search_duplicates(member_noun: str) -> list[tuple[str, str, str]]:
    """
    Searches for duplicates entries of a given member noun.

    Args:
        member_noun (str): The member noun to search for duplicates of.

    Returns:
        list[tuple[str, str, str]]: A list of tuples representing the data
        of the duplicates entries.
    """
    dups = get_duplicates()
    tups = []
    if member_noun in dups.keys():
        for tup in dups[member_noun]:
            tups.append(tup)

    return tups
