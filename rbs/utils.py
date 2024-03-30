import csv
from collections import defaultdict

def get_collective_noun_data(member_noun, return_type):
  with open("data/collective_nouns/collective_nouns.csv", "r", encoding="utf8") as cn_file:
    reader = csv.reader(cn_file, delimiter=",")
    next(reader)
    for i, row in enumerate(reader):
      if member_noun == row[3]:
        # main-d'oeuvre
        coll_noun = row[1].replace("’", "'")
        coll_phrase_gender = row[4]
        coll_phrase_number = row[5]

        if return_type == "coll_noun":
          return coll_noun
        elif return_type == "coll_phrase_gender":
          return coll_phrase_gender
        elif return_type == "coll_phrase_number":
          return coll_phrase_number
        elif return_type == "dups_data":
          dups = search_duplicates(member_noun)
          return dups

def get_duplicates():
  dups = defaultdict(list)
  with open("data/collective_nouns/collective_nouns.csv", "r", encoding="utf8") as cn_file:
    reader = csv.reader(cn_file, delimiter=",")
    next(reader)
    for row in reader:
        dups[row[3]].append((row[1], row[4], row[5]))

  dups = {k:v for k,v in dups.items() if len(v) > 1}

  return dups

def search_duplicates(member_noun):
  dups = get_duplicates()
  tups = []
  if member_noun in dups.keys():
    for tup in dups[member_noun]:
      tups.append(tup)

  return tups