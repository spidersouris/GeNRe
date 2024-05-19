"""
This script retrieves the section "Dérivés" of this French Wiktionary page:
https://fr.wiktionary.org/wiki/-phone

This is an automatic approach to get collective nouns that are derived
from the suffix "-phone" (e.g. "francophone" → "francophonie")
and add them to our collective nouns dictionary.
"""

import re
import requests
import csv
from bs4 import BeautifulSoup

URL = "https://fr.wiktionary.org/wiki/-phone"
r = requests.get(URL)
soup = BeautifulSoup(r.text, "html.parser")

# Getting section section Dérivés → Locuteur d'une langue
# by targeting <div>
content = soup.find("div",
                    {"class": "NavContent"}).find_next("div")
text = content.text

print(text)

# Delete parentheses and their content
text = re.sub(r"(\()[^)]*\)", "", text)

# When there's a comma, separate the words with a newline
text = re.sub(r"\,([^)].*)", r"\n\1", text)

with open("output/phone_wiktionary.csv", "w", encoding="utf8", newline="") as f:
    writer = csv.writer(f, delimiter=",")
    writer.writerow(["coll_noun", "member_noun_sg", "member_noun_pl"])
    for line in text.split("\n"):
        line = line.strip()
        if len(line) > 0:
            coll_noun = re.sub("phone", "phonie", line)
            writer.writerow([coll_noun, line, line + "s"])
