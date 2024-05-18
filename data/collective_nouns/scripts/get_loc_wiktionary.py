"""
This script retrieves the section "Dérivés" of this French Wiktionary page:
https://fr.wiktionary.org/wiki/-phone

This is a semi-automatic approach to get collective nouns that are derived
from the word "-phone" (e.g. "francophone" → "francophonie")
and add them to our collective nouns dictionary.
"""

import re
import requests
from bs4 import BeautifulSoup

URL = "https://fr.wiktionary.org/wiki/-phone"
r = requests.get(URL)
soup = BeautifulSoup(r.text, "html.parser")

# Getting section section Dérivés → Locuteur d'une langue
# by targeting <div>
content = soup.find("div",
                    attrs={"style": """column-width: 26em;
                           -webkit-column-width: 26em;
                           -moz-column-width: 26em;
                           vertical-align: top;
                           text-align: left;"""})
text = content.text

# Delete parentheses and their content
text = re.sub(r"(\()[^)]*\)", "", text)

# When there's a comma, separate the words with a newline
text = re.sub(r"\,([^)].*)", r"\n\1", text)
text = re.sub("phone", "phonie", text)
