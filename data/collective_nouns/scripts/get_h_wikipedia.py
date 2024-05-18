""""
Script used to get all the italicized words from this
Wikipedia page: https://fr.wikipedia.org/wiki/H_aspir%C3%A9_en_fran%C3%A7ais
We need this list of words because in French, the letter "h" may be
either silent or aspirated. When it is aspirated, the word preceding
the "h" is not contracted with the determiner "le" or "la".

Examples:
"le haricot" vs *"l'haricot" (aspirated "h")
"l'homme" vs *"le homme" (silent "h")
"""

import requests
from bs4 import BeautifulSoup
from string import punctuation

URL = "https://fr.wikipedia.org/wiki/H_aspir%C3%A9_en_fran%C3%A7ais"
page = requests.get(URL)
res = []
soup = BeautifulSoup(page.content, "html.parser")

content = soup.find(id="mw-content-text")

tables = content.find_all("table")
for table in tables:
    cells = table.find_all("td")
    for cell in cells:
        bullets = cell.find_all("li")
        for bullet in bullets:
            # get all the italicized words
            words = bullet.find_all("i")
            for word in words:
                # if word equals "eaux" or "euse", print
                # the previous word (the word before "eaux" or "euse")
                # minus the last 3 characters
                # + "eaux" or "euse" depending on what has been found
                wd = word.text.lower().strip(punctuation).strip()
                idx = words.index(word) - 1
                #print(wd)
                if wd in ["eaux", "euse"]:
                    res.append(words[idx].text[:-3].lower() + wd)
                    #print("Added: " + words[idx].text[:-3].lower() + wd)
                elif wd == "ienne":
                    res.append(words[idx].text[:-2].lower() + "enne")
                    #print("Added: " + words[idx].text[:-2].lower() + "enne")
                elif wd in ["oise", "aise", "ote"]:
                    res.append(words[idx].text.lower() + "e")
                    #print("Added: " + words[idx].text.lower() + "e")
                elif wd == "ette":
                    res.append(words[idx].text[:-1].lower() + "ette")
                    #print("Added: " + words[idx].text[:-1].lower() + "ette")
                elif wd == "aux":
                    res.append(words[idx].text[:-1].lower() + "ux")
                    #print("Added: " + words[idx].text[:-1].lower() + "ux")
                elif wd == "ère":
                    res.append(words[idx].text[:-2].lower() + "ère")
                    #print("Added: " + words[idx].text[:-2].lower() + "ère")
                elif wd == "is":
                    res.append(words[idx].text[:-1].lower() + "is")
                    #print("Added: " + words[idx].text[:-1].lower() + "is")
                elif wd == "se":
                    res.append(words[idx].text[:-1].lower() + "se")
                    #print("Added: " + words[idx].text[:-1].lower() + "se")
                elif wd == "onne":
                    res.append(words[idx].text[:-2].lower() + "onne")
                    #print("Added: " + words[idx].text[:-2].lower() + "onne")
                elif wd == "as":
                    res.append(words[idx].text[:-1].lower() + "as")
                    #print("Added: " + words[idx].text[:-1].lower() + "as")
                elif wd == "ies":
                    res.append(words[idx].text[:-1].lower() + "ies")
                    #print("Added: " + words[idx].text[:-1].lower() + "ies")
                # quick fix. bad formatting on wikipedia page for this word
                elif wd == "hackeur -euse":
                    res.append("hackeur")
                    res.append("hackeuse")
                # idem
                elif wd == "Hague, la":
                    res.append("hague")
                elif "," in wd:
                    res.append(wd.split(",")[0])
                elif "!" in wd:
                    res.append(wd.split("!")[0])
                else:
                    res.append(wd)

with open("output/h_wikipedia.txt", "w", encoding="utf8") as f:
    for word in set(res):
        if word.startswith("h"):
            f.write(word + "\n")
        else:
            print("Not h, tofix: " + word)
