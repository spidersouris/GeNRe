import requests
from bs4 import BeautifulSoup
import re

url = "https://fr.wiktionary.org/wiki/-phone"
r = requests.get(url)
soup = BeautifulSoup(r.text, "html.parser")

# Récupération de la section Dérivés → Locuteur d'une langue
# par ciblage du style de l'élément <div>
content = soup.find("div",
                    attrs={"style": """column-width: 26em;
                           -webkit-column-width: 26em;
                           -moz-column-width: 26em;
                           vertical-align: top;
                           text-align: left;"""})
text = content.text

# Suppression des parenthèses et de leur contenu
text = re.sub(r"(\()[^)]*\)", "", text)
# Dans les cas où il y a plusieurs dérivés,
# on les sépare par un retour à la ligne
text = re.sub(r"\,([^)].*)", r"\n\1", text)
text = re.sub("phone", "phonie", text)