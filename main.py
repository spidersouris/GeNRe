import argparse
import logging
import spacy
from inflecteur import inflecteur
from rbs.gender_neutralizer import test_sentence

print("Loading spaCy model...")
nlp = spacy.load("fr_core_news_md")
print("spaCy model loaded.")

print("Loading inflecteur...")
inflecteur_instance = inflecteur()
inflecteur_instance.load_dict()
print("Inflecteur loaded.")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-level",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                        default="INFO", help="Set the logging level")
    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level), format="{processName:<12} {message} ({filename}:{lineno})", style="{")

    while True:
        sent = input("Enter a sentence (or press Enter to exit): ")
        if not sent:
            break
        print(test_sentence(sent, nlp, inflecteur_instance))

if __name__ == "__main__":
    main()
