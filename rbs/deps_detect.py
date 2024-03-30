import re
import string
from collections import defaultdict
import spacy
from .utils import get_collective_noun_data

CRED = "\33[31m"
CGREEN = "\33[32m"
CYELLOW = "\33[33m"
CBLUE = "\33[34m"
CVIOLET = "\33[35m"
CCYAN = "\033[96m"
CEND = "\x1b[0m"

CBLUEBG = "\33[104m"

# ---

PROPN_BASE_MAX_SEARCH = 2
PROPN_CONJ_MAX_SEARCH = 4
COORD_MAX_SEARCH = 3
OBL_CONTEXT_MAX_SEARCH = 4

SMALL_CONTEXT_MAX_SEARCH = 3
MEDIUM_CONTEXT_MAX_SEARCH = 4
LARGE_CONTEXT_MAX_SEARCH = 6

EXTENDED_PUNCT = ["«", "»"]

nlp = spacy.load("fr_core_news_sm")

# comment
pattern = r"<n(?:-(\d+))?>(\w+\s(\w+))<\/n>"

def clean_tags(sent: str) -> str:
  sent = re.sub(r"<n>", "", sent)
  sent = re.sub(r"<n-\d+>", "", sent)
  sent = re.sub(r"</n>", "", sent)
  return sent

def is_proper_noun(tokenized_sent, target_token):
  punct = "".join(c for c in string.punctuation if c != ",")

  # do not modify dependencies if followed by PROPN
  # e.g. "Les professeurs Dupont et Durand sont arrivés."
  for token in tokenized_sent[target_token.i:target_token.i+PROPN_BASE_MAX_SEARCH]:
    if ((token.pos_ in ["PROPN", "NOUN"] and token.dep_ == "flat:name")
    or (token.pos_ == "PROPN" and token.dep_ in ["flat:name", "dep"])):
      return True

  # check with "comme", "tel"
  # e.g. "Il fait également représenter des auteurs tels que Jean Cocteau (La Machine infernale)."
  for token in tokenized_sent[target_token.i:target_token.i+PROPN_CONJ_MAX_SEARCH]:
    if token.text in ["comme", "tel", "tels"]:
      if token.nbor():
        if token.text in ["tel", "tels"] and token.nbor().text == "que":
          if token.nbor().nbor().pos_ in ["PROPN", "NOUN"] and token.nbor().nbor().dep_ in ["flat:name", "dep"]:
            return True
        elif token.nbor().pos_ in ["PROPN", "NOUN"] and token.nbor().dep_ in ["flat:name", "dep", "obj"]:
          return True

  # If we cannot find a proper noun when checking
  # the neighbors of the target token,
  # we may find it by checking
  # the neighbors of every child of the target token.
  # e.g. "Dreux Budé (entre 1396 et 1399-avant), maître, était secrétaire du royaume de France Charles VII et Louis XI."
  for child in target_token.children:
    for token in tokenized_sent[child.i:child.i+PROPN_BASE_MAX_SEARCH]:
      #print("(option 2) checking proper noun for:", token, " - ", token.pos_, token.dep_)
      # e.g. "Les mafieux : Bunny Capistrano, « Ice-Pick » Joe, Kenny, Tom Hennigan (shérif de Las Vegas)."
      if token.text in punct:
        return False

      if ((token.pos_ in ["PROPN", "NOUN"] and token.dep_ == "flat:name")
      or (token.pos_ == "PROPN" and token.dep_ in ["flat:name", "dep"])):
        return True

  return False

def extract_deps_baseline(inp_sent: str, target_nouns: list[str]):
  modified_words = defaultdict(list)
  target_tokens = []

  sent = nlp(inp_sent)
  for target_noun in target_nouns:
    for token in sent:
      if token.text == target_noun:
        target_tokens.append(token)

  for target_token in target_tokens:
    for child in target_token.children:
      if child.text not in string.punctuation:
        modified_words[target_token.text].append(child.text)

  return modified_words

def extract_deps(inp_sent: str, target_nouns: list[str], annotated=False):
  modified_words = defaultdict(list)
  target_tokens = []
  target_tokens_annotated_data = []
  proper_noun = False
  outer_break = False

  print(f"{CCYAN}={'=' * len(inp_sent)}{CEND}")
  print("Input sentence:", inp_sent)
  print(f"{CCYAN}={'=' * len(inp_sent)}{CEND}")

  if not annotated and len(target_nouns) > 0:
    sent = nlp(inp_sent)
    for target_noun in target_nouns:
        for token in sent:
            if token.text == target_noun:
                target_tokens.append(token)
  elif annotated:
    matches = re.findall(pattern, inp_sent)
    target_tokens_annotated_data.extend([(match[0], match[1], match[2]) for match in matches])
    sent = nlp(clean_tags(inp_sent))
    for tup in target_tokens_annotated_data:
      # This avoids having duplicates in target_tokens when we find the same
      # collective noun being used several times in the sentence.
      #
      # Example: "<n-55>Les rois</n> ont mangé avec <n-66>les rois</n>."
      # "rois" would be added four times: twice during the first tuple loop
      # (because there are two occurrences in the sentence)
      # and twice more during the second loop.
      if len(target_tokens) < len(target_tokens_annotated_data):
        for token in sent:
          if token.text == tup[2]:
            target_tokens.append(token)
  else:
    raise ValueError("""Argument 'annotated' set to True but target_nouns is empty!\n
    Could not find any annotated data in text.\n
    Are the nouns properly annotated?""")

  for i, target_token in enumerate(target_tokens):
    coordination = False
    print(f"{CYELLOW}Target_token {i}{CEND}:", target_token)

    # Logic to check if multiple target_tokens are coordinated.
    # If that's the case, we don't want to change the dependencies
    # as they already are plural.
    if not is_proper_noun(sent, target_token):
      for tt in target_tokens:
        if tt in target_token.children:
          print(tt, [k for k in target_token.children])
          context = sent[target_token.i:target_token.i+COORD_MAX_SEARCH]
          for token in context:
            print(token.text, target_token, token.nbor(1).text, token.nbor(1).dep_, type(tt), type(token.text))
            # do not apply if target_token is directly followed with adjective
            # as we still need to replace the adjective
            if token.text == target_token.text and token.nbor(1).dep_ in ["amod", "xcomp"]:
              outer_break = False
              break
            if token.text == "et":
              outer_break = True

      print("ob", outer_break)
      if outer_break:
        break

      # logic for obl
      # give more details later
      obl_context = sent[target_token.i:target_token.i+OBL_CONTEXT_MAX_SEARCH]
      member_noun_number = get_collective_noun_data(target_token.text, "coll_phrase_number")
      member_noun_gender = get_collective_noun_data(target_token.text, "coll_phrase_gender")
      for t in obl_context:
        if t.text == "les" and t.dep_ == "obj" and member_noun_number == "s":
          if t.nbor():
            if t.nbor().lemma_ == "avoir":
              if t.nbor(2):
                print(f"{CBLUE}Added {t.nbor(2).text}{CEND} (Code: TARGET_NBOR_IS_PP_WITH_AVOIR_AND_ANTECEDENT)")
                modified_words[target_token.text].append(t.nbor(2).text)

      # fixes undetected "qui" in 15
      try:
        if target_token.nbor() and target_token.nbor().dep_ == "nsubj" and target_token.nbor().pos_ == "PRON":
          print(f"{CBLUE}Added {target_token.nbor().head.text}{CEND} (Code: TARGET_NBOR_IS_VERB_PRECEDED_BY_REL_PRON)")
          modified_words[target_token.text].append(target_token.nbor().head.text)

          if target_token.nbor().head.nbor(-1) and target_token.nbor().head.nbor(-1).dep_ == "aux:tense":
            print(f"{CBLUE}Added {target_token.nbor().head.nbor(-1).text}{CEND} (Code: TARGET_NBOR_IS_AUX_OF_VERB_PRECEDED_BY_REL_PRON)")
            modified_words[target_token.text].append(target_token.nbor().head.nbor(-1).text)
      except IndexError:
        continue

      try:
        for token in sent[target_token.i:target_token.i+MEDIUM_CONTEXT_MAX_SEARCH]:
          if token.dep_ == "advmod" and token.nbor():
            if token.nbor().dep_ == "amod" and token.nbor().pos_ == "ADJ" and token in token.nbor().children:
              print(f"{CBLUE}Added {token.nbor().text}{CEND} (Code: TARGET_NBOR_IS_ADJ_PRECEDED_BY_ADV)")
              modified_words[target_token.text].append(token.nbor().text)

          if token.dep_ == "amod" and token.pos_ == "ADJ" and token in token.nbor(-1).children:
            print(f"{CBLUE}Added {token.text}{CEND} (Code: TARGET_NBOR_IS_ADJ)")
            modified_words[target_token.text].append(token.text)
      except IndexError:
        continue

      if target_token.dep_ == "acl:relcl":
        print(f"{CVIOLET}WARNING{CEND}: Found acl:relcl")
        for child in target_token.children:
          if child.dep_ == "aux:tense":
            print(f"{CBLUE}Added {child.text}{CEND} (Code: TARGET_CHILD_IS_AUX_TENSE_IN_REL_CL)")
            modified_words[target_token.text].append(child.text)

      else:

        for token in sent[target_token.i+SMALL_CONTEXT_MAX_SEARCH:target_token.i+LARGE_CONTEXT_MAX_SEARCH]:
          # s14
          if token.dep_ == "aux:tense" and token.nbor().pos_ == "VERB":
            print(f"{CBLUE}Added {token.text}{CEND} (Code: NON_CHILD_IS_AUX_TENSE_IN_REL_CL)")
            modified_words[target_token.text].append(token.text)

        # ancestor = ROOT ou autre
        for ancestor in target_token.ancestors:
          print("Ancestor:", ancestor.text, ancestor.dep_)

          # do not add digits nor punct
          if ancestor.text.isdigit() or ancestor.text in string.punctuation or ancestor.text in EXTENDED_PUNCT:
            continue

          if target_token in ancestor.children:
            # fixes: "En payant les gardiens, il y a possibilité de faire sortir des lettres."
            # second and: fixes "militaires" not detected in 29
            # if the target token is in an adverbial clause and directly followed by a comma,
            # there is very low (non-existent?) probability that a dependency will follow
            if ancestor.dep_ == "advcl" and target_token.nbor().text == ",":
              break

          # Verb/copula checking process
          if ancestor.pos_ == "VERB":
            # fixes wrongly-detected "désigne" in s10
            # the verb must be after the target
            if (target_token.i < ancestor.i):
              # fix not detected "rebaptisent"
              # in "À la fin des années 1980, les auteurs rebaptisent la série…"
              if not target_token.nbor().text == ancestor.text:
                medium_context = sent[target_token.i-MEDIUM_CONTEXT_MAX_SEARCH:target_token.i]
                context_text = [t.text for t in medium_context]

                large_context = sent[target_token.i-LARGE_CONTEXT_MAX_SEARCH:target_token.i]
                large_context_text = [t.text for t in large_context]
                for token in medium_context:
                  # second acl:relcl and: fixes undetected "prendront" in 16
                  # also take into account "y compris"
                  if target_token.nbor().dep_ != "acl:relcl":
                    #context_text = [t.text for t in context]
                    if (token.text in [",", "et"]):
                      if token.text == "," and ((token.i != target_token.i-1 and token.i != target_token.i-2) or (large_context_text.count(",") > 1)):
                        coordination = True
                      elif token.text == "et":
                        coordination = True
                      break
                    elif ("y" in context_text):
                      y_idx = context_text.index("y")
                      if y_idx > 0 and context_text[y_idx-1] == "compris":
                        coordination = True
                        break

              if not coordination:
                # fix "amènent" in s3
                # we need to include it with i+1
                if ancestor.pos_ == "VERB" and "Tense=Past|VerbForm=Part" not in ancestor.morph:
                  # Verbs preceded by auxiliary "avoir" should not be changed
                  if ancestor.nbor(-1).lemma_ != "avoir":
                    print(f"{CBLUE}Added {ancestor.text}{CEND} (Code: ANCESTOR_IS_VERB_AND_MODIFIES_SUBJ_TARGET (NOT_COORD)")
                    modified_words[target_token.text].append(ancestor.text)
                    if ancestor.nbor(-1).dep_ == "aux:pass":
                      print(f"{CBLUE}Added {ancestor.nbor(-1).text}{CEND} (Code: ANCESTOR_NBOR_IS_AUX_PASS)")
                      modified_words[target_token.text].append(ancestor.nbor(-1).text)

            for token in sent[ancestor.i-SMALL_CONTEXT_MAX_SEARCH:ancestor.i]:
              try:
                if ancestor.nbor() and ancestor.nbor().dep_ == "cop" and "VerbForm=Inf" not in ancestor.nbor().morph:
                  print(f"{CBLUE}Added {ancestor.nbor().text}{CEND} (Code: ANCESTOR_NBOR_IS_COP (VERB))")
                  modified_words[target_token.text].append(ancestor.nbor().text)
                  break

                if ancestor.pos_ == "VERB" and ancestor.nbor(-1).pos_ == "AUX" and any(x == target_token for x in sent[ancestor.i-SMALL_CONTEXT_MAX_SEARCH:ancestor.i] ):
                  print(f"{CBLUE}Added {ancestor.nbor(-1).text}{CEND} (Code: ANCESTOR_NBOR_IS_AUX)")
                  modified_words[target_token.text].append(ancestor.nbor(-1).text)
                  break
              except IndexError:
                continue

          if ancestor.pos_ == "AUX" and "Tense=Past|VerbForm=Part" not in ancestor.morph:
              print(f"{CBLUE}Added {ancestor.text}{CEND} (Code: ANCESTOR_IS_VERB_MISIDENTIFIED_AS_AUX_AND_MODIFIES_SUBJ_TARGET)")
              modified_words[target_token.text].append(ancestor.text)

          # Specific copula rule for verbs which are mistaken for adjectives
          # Example: Les souris sont **mangées** par le chat.
          if ancestor.pos_ == "ADJ":
            if target_token in ancestor.children:
              for token in sent[ancestor.i-SMALL_CONTEXT_MAX_SEARCH:ancestor.i]:
                # We don't want to get copulas which are VerbForm=Part (e.g. "étant")
                # as those do not need to be changed.
                if token.dep_ == "cop" and not any(x in token.morph for x in ("Number=Sing", "VerbForm=Part")):
                    print(f"{CBLUE}Added {token.text}{CEND} (Code: ANCESTOR_NBOR_IS_COP (ADJ))")
                    modified_words[target_token.text].append(token.text)
              if ancestor.dep_ == "ROOT":
                  print(f"{CBLUE}Added {ancestor.text}{CEND} (Code: ANCESTOR_IS_ADJ_AND_MODIFIES_TARGET")
                  modified_words[target_token.text].append(ancestor.text)


          print(f"Detected {len(list(ancestor.children))} children")
          for child in ancestor.children:
            print("child:", child)

            # do not add digits nor punct
            if child.text.isdigit() or child.text in string.punctuation or child.text in EXTENDED_PUNCT:
              continue

            if child == target_token and child.dep_ == "aux:tense" and child.nbor().pos_ == "VERB":
                print(f"{CBLUE}Added {child.text}{CEND} (Code: TARGET_CHILD_IS_AUX_TENSE)")
                modified_words[target_token.text].append(child.text)

            if child == target_token and (child.nbor().dep_ in ["acl", "amod"] and child.nbor() in target_token.children) and not any(child.nbor().text in v for v in modified_words.values()):
              print(f"{CBLUE}Added {child.nbor().text}{CEND} (Code: TARGET_CHILD_IS_ADJ_AND_MODIFIES_TARGET)")
              modified_words[target_token.text].append(child.nbor().text)

            if child == target_token and (child.dep_ == "amod" and child.nbor().dep_ == "nmod" and child in child.nbor().children):
              print(f"{CBLUE}Added {child.nbor().text}{CEND} (Code: TARGET_CHILD_IS_MISIDENTIFIED_ADJ_AND_MODIFIES_TARGET (child_dep={child.nbor().dep_}, target_dep={child.dep_}))")
              modified_words[target_token.text].append(child.nbor().text)

            # L'annonce de l'usage d'un pavillon de complaisance entraînera une nouvelle grève des marins allemands, ce qui conduira finalement TT-Line à mettre un terme aux activités d'Olau Line et dissoudre la société.
            if child == target_token and child.dep_ == "fixed" and child.nbor().dep_ == "nmod" and child.pos_ == "ADJ" and child.nbor().pos_ == "NOUN":
              print(f"{CBLUE}Added {child.nbor().text}{CEND} (Code: TARGET_CHILD_IS_MISIDENTIFIED_ADJ_AND_MODIFIES_TARGET (child_dep={child.nbor().dep_}, target_dep={child.dep_}))")
              modified_words[target_token.text].append(child.nbor().text)

            # check if coordinated element
            # s2
            try:
              if child.dep_ == "aux:tense" and child.lemma_ == "être" and child.nbor().pos_ == "VERB":
                print(f"{CBLUE}Added {child.text}{CEND} (Code: CHILD_IS_AUX_TENSE)")
                modified_words[target_token.text].append(child.text)

                if child.nbor().dep_ == "ROOT" and "VerbForm=Part" in child.nbor().morph:
                  print(f"{CBLUE}Added {child.nbor().text}{CEND} (Code: CHILD_NBOR_IS_AUX_TENSE)")
                  modified_words[target_token.text].append(child.nbor().text)

              # fixes s10 where "valeurs" was wrongly detected
              # we only want patterns such as "[ADJ] and [ADJ]"
              if child.nbor().dep_ == "cc" and child.dep_ == "xcomp" and "VerbForm=Inf" not in child.nbor().head.morph and not any(child.nbor().text in v for v in modified_words.values()):
                print(f"{CBLUE}Added {child.nbor().head.text}{CEND} (Code: CHILD_IS_ADJ_CC)")
                modified_words[target_token.text].append(child.nbor().head.text)

              if child.nbor().dep_ == "aux:pass" and target_token in ancestor.children and child.dep_ != "subj:pass":
                # first cond: fixes wrongly-detected "sont" in s21
                # do not add anything if target_token is obl:arg
                #
                # second cond: fixes wrongly-detected "il" in s??
                if target_token.dep_ == "obl:arg" or (child.pos_ == "PRON" and "Type=PronRel" not in child.morph):
                  break
                # fixes "sont" in "Les représentants ne sont pas soumis à une limitation du nombre de mandats."
                # second and: verifies if the ROOT refers to target_token to avoid false positives
                # e.g.: this way, "sont" in s11 is not detected b/c not related to "rois"
                print(f"{CBLUE}Added {child.nbor().text}{CEND} (Code: CHILD_IS_AUX_PASS)")
                modified_words[target_token.text].append(child.nbor().text)

                if child.nbor() in ancestor.children:
                  modified_words[target_token.text].append(ancestor.text)

                if child.nbor(2).dep_ in ["VERB", "ROOT"] and "VerbForm=Part" in child.nbor(2).morph:
                  print(f"{CBLUE}Added {child.nbor(2).text}{CEND} (Code: CHILD_NBOR_IS_PRECEDED_BY_AUX_PASS)")
                  modified_words[target_token.text].append(child.nbor(2).text)

            except IndexError:
              continue

            # take into account object pronouns such as "eux"
            # lequel list: fixes wrongly-detected "lequel" in "sur lequel les musulmans se prosternent"
            if child.dep_ == "obl:mod" and child.pos_ == "PRON" and child.text not in ["lequel", "laquelle"]:
              print(f"{CBLUE}Added {child.text}{CEND} (Code: CHILD_IS_PRON_OBL_MOD)")
              modified_words[target_token.text].append(child.text)

            # fixes misidentified "élèvent" as noun
            if child.dep_ == "obl:mod" and child.head.pos_ == "VERB" and child.pos_ != "VERB":
              try:
                if child.nbor(-1).pos_ == "PRON" and "Person=3" in child.nbor(-1).morph:
                  print(f"{CBLUE}Added {child.text}{CEND} (Code: CHILD_IS_MISIDENTIFIED_VERB_PRECEDED_BY_PRON)")
                  modified_words[target_token.text].append(child.text)
              except IndexError:
                continue

            # do not add "amod" to "xcomp"; this adds too many things!
            # adding "amod" to "xcomp" was originally done for this sentence:
            # "Le Sénat vote le projet de loi autorisant le report à 70 ans de l'âge limite de départ à la retraite pour les salariés volontaires."
            if child.dep_ in ["xcomp"] and (child.pos_ == "ADJ" or "VerbForm=Part" in child.morph) and (ancestor.dep_ not in ["nsubj"]) and target_token.i < child.i:
              if ancestor.nbor(-1).lemma_ != "avoir":
                print(f"{CBLUE}Added {ancestor.text}{CEND} (Code: ANCESTOR_OF_CHILD_IS_PP_AND_MODIFIES_TARGET: {child.dep_})")
                modified_words[target_token.text].append(ancestor.text)
              # fixes wrong spaCy POS in 22
              # where "salariés" is considered AMOD and "volontaires" NMOD
              if child.nbor().dep_ not in ["nmod", "nsubj"]:
                if child.nbor(-1).lemma_ != "avoir":
                  print(f"{CBLUE}Added {child.text}{CEND} (Code: CHILD_IS_PP_AND_MODIFIES_TARGET: {child.dep_})")
                  modified_words[target_token.text].append(child.text)

  for token in sent:
    print(token.text, token.dep_, token.head.text, token.head.pos_, token.pos_, token.morph, token.lemma_,
            [child for child in token.children])

  print(modified_words)
  return modified_words