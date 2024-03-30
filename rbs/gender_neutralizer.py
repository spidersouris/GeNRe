import csv
import re
import string
import logging
from collections import defaultdict
from typing import List, Dict
import spacy
from .deps_detect import extract_deps, is_proper_noun

CRED = "\33[31m"
CGREEN = "\33[32m"
CYELLOW = "\33[33m"
CBLUE = "\33[34m"
CVIOLET = "\33[35m"
CCYAN = "\033[96m"
CEND = "\x1b[0m"

SMALL_CONTEXT_MAX_SEARCH = 3

member_nouns = []
inflection_data = []
words_to_inflect = []

def test_sentence(sent, nlp, inflecteur, inflection_data=None, tests=False):
  inflection_data = []
  tokenized_sent = nlp(sent)
  member_nouns = find_targets(tokenized_sent)

  if len(member_nouns) == 0:
    logging.info("Found nothing to change")
    if tests:
      return get_res(sent, inflecteur, nlp, no_changes=True, allow_return=True)
    get_res(sent, inflecteur, nlp, no_changes=True)

  else:
    words_to_inflect: Dict[str, list] = extract_deps(sent, member_nouns)
    if words_to_inflect is not None:
      inflection_data.append((member_nouns, words_to_inflect, sent))

    if tests:
      return get_res(sent, inflecteur, nlp, member_nouns, inflection_data, allow_return=True)
    get_res(sent, inflecteur, nlp, member_nouns, inflection_data)

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

def find_targets(tokenized_sent):
    member_nouns = []
    member_nouns_mult = []
    token_indices = []
    token_texts = [token.text for token in tokenized_sent]

    with open("data/collective_nouns/collective_nouns.csv", "r", encoding="utf8") as cn_file:
        reader = csv.reader(cn_file, delimiter=",")
        next(reader)

        for row in reader:
            for token_text in token_texts:
                if (token_text.lower() == row[3]) and (member_nouns.count(token_text) < token_texts.count(token_text)) and (token_text not in member_nouns_mult):
                    member_nouns.append(token_text.lower())

        for noun in member_nouns:
          while True:
            try:
              idx = token_texts.index(noun)
              token_indices.append(idx)
              token_texts[idx] = None
            except ValueError:
              break

    member_nouns = [x for _, x in sorted(zip(token_indices, member_nouns))]

    return member_nouns

def get_collective_noun_data(member_noun, return_type):
  with open("data/collective_nouns/collective_nouns.csv", "r", encoding="utf8") as cn_file:
    reader = csv.reader(cn_file, delimiter=",")
    headers = next(reader)
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

def replacement_pipeline(tokenized_sent, token, member_noun, collective_noun, collective_noun_number,
                        collective_noun_gender, member_nouns=None, simple_replace=False):
  tokens = [tkn.text for tkn in tokenized_sent]

  dets = {"les": ["la", "le", "l'"], "aux": ["à la", "au", "à l'"],
          "des": ["de la", "du", "de l'"], "ces": ["cette", "ce", "cet"],
          "ses": ["sa", "son"], "mes": ["ma", "mon"]}
  complex_dets = ["tous", "toutes"]
  complex_det = False

  coll_noun_starts_vowel = starts_vowel(collective_noun)

  logging.debug(f"{CRED}Features of collective noun counterpart of {member_noun} ({collective_noun}): number {collective_noun_number} and gender {collective_noun_gender}{CEND}")

  if simple_replace:
    if token.nbor(-1).text in dets.keys() and token.text == member_noun:
        det = token.nbor(-1).text

        if coll_noun_starts_vowel and det not in dets["ses"]:
          idx = 2
        elif collective_noun_gender == "f":
          idx = 0
        elif collective_noun_gender == "m":
          idx = 1

        tokens[tokens.index(det)] = dets[det][idx]
        tokens[tokens.index(token.text)] = collective_noun
        return " ".join(tokens)

  if token.nbor(-1).text.lower() in dets.keys():
    logging.debug(f"{CRED}Found determiner {token.nbor(-1).text.lower()}{CEND}")

    try:
      if token.nbor(-2).text.lower() in complex_dets:
        complex_det = True
        logging.debug(f"{CRED}Found complex determiner {token.nbor(-2).text.lower()}{CEND}")
    except IndexError:
      pass

    # Logic for choosing the right determiner
    # for collective nouns starting with a vowel
    if coll_noun_starts_vowel and tokens[token.i-1] not in ["ses", "mes"]:
      logging.debug("Replacing determiner and member noun")
      # special case for "ces"
      if collective_noun_gender == "f" and tokens[token.i-1].lower() == "ces":
        tokens[token.i-1] = dets["ces"][0]
        if complex_det:
          tokens[token.i-2] = "toute"
      elif coll_noun_starts_vowel:
        tokens[token.i-1] = dets[token.nbor(-1).text.lower()][2]
        if collective_noun_gender == "f" and complex_det:
          tokens[token.i-2] = "toute"
        elif collective_noun_gender == "m" and complex_det:
          tokens[token.i-2] = "tout"
        # if complex_det: remove complex_det?
    elif tokens[token.i-1].lower() == "des" and token.nbor(-1).pos_ == "DET" and not complex_det:
      if collective_noun_gender == "f":
        tokens[token.i-1] = dets["les"][0]
      else:
        tokens[token.i-1] = dets["les"][1]
    else:
      if collective_noun_number == "s":
        if collective_noun_gender == "f":
          # get the index of token.text in sent and replace it with dict value
          if complex_det:
            tokens[token.i-2] = "toute"
          logging.debug("Replacing determiner and member noun (feminine singular)")
          logging.debug(tokens)
          if coll_noun_starts_vowel:
            # that means we have "ses"/"mes"
            # (because the very first "if" has not been accessed)
            tokens[token.i-1] = dets[token.nbor(-1).text.lower()][1]
          else:
            tokens[token.i-1] = dets[token.nbor(-1).text.lower()][0]
          logging.debug("Determiner and member noun replaced with collective noun equivalents")
          #logging.debug("after:" + " ".join(tokens))
          new_sent = " ".join(tokens)
        elif collective_noun_gender == "m":
          if complex_det:
            tokens[token.i-2] = "tout"
          logging.debug("Replacing determiner and member noun (masculine singular)")
          logging.debug(tokens)
          logging.debug(tokens[token.i-1] + tokens[token.i])
          tokens[token.i-1] = dets[token.nbor(-1).text.lower()][1]
          logging.debug("Determiner and member noun replaced with collective noun equivalents")
          #logging.debug("after:", " ".join(tokens))
          new_sent = " ".join(tokens)
        else:
          raise ValueError(f"Unknown collective_noun_gender {collective_noun_gender}")
      elif collective_noun_number == "p":
          logging.debug("Replacing determiner and member noun (plural)")
          logging.debug("Determiner and member noun replaced with collective noun equivalents")
          logging.debug("after:", " ".join(tokens))
          new_sent = " ".join(tokens)
      else:
        raise ValueError(f"Unknown collective_noun_number {collective_noun_number}")

  tokens[token.i] = collective_noun
  new_sent = " ".join(tokens)
  return new_sent


def replace_member_phrases(member_nouns, sent, nlp, allow_mult_sents=False):
  additional_suggs = []
  additional_suggs_tups = []
  additional_suggs_dict = defaultdict(list)
  new_sents = []
  mult_sents = False
  mult_passes = False
  new_sent = None
  sent = fix_obl(sent, nlp, member_nouns=member_nouns, prelim=True)
  tokenized_sent = nlp(sent)

  final_sents = []

  if len(member_nouns) > 1:
    mult_passes = True

  if allow_mult_sents:
    for member_noun in member_nouns:
      if len(search_duplicates(member_noun)) > 0:
        additional_suggs_tups = search_duplicates(member_noun)
        mult_sents = True

        for tup in additional_suggs_tups:
          additional_suggs.append(tup[0])

        if member_noun not in additional_suggs_dict.keys():
          additional_suggs_dict[member_noun] = additional_suggs

  for token in tokenized_sent:
    for member_noun in member_nouns:
      logging.debug(f"{CRED}Checking if {member_noun} is token{CEND}")
      if not member_noun in sent:
        #raise ValueError(f"ERROR: member_noun {member_noun} not found in sent!")
        continue
      else:
        if token.text == member_noun:
          # do not modify member_phrase if followed by PROPN
          # e.g. "Les professeurs Dupont et Durand sont arrivés."
          if is_proper_noun(tokenized_sent, token):
            logging.debug("Not changing b/c NPRON")
          else:
            if mult_sents and member_noun in additional_suggs_dict.keys():
              # I commented this because otherwise the index of token was off and the wrong word was being replaced with the collective noun
              # for ref, sent was: Il est possible qu'il ait découvert l'alcool à l'université en 1826, comme nombre d'autres jeunes gens, mais l'un de ses camarades a témoigné du fait qu'il était réputé, parmi les professeurs, pour sa sobriété, son calme et sa discipline.
              # idk if I should leave it like this tho...
              # tokenized_sent = nlp(sent)
              for tup in additional_suggs_tups:
                collective_noun, collective_noun_gender, collective_noun_number = tup
                new_sent = replacement_pipeline(tokenized_sent, token, member_noun, collective_noun, collective_noun_number, collective_noun_gender)
                new_sents.append(new_sent)

            # if len(new_sents) > 0:
            #   for s in new_sents:
            #     collective_noun = get_collective_noun_data(member_noun, "coll_noun")
            #     collective_noun_gender = get_collective_noun_data(member_noun, "coll_phrase_gender")
            #     collective_noun_number = get_collective_noun_data(member_noun, "coll_phrase_number")
            #     new_sent = replacement_pipeline(s, token, member_noun, collective_noun, collective_noun_number, collective_noun_gender, simple_replace=True)
            #     final_sents.append(new_sent)
            collective_noun = get_collective_noun_data(member_noun, "coll_noun")
            collective_noun_gender = get_collective_noun_data(member_noun, "coll_phrase_gender")
            collective_noun_number = get_collective_noun_data(member_noun, "coll_phrase_number")

            if mult_passes:
              # after doing one replacement,
              # if several other replacements are needed,
              # we need to retokenize the sentence
              # to update the indices
              # and make the necessary modificatons
              # in the tokens list inside replacement_pipeline()
              # because we're using token.i
              # otherwise, token.i won't get updated
              tokenized_sent = nlp(sent)
              for token in tokenized_sent:
                if token.text == member_noun:
                  new_token = token
              sent = replacement_pipeline(tokenized_sent, new_token, member_noun, collective_noun, collective_noun_number, collective_noun_gender)
            else:
              sent = replacement_pipeline(tokenized_sent, token, member_noun, collective_noun, collective_noun_number, collective_noun_gender)
            member_nouns.pop(0)

  if len(new_sents) > 0:
    new_sents2 = []
    for new_sent in new_sents:
      new_sent = remove_spaces(new_sent)
      new_sent = fix_elision(new_sent.split(), nlp)
      new_sents2.append(new_sent)
    # remove additional_suggs? no
    #return new_sents2, additional_suggs
    return new_sents2, additional_suggs_dict
  else:
    new_sent = sent
    new_sent = remove_spaces(new_sent)
    new_sent = fix_elision(new_sent.split(), nlp)
    return [new_sent], additional_suggs

def starts_vowel(word):
  vowels = "aâéeèiïîoôöuûüAÂÉEÈIÏÎOÔÖUÛÜ"
  h_words = {line.rstrip() for line in open("data/collective_nouns/scripts/output/h_wikipedia.txt")}

  if type(word) == str:
    if (word[0] in vowels) or (word[0] == "h" and word.strip(string.punctuation) not in h_words):
      return True
  elif type(word) == spacy.tokens.doc.Doc:
    for w in word:
      if (w.text[0] in vowels) or (w.text[0] == "h" and w.lemma_ not in h_words):
        return True

  return False

def remove_spaces(sent):
  # merge all of this into one regex
  sent = re.sub(r"\s+[\-]\s+", "-", sent)
  sent = re.sub(r"\s+[\-]", "-", sent)
  sent = re.sub(r"\s+[\,]", ",", sent)
  sent = re.sub(r"\s+[\.]", ".", sent)
  sent = re.sub(r"[\(]\s+", "(", sent)
  sent = re.sub(r"\s+[\)]", ")", sent)
  sent = re.sub(r"[\[]\s+", "[", sent)
  sent = re.sub(r"\s+[\]]", "]", sent)
  sent = re.sub(r"\s+…", '…', sent)
  sent = re.sub(r"\s+\.{3}", '...', sent)
  sent = re.sub(r"[\"]\s+", '"', sent)
  # https://stackoverflow.com/a/61267770
  sent = re.sub(r'\"\s*([^\"]*?)\s*\"', '"\\1"', sent)
  # comment
  sent = re.sub(r"\s?['’]\s+", "'", sent)
  return sent

def fix_elision(tokens, nlp):
  for word in tokens:
    if word in ["la", "le", "ne", "se"]:
      noun = tokens[tokens.index(word)+1]
      if starts_vowel(nlp(noun)):
        logging.debug(f"fixing elision for {tokens[tokens.index(word)+1]}")
        det_idx = tokens.index(word)
        next_idx = det_idx + 1
        if word in ["la", "le"]:
          tokens[tokens.index(word)] = "l'"
        elif word == "ne":
          tokens[tokens.index(word)] = "n'"
        elif word == "se":
          tokens[tokens.index(word)] = "s'"
        # merge
        tokens[det_idx] = tokens[det_idx] + tokens[next_idx]
        # delete the word that comes after the determiner after merge
        del tokens[next_idx]

  return remove_spaces(" ".join(tokens))

def fix_obl(sent, nlp, member_noun=None, member_nouns=None, words=None, prelim=False):
  need_to_fix = False
  tokenized_sent = nlp(sent)
  tokens = [token.text for token in tokenized_sent]

  for token in tokenized_sent:
    if words:
      if token.text in words[member_noun]:
        for word in words[member_noun]:
          if word in ["eux", "elles"] and get_collective_noun_data(member_noun, "coll_phrase_number") == "s":
            if get_collective_noun_data(member_noun, "coll_phrase_gender") == "f":
              tokens[token.i] = "elle"
            else:
              tokens[token.i] = "lui"

            need_to_fix = True

    if prelim:
      for member_noun in member_nouns:
        if token.text == member_noun:
          context = tokenized_sent[token.i:token.i+4]
          member_noun_number = get_collective_noun_data(member_noun, "coll_phrase_number")
          member_noun_gender = get_collective_noun_data(member_noun, "coll_phrase_gender")
          for t in context:
            if t.text == "les" and t.dep_ == "obj" and member_noun_number == "s":
              #logging.debug("before" + tokens)
              if starts_vowel(tokens[t.i+1]):
                tokens[t.i] = "l'"
              elif member_noun_gender == "m":
                tokens[t.i] = "le"
              elif member_noun_gender == "f":
                tokens[t.i] = "la"

              #logging.debug("after", tokens)

              need_to_fix = True

  if need_to_fix:
    fixed_sent = remove_spaces(" ".join(tokens))
    fixed_sent = fix_elision(fixed_sent.split(), nlp)
    return fixed_sent
  else:
    return sent


def fix_pp(member_noun, sent, nlp, words=None, word=None):
  fixed_sent = sent
  tokenized_sent = nlp(sent)
  coll_noun_gender = get_collective_noun_data(member_noun, "coll_phrase_gender")
  aux = False

  for token in tokenized_sent:
    repl = False
    if words:
      if token.text in words[member_noun] or token.text+"s" in words[member_noun]:
        repl = True
    elif word:
      if token.text == word:
        repl = True

    #logging.debug("repl", repl, word, words)

    if repl:
      if (token.dep_ in ["xcomp", "conj", "acl:relcl", "ROOT"] and token.pos_ == "VERB"):
        for t in tokenized_sent[token.i-max(0,SMALL_CONTEXT_MAX_SEARCH):token.i-1]:
          if t.lemma_ == "avoir" and t.nbor().lemma_ != "être":
            aux = True
            break

        if aux == True:
          continue

        # number can be a problem in new sentences; keep that in mind!
        # this is only a quick fix
        if token.morph.get("VerbForm") == ["Part"] and token.morph.get("Number") in [["Plur"], []]:
          if get_collective_noun_data(member_noun, "coll_phrase_gender") == "f":
            #if token.text[-1] == "s" and "Number=Sing" not in token.morph:
            if token.text in ["soumis", "admis"]:
              # soumise
              fixed_sent = fixed_sent.replace(token.text, token.text[:-1] + "se")
            else:
              fixed_sent = fixed_sent.replace(token.text, token.text[:-1] + "e")
          else:
            fixed_sent = fixed_sent.replace(token.text, token.text[:-1])
        elif token.morph.get("VerbForm") == ["Part"] and token.morph.get("Number") == ["Sing"]:
          if get_collective_noun_data(member_noun, "coll_phrase_gender") == "f" and token.text[-1] in ["s", "e"]:
            # côtoyés in "Il y dresse les portraits des princes qu’il a côtoyés et évoque les conflits auxquels il a participé."
            fixed_sent = fixed_sent.replace(token.text, token.text[:-1] + "e")
          elif get_collective_noun_data(member_noun, "coll_phrase_gender") == "f" and token.text[-1] != "s":
            fixed_sent = fixed_sent.replace(token.text, token.text + "e")
        else:
          if get_collective_noun_data(member_noun, "coll_phrase_gender") == "f" and token.text[-1] in ["s", "e"]:
            fixed_sent = fixed_sent.replace(token.text, token.text[:-1] + "e")
          elif get_collective_noun_data(member_noun, "coll_phrase_gender") == "f" and token.text[-1] != "s":
            fixed_sent = fixed_sent.replace(token.text, token.text + "e")

      elif token.dep_ == "amod" and token.pos_ == "ADJ":
        if coll_noun_gender == "f" and token.text[-1] in ["s", "e"]:
          fixed_sent = fixed_sent.replace(token.text, token.text[:-1] + "e")
        elif coll_noun_gender == "f" and token.text[-1] != "s":
          fixed_sent = fixed_sent.replace(token.text, token.text + "e")
        #else:
          #fixed_sent = fixed_sent.replace(token.text, token.text[:-1])

  #logging.debug("Fixed sent:", fixed_sent)
  return fixed_sent

def check_dep_replacement(sent, dep, collective_noun, replacement):
  # Les intellectuels sont assimilés, mais les masses sont fanatiquement religieuses et voient les comédiens juifs avec dédain
  # we only want to replace the first "sont" in that sentence
  words = re.findall(r"\w+|[^\w\s]", sent, re.UNICODE)
  clean_collective_noun = collective_noun.strip(string.punctuation)

  indices = [i for i, word in enumerate(words) if word == dep]

  for idx in indices:
    for w in words:
      if w == dep:
        neighbors = words[max(0, idx-2):idx+2]

        if any(clean_collective_noun in neighbor for neighbor in neighbors):
          words[idx] = replacement

  # for index in indices_dep:
  #   neighbors = words[max(0, index-2):index+2]
  #   clean_collective_noun = collective_noun.strip(string.punctuation)
  #   if any(clean_collective_noun in neighbor for neighbor in neighbors):
  #       words[index] = replacement

  return " ".join(words)

def get_res(sent, inflecteur, nlp, member_nouns=None, inflection_data=None,
            allow_return=False, no_changes=False):
  suggestions = None
  generate_multiple_sents = False
  new_sents = []

  if inflection_data and len(inflection_data) > 0:
    for tup in inflection_data:
      member_nouns: List = tup[0]
      words: Dict[str, list] = tup[1]
      sent: str = tup[2]

      # we need to give a list to replace_member_phrases
      # because the words dictionary does not contain all the member_nouns as keys
      # the keys are only for member_nouns with dependencies.
      sents, suggestions = replace_member_phrases(member_nouns, sent, nlp)

      if len(suggestions) > 0:
        generate_multiple_sents = True

      # key = member_noun
      # values = member_noun's dependencies as list
      if len(words) == 0:
        for sent in sents:
          new_sents.append(sent)
      elif len(words) > 0:
          logging.info("Calling inflecteur... This may take a few seconds on the first run.")
          for k, v in words.items():
            k = str(k)
            if generate_multiple_sents and k in suggestions:
              dups_data = get_collective_noun_data(k, "dups_data")
              for i, sent in enumerate(sents):
                logging.debug("new_sents", new_sents)
                for dep in v:
                  replacement = inflecteur.inflect_sentence(dep, gender=dups_data[i][1], number=dups_data[i][2])
                  # count how many times we have "sont". if >1, run replace_dep. else, standard replace
                  if sent.count(dep) > 1 and dups_data:
                    coll_noun = dups_data[i][0]
                    sent = check_dep_replacement(sent, dep, coll_noun, replacement)
                  else:
                    sent = sent.replace(dep, replacement)
                sent = fix_obl(sent, nlp, k, words=words)
                sent = fix_pp(k, sent, nlp, words)
                sent = fix_elision([token.text for token in nlp(sent)], nlp)
                new_sents.append(sent)
            else:
              sent = sents[0]
              collective_noun_gender = get_collective_noun_data(k, "coll_phrase_gender")
              collective_noun_number = get_collective_noun_data(k, "coll_phrase_number")
              for dep in v:
                # Some problems have been identified with how inflecteur
                # inflects some dependencies (especially verbs).
                try:
                  replacement = inflecteur.inflect_sentence(dep, gender=collective_noun_gender, number=collective_noun_number)
                  if " " in replacement:
                    logging.debug("Detected word ill-converted by inflecteur: " + replacement)
                    replacement = dep
                except IndexError:
                  logging.debug("Word not recognized by inflecteur")
                  replacement = dep

                # checks if the dependency is a participle
                # all actual pp will have a gender associated with them
                pp = False
                inf_word_form = inflecteur.get_word_form(dep)
                if inf_word_form is not None:
                  inf_gender_value = inf_word_form["gender"].values[0]

                  if inf_gender_value is not None:
                    pp = True

                # count how many times we have dep. if >1, run replace_dep. else, standard replace
                if sent.count(dep) > 1:
                  sent = check_dep_replacement(sent, dep, get_collective_noun_data(k, "coll_noun"), replacement)
                else:
                  sent = sent.replace(dep, replacement)

                if pp:
                  sent = fix_pp(k, sent, nlp, word=replacement)

              sent = fix_obl(sent, nlp, k, words=words)
              # todo: uniformize
              sent = fix_elision([token.text for token in nlp(sent)], nlp)
              new_sents.append(sent)

  logging.debug("--- All the words in the sentence have been converted ---")

  if allow_return and not no_changes:
    if len(set(new_sents)) > 1:
      logging.info(new_sents)
      return new_sents
    else:
      new_sent = new_sents[0]
      logging.info(new_sent[0].upper() + new_sent[1:])
      return new_sent[0].upper() + new_sent[1:]
  elif allow_return and no_changes:
    return sent

  if len(new_sents) == 0 and not no_changes:
    raise ValueError("An unexpected error occurred!")
  elif len(new_sents) == 0 and no_changes:
    logging.info("Final (unchanged) sentence is: " + sent)
    if allow_return:
      return sent
  elif len(set(new_sents)) > 1:
    logging.info("Several sentence suggestions:")
    # use this when writing to file multiple sents
    for sent in new_sents:
      logging.info(sent[0].upper() + sent[1:])
  else:
    new_sent = new_sents[0]
    logging.info("Final sentence is: " + new_sent[0].upper() + new_sent[1:])

  if suggestions:
    logging.info("Suggestions:" + suggestions)