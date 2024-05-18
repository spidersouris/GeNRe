"""
Unit tests for the RBS (both the dependency extraction and the generation components).
"""

import csv
from collections import defaultdict
import unittest

import spacy
from inflecteur import inflecteur

from ..deps_detect import extract_deps
from ..gender_neutralizer import test_sentence

print("(tests) Loading spaCy model...")
nlp = spacy.load("fr_core_news_md")
print("(tests) spaCy model loaded.")

print("(tests) Loading inflecteur...")
inflecteur_instance = inflecteur()
inflecteur_instance.load_dict()
print("(tests) inflecteur loaded.")

class TestExtractDeps(unittest.TestCase):

  def test_from_file(self, file="data/tests/deps_tests.csv"):
    with open(file, "r", encoding="utf8") as f:
        reader = csv.reader(f)
        next(reader)
        for i, row in enumerate(reader, start=1):
            expected_deps = defaultdict(set)
            id, inp_sent, target_nouns, deps = row
            target_nouns = target_nouns.split(",")

            for target_noun in target_nouns:
              if deps:
                expected_deps[target_noun] = set([dep.strip() for dep in deps.split(",")])
              else:
                expected_deps[target_noun] = set()

              actual_deps = extract_deps(inp_sent, [target_noun])

              if target_noun in actual_deps:
                actual_deps[target_noun] = set(actual_deps[target_noun])
              else:
                actual_deps[target_noun] = set()

              self.assertDictEqual(actual_deps, expected_deps)

class TestTransformationPipeline(unittest.TestCase):
  """
  Test case class for the RBS generation component.
  """

  def test_from_file(self, file="data/tests/transformation_tests.csv"):
    with open(file, "r", encoding="utf8") as f:
      reader = csv.reader(f)
      next(reader)
      for i, row in enumerate(reader, start=1):
        id, inp_sent, out_sent = row
        out_sent = out_sent.replace("’", "'")

        self.assertEqual(test_sentence(inp_sent, nlp, inflecteur_instance, tests=True), out_sent)


if __name__ == "__main__":
  suite = unittest.TestSuite()
  suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestExtractDeps))
  suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestTransformationPipeline))
  runner = unittest.TextTestRunner(verbosity=0)
  runner.run(suite)
