import csv
from collections import defaultdict
import unittest
from .. import deps_detect, gender_neutralizer

class TestExtractDeps(unittest.TestCase):

  def test_from_file(self, file="/content/drive/My Drive/Colab Notebooks/genre/tests/deps_tests.csv"):
    total_tests = sum(1 for line in open(file)) - 1

    with open(file) as f:
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

              actual_deps = deps_detect.extract_deps(inp_sent, [target_noun])

              if target_noun in actual_deps:
                actual_deps[target_noun] = set(actual_deps[target_noun])
              else:
                actual_deps[target_noun] = set()

              self.assertDictEqual(actual_deps, expected_deps)

    print("Tests ended")

class TestTransformationPipeline(unittest.TestCase):

  def test_from_file(self, file="/content/drive/My Drive/Colab Notebooks/genre/tests/transformation_tests.csv"):
    total_tests = sum(1 for line in open(file)) - 1

    with open(file) as f:
        reader = csv.reader(f)
        next(reader)
        for i, row in enumerate(reader, start=1):
            id, inp_sent, out_sent = row
            out_sent = out_sent.replace("’", "'")

            self.assertEqual(gender_neutralizer.test_sentence(inp_sent, tests=True), out_sent)

if __name__ == '__main__':
    suite = unittest.TestSuite()
    suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestExtractDeps))
    suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestTransformationPipeline))
    runner = unittest.TextTestRunner(verbosity=0)
    runner.run(suite)