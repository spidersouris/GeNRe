"""
Module used for evaluating the performance of the RBS components
(dependency detection and generation), and LLM generation.
"""

import ast
import os
import csv
import random
import argparse

from unicodedata import normalize
from typing import Dict
from collections import Counter

import spacy
from inflecteur import inflecteur

from jiwer import wer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sacrebleu import corpus_bleu

from rbs import gender_neutralizer, deps_detect

print("(eval) Loading spaCy model...")
nlp = spacy.load("fr_core_news_md")
print("(eval) spaCy model loaded.")

print("(eval) Loading inflecteur...")
inflecteur_instance = inflecteur()
inflecteur_instance.load_dict()
print("(eval) inflecteur loaded.")


def get_cos_sim(ref, hyp):
    """
    Calculates the cosine similarity between two sentences.

    Args:
      ref (str): The reference string.
      hyp (str): The hypothesis string.

    Returns:
      float: The cosine similarity between the reference and hypothesis sentences.
    """
    array = [ref, hyp]
    count_vectorizer = CountVectorizer()
    vector_matrix = count_vectorizer.fit_transform(array)
    cosine_similarity_matrix = cosine_similarity(vector_matrix)
    return round(cosine_similarity_matrix[0][1], 2) * 100


def get_wer(ref, hyp):
    """
    Calculate the Word Error Rate (WER) between two sentences.

    Args:
      ref (str): The reference string.
      hyp (str): The hypothesis string.

    Returns:
      float: The Word Error Rate (WER) between the reference and hypothesis sentences.
    """
    return wer(ref, hyp)


def get_bleu(ref, hyp):
    """
    Calculates the BLEU score between two sentences.

    Args:
      ref (str): The reference string.
      hyp (str): The hypothesis string.

    Returns:
      float: The BLEU score rounded to 2 decimal places for the reference
      and hypothesis sentences.
    """
    ref = [[ref]]
    hyp = [hyp]
    return round(corpus_bleu(hyp, ref).score, 2)


def extract_eval_data(
    inp_file,
    out_file,
    devset=None,
    evalset=None,
    corpus=None,
    max_sents=250,
    annotated=True,
):
    """
    Extracts evaluation data from the input file and writes it to the output file.

    Args:
      inp_file (str): The path to the input file.
      out_file (str): The path to the output file.
      devset (str, optional): The path to the development set file. Defaults to None.
      evalset (str, optional): The path to the evaluation set file. Defaults to None.
      corpus (str, optional): The corpus type. Can be "wiki" or "eupr". Defaults to None.
      max_sents (int, optional): The maximum number of sentences to extract.
      Defaults to 250.
      annotated (bool, optional): Indicates whether the data is annotated.
      Defaults to True.
    """
    count = 0
    sent_id = 0
    edata = []
    existdata = []

    if corpus == "wiki":
        MEMBER_NOUN_ROW = 3
        NON_INCL_SENT_ROW = 4
    elif corpus == "eupr":
        MEMBER_NOUN_ROW = 1
        NON_INCL_SENT_ROW = 2
    else:
        raise ValueError("Invalid corpus type", corpus)

    if devset is not None:
        with open(devset, "r", encoding="utf8") as devs:
            reader = csv.reader(devs, delimiter=",")
            next(reader)
            dev_data = list(reader)
    else:
        dev_data = []

    if evalset is not None:
        with open(evalset, "r", encoding="utf8") as evals:
            reader = csv.reader(evals, delimiter=",")
            next(reader)
            eval_data = list(reader)
            for row in eval_data:
                edata.append(row[3])
    else:
        eval_data = []

    with open(inp_file, "r", encoding="utf8") as inp:
        reader = csv.reader(inp, delimiter=",")
        next(reader)
        data = list(reader)

        # eval 1 seed: 18741
        # eval 2 seed: 48412
        # final dataset part 2 seed: 5565
        # random.seed(5565)
        random.seed(556007)
        random.shuffle(data)

        for row in data:
            non_incl_sents = []
            all_member_nouns = []
            detected_deps = []
            incl_sents = []
            sent_ids = []

            if (
                deps_detect.clean_tags(row[NON_INCL_SENT_ROW]) not in dev_data
                and deps_detect.clean_tags(row[NON_INCL_SENT_ROW]) not in edata
                and deps_detect.clean_tags(row[NON_INCL_SENT_ROW]) not in existdata
            ):
                print(row)
                if count >= max_sents:
                    break

                if not row[MEMBER_NOUN_ROW]:
                    continue

                try:
                    if not annotated:
                        non_incl_sents.append(row[1])
                        incl_sents.append(
                            gender_neutralizer.test_sentence(
                                row[1], nlp, inflecteur_instance, tests=True
                            )
                        )
                    else:
                        inflection_data = []
                        member_nouns = ast.literal_eval(
                            row[MEMBER_NOUN_ROW].replace("“", "'").replace("”", "'")
                        )
                        member_nouns2 = ast.literal_eval(
                            row[MEMBER_NOUN_ROW].replace("“", "'").replace("”", "'")
                        )
                        non_incl_sent = row[NON_INCL_SENT_ROW]
                        non_incl_sent_cleaned = deps_detect.clean_tags(non_incl_sent)

                        words_to_inflect: Dict[str, list] = deps_detect.extract_deps(
                            non_incl_sent, member_nouns, annotated=True
                        )

                        if words_to_inflect is not None:
                            inflection_data.append(
                                (member_nouns, words_to_inflect, non_incl_sent_cleaned)
                            )

                        res = gender_neutralizer.get_res(
                            non_incl_sent_cleaned,
                            member_nouns,
                            inflection_data,
                            allow_return=True,
                        )

                        non_incl_sents.append(non_incl_sent_cleaned)
                        all_member_nouns.append(member_nouns2)
                        detected_deps.append(words_to_inflect)
                        incl_sents.append(res)

                    sent_id += 1
                    sent_ids.append(sent_id)

                    count += 1

                    if not os.path.exists(out_file):
                        with open(out_file, "w", encoding="utf8", newline="") as out:
                            writer = csv.writer(out)
                            if not annotated:
                                writer.writerow(
                                    [
                                        "id",
                                        "non_incl_sent",
                                        "auto_incl_sent",
                                        "manual_incl_sent",
                                    ]
                                )
                            else:
                                writer.writerow(
                                    [
                                        "id",
                                        "member_nouns",
                                        "detected_deps",
                                        "non_incl_sent",
                                        "auto_incl_sent",
                                        "manual_incl_sent",
                                    ]
                                )

                    with open(out_file, "a", encoding="utf8", newline="") as out:
                        writer = csv.writer(out)
                        if not annotated:
                            for s_id, non_incl, incl in zip(
                                sent_ids, non_incl_sents, incl_sents
                            ):
                                writer.writerow([s_id, non_incl, incl])
                        else:
                            for s_id, member_nouns, deps, non_incl, incl in zip(
                                sent_ids,
                                all_member_nouns,
                                detected_deps,
                                non_incl_sents,
                                incl_sents,
                            ):
                                writer.writerow(
                                    [s_id, member_nouns, deps, non_incl, incl]
                                )
                except (IndexError, ValueError, KeyError, TypeError) as e:
                    print("Unexpected error", e)
                    continue

        print("Successfully extracted eval data to", out_file)


def get_gen_scores(
    inp_file, out_file=None, write_to_file=False, auto_col=None, print_error_types=False
):
    """
    Calculates generation scores for automatically generated sentences.

    Args:
        inp_file (str): The input file path.
        out_file (str, optional): The output file path. Defaults to None.
        write_to_file (bool, optional): Whether to write the scores to a file.
        Defaults to False.
        auto_col (int, optional): The index of the column containing
        the generated sentences. Defaults to None.
        print_error_types (bool, optional): Whether to print the error types.
        Defaults to False.
    """
    types = {
        "rbs_auto_incl_sent": "RBS",
        "t5_auto_incl_sent": "LLM-T5",
        "m2m100_auto_incl_sent": "LLM-M2M100",
        "mixtral_lazy_auto_incl_sent": "MIXTRAL-BASE",  # shelved
        "mixtral_dict_auto_incl_sent": "MIXTRAL-DICT",  # shelved
        "mixtral_correction_auto_incl_sent": "MIXTRAL-CORR",  # shelved
        "mixtral_lazy_auto_incl_sent_fs": "MIXTRAL-BASE-FS",
        "mixtral_dict_auto_incl_sent_fs": "MIXTRAL-DICT-FS",
        "mixtral_correction_auto_incl_sent_fs": "MIXTRAL-CORR-FS",
        "claude_lazy_auto_incl_sent_fs": "CLAUDE-BASE-FS",
        "claude_dict_auto_incl_sent_fs": "CLAUDE-DICT-FS",
        "claude_correction_auto_incl_sent_fs": "CLAUDE-CORR-FS",
    }
    count_sents = 0
    count_errors = Counter()
    cos_sim_scores_auto = []
    wer_scores_auto = []
    bleu_scores_auto = []

    cos_sim_scores_baseline = []
    wer_scores_baseline = []
    bleu_scores_baseline = []

    baseline_sents = []
    auto_incl_sents = []
    manual_incl_sents = []

    with open(inp_file, "r", encoding="utf8") as inp:
        reader = csv.reader(inp, delimiter=",")
        headers = next(reader)
        data = list(reader)

        if len(headers) > 21:
            raise ValueError("file not formatted correctly")

        header_types = {header: types.get(header, "UNKNOWN") for header in headers}
        header_type = header_types.get(headers[auto_col]) if auto_col else "RBS"

        for row in data:
            baseline_sent = normalize("NFKD", row[1])
            if auto_col:
                if auto_col not in range(2, 14):
                    raise ValueError("Invalid auto_col", auto_col)
                auto_incl_sent = normalize("NFKD", row[auto_col])
            else:
                auto_incl_sent = normalize("NFKD", row[2])
            manual_incl_sent = normalize("NFKD", row[14])

            if print_error_types:
                if not header_type:
                    raise ValueError("print_error_types requires auto_col")
                if header_type == "RBS":
                    error_types_row = row[15]
                elif header_type == "LLM-T5":
                    error_types_row = row[16]
                elif header_type == "LLM-M2M100":
                    error_types_row = row[17]
                else:
                    raise ValueError(
                        f"print_error_types: unknown header_type {header_type}"
                    )

                error_types = error_types_row.split("\n")
                for error_type in error_types:
                    count_errors[error_type] += 1

            if header_type == "LLM-T5":
                note_row = row[18].split("\n")
                if "GOOD_ALT_CHANGE_T5" in note_row:
                    print("Found alternative for T5")
                    print("Using alternative", row[19])
                    auto_incl_sent = row[19]
            elif header_type == "LLM-M2M100":
                note_row = row[17].split("\n")
                if "GOOD_ALT_CHANGE_M2M100" in note_row:
                    print("Found alternative for M2M100")
                    print("Using alternative", row[20])
                    auto_incl_sent = row[20]

            cos_sim_auto = get_cos_sim(auto_incl_sent, manual_incl_sent)
            wer_auto = get_wer(auto_incl_sent, manual_incl_sent)
            bleu_auto = get_bleu(auto_incl_sent, manual_incl_sent)

            cos_sim_baseline = get_cos_sim(baseline_sent, manual_incl_sent)
            wer_baseline = get_wer(baseline_sent, manual_incl_sent)
            bleu_baseline = get_bleu(baseline_sent, manual_incl_sent)

            cos_sim_scores_auto.append(cos_sim_auto)
            wer_scores_auto.append(wer_auto)
            bleu_scores_auto.append(bleu_auto)

            cos_sim_scores_baseline.append(cos_sim_baseline)
            wer_scores_baseline.append(wer_baseline)
            bleu_scores_baseline.append(bleu_baseline)

            baseline_sents.append(baseline_sent)
            auto_incl_sents.append(auto_incl_sent)
            manual_incl_sents.append(manual_incl_sent)

            count_sents += 1

            print(
                f"""Sentence {count_sents}\n"""
                f"""Cosine similarity: (BASELINE) {round(cos_sim_baseline, 3)}"""
                f""" | ({header_type}) {round(cos_sim_auto, 3)}\n"""
                f"""WER: (BASELINE) {round(wer_baseline*100, 3)}%"""
                f""" | ({header_type}) {round(wer_auto*100, 3)}%\n"""
                f"""BLEU: (BASELINE) {round(bleu_baseline, 3)}"""
                f""" | ({header_type}) {round(bleu_auto, 3)}\n"""
            )

        print(
            f"""Evaluation scores for {count_sents} sentences (file {inp_file}):\n
        BASELINE Average cosine similarity, WER, BLEU:\n
        {round(sum(cos_sim_scores_baseline)/count_sents, 3)}\t
        {round((sum(wer_scores_baseline)/count_sents)*100, 3)}%\t
        {round(sum(bleu_scores_baseline)/count_sents, 3)}\n
        ---\n
        {header_type} Average cosine similarity, WER, BLEU:\n
        {round(sum(cos_sim_scores_auto)/count_sents, 3)}\t
        {round((sum(wer_scores_auto)/count_sents)*100, 3)}%\t
        {round(sum(bleu_scores_auto)/count_sents, 3)}\n\n"""
        )

        if print_error_types:
            print(count_errors)

        if write_to_file and out_file is not None:
            with open(out_file, "w", encoding="utf8", newline="") as out:
                writer = csv.writer(out)
                writer.writerow(
                    [
                        "id",
                        "baseline_sent",
                        "auto_incl_sent",
                        "manual_incl_sent",
                        "cos_sim_base",
                        "cos_sim_auto",
                        "wer_base",
                        "wer_auto",
                        "bleu_base",
                        "bleu_auto",
                    ]
                )

                for i, (
                    baseline_sent,
                    auto_incl_sent,
                    manual_incl_sent,
                    cos_sim_base,
                    cos_sim_auto,
                    wer_base,
                    wer_auto,
                    bleu_base,
                    bleu_auto,
                ) in enumerate(
                    zip(
                        baseline_sents,
                        auto_incl_sents,
                        manual_incl_sents,
                        cos_sim_scores_baseline,
                        cos_sim_scores_auto,
                        wer_scores_baseline,
                        wer_scores_auto,
                        bleu_scores_baseline,
                        bleu_scores_auto,
                    )
                ):
                    writer.writerow(
                        [
                            i,
                            baseline_sent,
                            auto_incl_sent,
                            manual_incl_sent,
                            cos_sim_base,
                            cos_sim_auto,
                            wer_base,
                            wer_auto,
                            bleu_base,
                            bleu_auto,
                        ]
                    )
            print("Metrics saved to", out_file)


def eval_deps(pred_deps: str, true_deps: str, dct=False) -> tuple[float, float]:
    """
    Evaluates the precision and recall of predicted dependencies
    compared to true dependencies.

    Args:
        pred_deps (str): Comma-separated string of predicted dependencies.
        true_deps (str): Comma-separated string of true dependencies.
        dct (bool, optional): Whether the dependencies are in a dictionary.
        Defaults to False.

    Returns:
        tuple: A tuple containing the precision and recall values.
    """
    tps = set()
    if not dct:
        pred_list = list(filter(None, pred_deps.split(",")))
        true_list = list(filter(None, true_deps.split(",")))

        for dep in pred_list:
            if dep in true_list:
                print("Adding tp+1", dep, len(true_list))
                tps.add(dep)

        # ZeroDivisionError occurs when pred_list or true_list are empty,
        # which means that no dependencies are predicted
        # or are found in the sentence,
        # so we consider the prediction to be accurate.
        try:
            precision = len(tps) / len(pred_list)
        except ZeroDivisionError:
            precision = 1.0

        try:
            recall = len(tps) / len(true_list)
        except ZeroDivisionError:
            recall = 1.0

    else:
        try:
            pred_dict = dict(x.split(":") for x in pred_deps.split("|"))
            true_dict = dict(x.split(":") for x in true_deps.split("|"))
        except ValueError as e:
            raise ValueError(
                "Row is not formatted correctly."
                "\nPlease make sure to follow this format:"
                "\ntarget_noun1:dep1,dep2|target_noun2:dep1,dep2"
                f"\nCurrent: {pred_deps=} — {true_deps=}"
            ) from e

        for key, deps in pred_dict.items():
            if key in true_dict:
                for dep in deps.split(","):
                    if dep in true_dict[key].split(","):
                        tps.add(dep)

        precision = len(tps) / sum(len(v.split(",")) for v in pred_dict.values())
        recall = len(tps) / sum(len(v.split(",")) for v in true_dict.values())

    return precision, recall


def get_deps_score(inp_file, out_file=None):
    """
    Calculates the precision, recall, and F-score for RBS and baseline (spaCy)
    dependency parsing models based on the input file.

    Args:
        inp_file (str): The path to the input file.
    """

    count_sents = 0
    rbs_precisions = []
    rbs_recalls = []
    baseline_precisions = []
    baseline_recalls = []

    with open(inp_file, "r", encoding="utf8") as inp:
        reader = csv.reader(inp, delimiter=",")
        next(reader)
        data = list(reader)

        for row in data:
            rbs_deps = row[3]
            baseline_deps = row[4]
            manual_deps = row[5]

            is_dict = any("|" in dep for dep in [rbs_deps, baseline_deps, manual_deps])

            rbs_precision, rbs_recall = eval_deps(rbs_deps, manual_deps, dct=is_dict)
            baseline_precision, baseline_recall = eval_deps(
                baseline_deps, manual_deps, dct=is_dict
            )

            rbs_precisions.append(rbs_precision)
            rbs_recalls.append(rbs_recall)
            baseline_precisions.append(baseline_precision)
            baseline_recalls.append(baseline_recall)

            count_sents += 1

            print(
                f"""Sentence {count_sents} (ID {row[0]}): {row[1]}\n
            RBS Precision, Recall: {rbs_precision}, {rbs_recall}\n
            Baseline Precision, Recall: {baseline_precision}, {baseline_recall}\n\n"""
            )

    rbs_precision_sum = sum(rbs_precisions)
    rbs_recall_sum = sum(rbs_recalls)

    baseline_precision_sum = sum(baseline_precisions)
    baseline_recall_sum = sum(baseline_recalls)

    rbs_avg_precision = rbs_precision_sum / len(rbs_precisions)
    rbs_avg_recall = rbs_recall_sum / len(rbs_precisions)

    baseline_avg_precision = baseline_precision_sum / len(baseline_precisions)
    baseline_avg_recall = baseline_recall_sum / len(baseline_precisions)

    rbs_fscore = (
        2 * (rbs_avg_precision * rbs_avg_recall) / (rbs_avg_precision + rbs_avg_recall)
    )
    baseline_fscore = 2 * (
        (baseline_avg_precision * baseline_avg_recall)
        / (baseline_avg_precision + baseline_avg_recall)
    )

    print(
        f"""RBS Average Precision: {round(rbs_avg_precision, 3)}\n
    Baseline Average Precision: {round(baseline_avg_precision, 3)}\n\n
    RBS Average Recall: {round(rbs_avg_recall, 3)}\n
    Baseline Average Recall: {round(baseline_avg_recall, 3)}\n\n
    RBS F-score: {round(rbs_fscore, 3)}\n
    Baseline F-score: {round(baseline_fscore, 3)}"""
    )

    if out_file:
        out_file = os.path.splitext(out_file)[0]
        with open(out_file + "_detailed.csv", "w", newline="", encoding="utf8") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "id",
                    "sentence",
                    "rbs_precision",
                    "rbs_recall",
                    "baseline_precision",
                    "baseline_recall",
                ]
            )

            for i, (
                rbs_precision,
                rbs_recall,
                baseline_precision,
                baseline_recall,
                row,
            ) in enumerate(
                zip(
                    rbs_precisions,
                    rbs_recalls,
                    baseline_precisions,
                    baseline_recalls,
                    data,
                )
            ):
                writer.writerow(
                    [
                        i,
                        row[1],
                        rbs_precision,
                        rbs_recall,
                        baseline_precision,
                        baseline_recall,
                    ]
                )

        print(f"Successfully wrote detailed dependency data to {out_file}_detailed")

        with open(out_file + "_fscore.csv", "w", newline="", encoding="utf8") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "rbs_avg_precision",
                    "rbs_avg_recall",
                    "baseline_avg_precision",
                    "baseline_avg_recall",
                    "rbs_fscore",
                    "baseline_fscore",
                ]
            )

            writer.writerow(
                [
                    rbs_avg_precision,
                    rbs_avg_recall,
                    baseline_avg_precision,
                    baseline_avg_recall,
                    rbs_fscore,
                    baseline_fscore,
                ]
            )

        print(f"Successfully wrote fscore data to {out_file}_fscore")


def write_deps_to_file(inp_file, out_file, max_sents=250):
    """
    Write dependencies to a CSV file.

    Args:
        inp_file (str): The input file path.
        out_file (str): The output file path.
        max_sents (int, optional): The maximum number of sentences to process.
        Defaults to 250.
    """
    count_sents = 0
    rbs_deps = []
    baseline_deps = []
    inp_sents = []
    all_target_nouns = []

    with open(inp_file, "r", encoding="utf8") as inp:
        reader = csv.reader(inp, delimiter=",")
        next(reader)
        data = list(reader)

        for i, row in enumerate(data):
            if count_sents >= max_sents:
                break

            rbs_dict_pairs = []
            baseline_dict_pairs = []
            inp_sent = row[1]  # before: row3
            target_nouns = gender_neutralizer.find_targets(nlp(inp_sent))

            rbs_deps_dict = deps_detect.extract_deps(inp_sent, target_nouns)
            baseline_deps_dict = deps_detect.extract_deps_baseline(
                inp_sent, target_nouns
            )

            for k, v in rbs_deps_dict.items():
                if len(rbs_deps_dict) > 1:
                    rbs_dict_pairs.append(f"{k}:{','.join(v)}")
                else:
                    rbs_dict_pairs.append(f"{','.join(v)}")

            for k, v in baseline_deps_dict.items():
                if len(baseline_deps_dict) > 1:
                    baseline_dict_pairs.append(f"{k}:{','.join(v)}")
                else:
                    baseline_dict_pairs.append(f"{','.join(v)}")

            rbs_deps.append("|".join(rbs_dict_pairs))
            baseline_deps.append("|".join(baseline_dict_pairs))

            inp_sents.append(inp_sent)
            all_target_nouns.append(target_nouns)

            count_sents += 1

    with open(out_file, "w", encoding="utf8") as out:
        writer = csv.writer(out)
        writer.writerow(
            [
                "id",
                "inp_sent",
                "target_nouns",
                "rbs_deps",
                "baseline_deps",
                "manual_deps",
            ]
        )

        for i, (inp_sent, target_nouns, rbs_dep, baseline_dep) in enumerate(
            zip(inp_sents, all_target_nouns, rbs_deps, baseline_deps)
        ):
            writer.writerow([i, inp_sent, target_nouns, rbs_dep, baseline_dep])

    print(f"Successfully wrote dependency data to {out_file}")


def main():
    """
    Main function for the evaluation script.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("inp_file", type=str, help="The path to the input file.")
    parser.add_argument(
        "out_file", type=str, help="The path to the output file.", nargs="?"
    )
    parser.add_argument(
        "--component",
        "-c",
        type=str,
        help="The component to evaluate.",
        choices=["gen", "deps"],
    )
    parser.add_argument(
        "--devset", "-ds", type=str, help="The path to the development set file."
    )
    parser.add_argument(
        "--evalset", "-es", type=str, help="The path to the evaluation set file."
    )
    parser.add_argument(
        "--print_error_types", "-p", action="store_true", help="Print the error types."
    )
    parser.add_argument(
        "--autocol",
        "-ac",
        type=int,
        default=3,
        help="""The column number of the automatically generated
                        sentences to evaluate.""",
    )
    parser.add_argument(
        "--maxsents",
        "-ms",
        type=int,
        default=250,
        help="The maximum number of sentences to evaluate.",
    )
    parser.add_argument(
        "--annotated",
        "-a",
        action="store_true",
        help="Indicates whether the data is annotated.",
    )
    parser.add_argument(
        "--extract",
        "-e",
        action="store_true",
        help="Extract evaluation data from the input file.",
    )
    parser.add_argument(
        "--corpus", "-cp", type=str, help="The corpus type.", choices=["wiki", "eupr"]
    )
    parser.add_argument(
        "--write_deps",
        "-wd",
        action="store_true",
        help="Write dependencies to a file.",
    )
    args = parser.parse_args()

    if args.extract:
        if not args.corpus:
            raise ValueError("Please specify the corpus type: 'wiki' or 'eupr'.")
        extract_eval_data(
            args.inp_file,
            args.out_file,
            devset=args.devset,
            evalset=args.evalset,
            corpus=args.corpus,
            max_sents=args.maxsents,
            annotated=args.annotated,
        )
    else:
        if not args.component:
            raise ValueError("Please specify a component to evaluate: 'gen' or 'deps'.")
        if args.component == "gen":
            get_gen_scores(
                args.inp_file,
                args.out_file,
                write_to_file=True,
                auto_col=args.autocol,
                print_error_types=args.print_error_types,
            )
        elif args.component == "deps":
            if args.write_deps:
                write_deps_to_file(
                    args.inp_file, args.out_file, max_sents=args.maxsents
                )
            else:
                get_deps_score(args.inp_file, args.out_file)


if __name__ == "__main__":
    main()
