"""
Utilities to pre-process the source data, including extracting information
from it and manipulating the fields to adapt the values as necessary.
"""

import json
import os
import re
import sys

import numpy as np
import pandas as pd

# Hack to import local packages when this script is run from the terminal
sys.path.append(os.path.abspath("."))

import deft
from classify_questions import get_average_by_difficulty
from process_output import (
    get_results_dataframe,
    latex_print_results,
)


def invert_medshake_difficulty(
        corpus_path: str = "data/test-medshake-score.json",
        output_path: str = "data/test-medshake-score-INVERTED.json",
):
    """
    Inverts the value of MedShake difficulty (X' = 1 - X).

    Source data contains correct response rate, e.g. 0.6 if 60% of students
    replied to the correct answer, so the higher the value the easier the
    question. With this we invert this value, so higher values of
    "medshake_difficulty" represent harder questions.
    """
    print(f"Loading corpus '{corpus_path}'")
    df = pd.read_json(corpus_path)
    df["medshake_difficulty"] = df["medshake_difficulty"].apply(lambda x: 1 - x)
    with open(output_path, "w") as fp:
        json.dump(df.to_dict(orient="records"), fp, indent=4, ensure_ascii=False)


def student_rates():
    """
    Calculates MedShake and other rates for student responses and prints a LaTeX
    table.
    """

    # Test MedShake rate of all students
    corpus_path = "data/test-medshake-score.json"
    with open(corpus_path, "r") as f:
        corpus = json.load(f)

    all_match = []
    all_hamming = []
    all_medshake = []
    for instance in corpus:
        expected = instance["correct_answers"]
        total_nb = sum(x["nb_answer"] for x in instance["medshake"].values())

        # Average EMR of all students based on their answers
        # Average is calculated dividing by the total number of answers
        emr_avg = (
            # Only count students who answered the exact answer
            instance["medshake"][" ".join(expected)]["nb_answer"]
            / total_nb
        )

        # Average Hamming rate of all students based on their answers
        hamming_avg = (
            sum(
                # Hamming rate of each possible answer given by students
                v["nb_answer"] * deft.hamming(k.split(), expected)
                for k, v in instance["medshake"].items()
            )
            / total_nb
        )

        # Average MedShake rate of all students based on their answers
        medshake_avg = (
            sum(
                # Total MedShake score with values projected to range 0..1
                v["nb_answer"] * v["score"]
                    / max(y["score"] for y in instance["medshake"].values())
                for v in instance["medshake"].values()
            )
            / total_nb
        )

        all_match.append(emr_avg)
        all_hamming.append(hamming_avg)
        all_medshake.append(medshake_avg)

    emr_by_class, hamming_by_class, medshake_by_class = \
        get_average_by_difficulty(
            corpus,
            all_match,
            all_hamming,
            all_medshake,
        )

    results = {
        "emr": np.average(all_match),
        "hamming": np.average(all_hamming),
        "medshake": np.average(all_medshake),
        "emr_by_class": emr_by_class,
        "hamming_by_class": hamming_by_class,
        "medshake_by_class": medshake_by_class,
    }

    print(json.dumps(results, indent=2))

    df = get_results_dataframe([results])
    latex_print_results(df, single_table=True, table_title="Human Results",
                        highlight_top=False)

    # OUTPUT:
    #
    # {
    # "emr": 0.5171473715965633,
    # "hamming": 0.676611666347182,
    # "medshake": 0.5939119337786897,
    # "emr_by_class": {
    #     "very easy": 0.8850233316478332,
    #     "easy": 0.6918711644051261,
    #     "medium": 0.5010441118949224,
    #     "hard": 0.3395159967606134,
    #     "very hard": 0.16813016654048932
    # },
    # "hamming_by_class": {
    #     "very easy": 0.8898294805008868,
    #     "easy": 0.7437380828000256,
    #     "medium": 0.6419392783253318,
    #     "hard": 0.5785663094539305,
    #     "very hard": 0.5284604500280369
    # },
    # "medshake_by_class": {
    #     "very easy": 0.8880239906038166,
    #     "easy": 0.7286252130925659,
    #     "medium": 0.5852659598550435,
    #     "hard": 0.45975141903050926,
    #     "very hard": 0.30782834063664927
    # }
    # }
    # LaTeX table:
    # \begin{table}[H]
    # \centering
    # \begin{tabular}{@{}lllllll@{}}
    # \\
    # \multicolumn{8}{c}{\textbf{Human Results}} \\
    # \\
    # \multicolumn{7}{c}{MedShake rate} \\
    # \toprule
    # shots & medshake & very\_easy & easy  & medium & hard  & very\_hard \\ \midrule
    # 0     & 0.594    & 0.888      & 0.729 & 0.585  & 0.46  & 0.308      \\ \bottomrule
    # \\
    # \multicolumn{7}{c}{EMR} \\
    # \toprule
    # shots & emr   & very\_easy & easy  & medium & hard  & very\_hard \\ \midrule
    # 0     & 0.517 & 0.885      & 0.692 & 0.501  & 0.34  & 0.168      \\ \bottomrule
    # \\
    # \multicolumn{7}{c}{Hamming score} \\
    # \toprule
    # shots & hamming & very\_easy & easy  & medium & hard  & very\_hard \\ \midrule
    # 0     & 0.677   & 0.89       & 0.744 & 0.642  & 0.579 & 0.528      \\ \bottomrule
    # \end{tabular}
    # \caption{Rate results...}
    # \label{table:res_...}
    # \end{table}

    # med_rates = {
    #     instance["id"]:
    #     for instance in corpus
    # }
    # print(med_rates)

    # data = [
    #     {
    #         "a": {"nb_answer": 5, "score": 0},
    #         "a b": {"nb_answer": 10, "score": 1},
    #         "a c": {"nb_answer": 5, "score": 1},
    #         "a b c": {"nb_answer": 20, "score": 2},
    #     },
    #     {
    #         "a": {"nb_answer": 10, "score": 0},
    #         "a b c": {"nb_answer": 30, "score": 0},
    #         "a b c d e": {"nb_answer": 10, "score": 2},
    #     },
    # ]
    #
    # med_rate = np.average([
    #     sum(
    #         # Total score of students in this question
    #         # with scores projected to range 0..1
    #         v["nb_answer"] * deft.medshake_rate(k.split(), question_scores)
    #         for k, v in question_scores.items()
    #     )
    #     # Rate for this question (division of score by total number of answers)
    #     / sum(y["nb_answer"] for y in question_scores.values())
    #     for question_scores in data
    # ])
    #
    # print(med_rate)
    # assert med_rate == (27.5/40 + 10/50) / 2  # 0.44375


if __name__ == "__main__":
    import fire
    fire.Fire(invert_medshake_difficulty)
    # fire.Fire(student_rates)
