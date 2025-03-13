"""
Utilities to pre-process the source data, including extracting information
from it and manipulating the fields to adapt the values as necessary.
"""

import json
import os
import re
import sys

import lzma
from multiprocessing import Pool
import shutil

import numpy as np
import pandas as pd

from nltk.metrics.distance import edit_distance

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
    with open(corpus_path, encoding="utf-8") as fp:
        corpus = json.load(fp)
    for inst in corpus:
        if "medshake_difficulty" in inst:
            inst["medshake_difficulty"] = 1 - inst["medshake_difficulty"]
    with open(output_path, "w", encoding="utf-8") as fp:
        json.dump(corpus, fp, indent=4, ensure_ascii=True)


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


def _load_medshake_data(extra_data_path: str) -> list[dict]:
    """
    Helper to load a file with the latest data downloaded from MedShake.
    """
    print(f"Loading extra data '{extra_data_path}")
    with open(extra_data_path, encoding="utf-8") as fp:
        extra_list = json.load(fp)
    return extra_list


def _load_corpus(corpus_path: str) -> list[dict]:
    """
    Helper to load the corpus as a list of Python dicts.
    """
    print(f"Loading corpus '{corpus_path}")
    with open(corpus_path) as fp:
        corpus = json.load(fp)
    return corpus


def _async_question_dist(choices, inst, data):
    if data is None:
        return None
    res = []
    res.append(edit_distance(inst["question"], data["question"]))
    for choice in choices:
        res.append(edit_distance(
            inst["answers"][choice], data["answers"][choice]))
    return res


def calc_question_distances(num_cpus: int = 32, ignore_matches: bool = True):
    """
    For each sample not yet tagged in the corpus, it calculates the Levenshtein
    edit-distance between the question and answers and the latest samples
    downloaded from MedShake.

    if `ignore_matches` is True, then samples in the corpus and MedShake data
    that exactly match each other are ignored. The resulting matrix will be
    of equal size as the large full datasets, but filled with "inf" where
    samples were ignored.

    Stores a large matrix with the distances between a sample in the corpus to
    each available sample in the MedShake data.

    The matrix is organised as follows:
     - There is one colum for each question in the corpus (e.g. i=0), followed
       by five columns for the answer choices (e.g. i=1..4).
     - There is one row for each question in the MedShake data, followed by five
       rows for each answer choice.
     - Each cell stores the distance from a question or answer text in the
       corpus to the corresponding element in the MedShake data.
     - Thus, cells in the intersection of question x answer have an infinite
       distance. The same applies at the intersection of answers with different
       indexes, e.g. the distance between Answer A and Answer B is infinite.

    With Qi being a corpus question, QiAl the answer l of question Qi, Kj a
    MedShake question, and KjAl the answer l of question Kj, and d the value of
    a distance, the matrix would look like:
                K1     K1A1      ...  Kn     ...  KnA5
        Q1      dQ1K1  inf       ...  dQnK1  ...  inf
        Q1A1    inf    dQ1K1_A1  ...  inf    ...  dQnK1_A5
        ...
        Qm      dQ1Km  inf       ...  dQnKm  ...  inf
        ...
        QmA5    inf    dQ1Km_A1  ...  inf    ...  dQnKm_A5

    E.g.:
        Distances for 2 questions and answers Qi and Kj:
            Q0 K0 => [89, 27, 17, 99, 31, 13]
            Q0 K1 => [74, 27, 16, 98, 34, 15]
            Q1 K0 => [91, 18, 21, 18, 21, 22]
            Q1 K1 => [83, 19, 21, 19, 20, 21]

        Matrix:
            [[89. inf inf inf inf inf 74. inf inf inf inf inf]
            [inf 27. inf inf inf inf inf 27. inf inf inf inf]
            [inf inf 17. inf inf inf inf inf 16. inf inf inf]
            [inf inf inf 99. inf inf inf inf inf 98. inf inf]
            [inf inf inf inf 31. inf inf inf inf inf 34. inf]
            [inf inf inf inf inf 13. inf inf inf inf inf 15.]
            [91. inf inf inf inf inf 83. inf inf inf inf inf]
            [inf 18. inf inf inf inf inf 19. inf inf inf inf]
            [inf inf 21. inf inf inf inf inf 21. inf inf inf]
            [inf inf inf 18. inf inf inf inf inf 19. inf inf]
            [inf inf inf inf 21. inf inf inf inf inf 20. inf]
            [inf inf inf inf inf 22. inf inf inf inf inf 21.]]
    """
    corpus_path = "data/train.json"
    extra_data_path = "data/medshake-year-and-topic.json"
    output_path = corpus_path.replace(".json", "_dist.bin.xz")
    choices = "a b c d e".split()  # This is known and fixed for the corpus
    m_span = len(choices) + 1

    corpus = _load_corpus(corpus_path)
    extra_list = _load_medshake_data(extra_data_path)

    # Ignore samples that match between the corpus and MedShake data
    if ignore_matches:
        for i, inst in enumerate(corpus):
            match = _find_exact_match(inst, extra_list)
            if match and type(match) == dict:
                corpus[i] = None
                extra_list[extra_list.index(match)] = None

    # Clean all questions and answers once
    for _list in (corpus, extra_list):
        for inst in _list:
            if inst is not None:
                inst["question"] = _clean_question(inst["question"])
                for c in inst["answers"]:
                    inst["answers"][c] = _clean_question(inst["answers"][c])

    # For debugging
    # corpus = [s if i in (235, 381) else None for i, s in enumerate(corpus)]
    # extra_list = [s if i in (39, 467) else None for i, s in enumerate(extra_list)]

    # Calculate distances between remaining samples
    m_size = (len(corpus) * m_span, len(extra_list) * m_span)
    m_dist = np.ones(m_size) * np.inf
    for i, inst in enumerate(corpus):
        if inst is not None:
            row = i * (len(choices) + 1)
            with Pool(num_cpus) as p:
                results = p.starmap(
                    _async_question_dist,
                    [(choices, inst, data) for data in extra_list],
                )
            for j, dist in enumerate(results):
                if dist is not None:
                    # print(i, j, dist)
                    col = j * m_span
                    m_dist[row, col] = dist[0]
                    for l in range(1, m_span):
                        m_dist[row+l, col+l] = dist[l]

    # Save as binary and compress
    if output_path.endswith(".xz"):
        with lzma.open(output_path, "wb") as fp:
            fp.write(m_dist.tobytes())
    else:
        m_dist.tofile(output_path)
    print(f"Matrix of shape {m_dist.shape} stored in '{output_path}'")


def _load_distance_matrix(
        matrix_path: str, corpus_len: int = None, extras_len: int = None,
        num_choices: int = 5, file_sep: str = ""
) -> np.ndarray:
    if matrix_path.endswith(".xz"):
        with lzma.open(matrix_path) as fp_in:
            matrix_path = matrix_path.removesuffix(".xz")
            with open(matrix_path, "wb") as fp_out:
                shutil.copyfileobj(fp_in, fp_out)
    if matrix_path.endswith(".npz"):
        # Storing in npz format takes about 2/3 more space but also stores the
        # shape. Compressing the npz archive with xz takes in total 1/3 more
        # than the binary option
        m_dist_z = np.load(matrix_path)
        m_dist: np.ndarray = m_dist_z["m"]
    else:
        # Storing as binary and then compressed with xz uses less space
        # but requires knowing the shape of the matrix
        m_span = num_choices + 1
        m_dist = np.fromfile(matrix_path, sep=file_sep)
        size = corpus_len * m_span, extras_len * m_span
        m_dist = m_dist.reshape(size)
    print("Loaded matrix of shape", m_dist.shape)
    return m_dist


def test_load_distance_matrix():
    matrix_path = "data/train_dist_sm.bin.xz"
    m_dist = _load_distance_matrix(matrix_path, 2171, 2460)
    print(m_dist)
    print(m_dist[1410:1416, 234:240])
    # # Save as bin.xz
    # matrix_path = matrix_path.replace(".bin", "-test.bin")
    # with lzma.open(matrix_path, "wb") as fp:
    #     fp.write(m_dist.tobytes())
    # m_dist = _load_distance_matrix(matrix_path, 2171, 2460)
    # print(m_dist)
    # print(m_dist[1410:1416, 234:240])
    # # Save as npz
    # matrix_path = matrix_path.replace(".bin.xz", ".npz")
    # np.savez_compressed(matrix_path, m=m_dist)
    # m_dist = _load_distance_matrix(matrix_path)
    # print(m_dist)
    # print(m_dist[1410:1416, 234:240])


def _clean_question(q: str) -> str:
    """
    Utility to clean questions and answers, removing spaces before ":?!".
    """
    q = re.sub(" ", " ", q)
    q = re.sub(r"\n", " ", q)
    q = re.sub(r" ([:?!])", r"\1", q)
    q = re.sub(r"’", "'", q)
    q = eval(json.dumps(q))
    return q


def _clean_compare(a: str, b: str) -> bool:
    """
    Compare two strings cleaned with `_clean_question`.
    """
    return _clean_question(a) == _clean_question(b)


def _clean_dist(a: str, b: str) -> float:
    """
    Compute the distance between two strings cleaned with `_clean_question`.
    """
    return edit_distance(_clean_question(a), _clean_question(b))


def _find_exact_match(
        inst: dict, extra_list: list[dict]
) -> dict|list[tuple[int, dict]]:
    """
    Find a sample that matches exactly the given instance.

    Returns the exact match if only one is found, and None if not found.

    Returns a list with all the samples found for the question if the question
    is duplicated and no sample matches the answers exactly.
    """
    q = _clean_question(inst["question"])
    possibles = [
        (i, s) for i, s in enumerate(extra_list)
        if s and q == _clean_question(s["question"])
    ]
    # No exact question match :(
    if not possibles:
        return None
    # Found match with duplicated question
    if len(possibles) > 1:
        matches = [
            (i, sample) for i, sample in possibles
            if all(
                all(map(_clean_compare, inst[c], sample[c]))
                for c in ("answers", "correct_answers")
            )
        ]
        # Return the exact match and the other possibilities evaluated
        if len(matches) > 0:
            # Very unlikely to have multiple questions matching exactly, but we
            # return the last sample
            return matches[0][1]
        return possibles
    # Matched a non-duplicated question!
    return possibles[0][1]


def _find_closest(
        inst: dict, extra_list: list[dict] = None,
        closest_samples: list[tuple[int, dict]] = None,
        inst_idx: int = None, m_dist: np.ndarray = None,
) -> tuple[dict, float]:
    """
    Returns a sample that's closest to the given instance based on the edit
    distance between questions and answers.

    If `closest_samples` is given, it must be a list of tuples (sample, index)
    with the samples and their index in the original MedShake data. If given,
    the initial search to find the closest samples is skipped and we proceed to
    compare answers.

    If the matrix of pre-calculated distances `m_dist` is given, the parameter
    `inst_idx` is required and it will be used to search for the distance in the
    matrix.
    """
    result = None
    m_span = 6  # Size occupied in the matrix per question: nbr choices + 1
    m_row = inst_idx * m_span  # Adjusted matrix row index

    # Compare questions
    assert extra_list is not None or closest_samples is not None
    assert m_dist is None or inst_idx is not None
    closest = np.inf
    if closest_samples is None:
        closest_samples = []
        for j, sample in enumerate(extra_list):
            if m_dist is not None:
                m_col = j * m_span  # Adjusted matrix column index
                dist = m_dist[m_row, m_col]
            else:
                dist = _clean_dist(inst["question"], sample["question"])
            if dist <= closest:
                closest = dist
                closest_samples.append((j, sample))

    if len(closest_samples) == 1:
        return closest_samples[0][1], closest

    # Compare answers
    q_dist = closest if closest < np.inf else 0
    closest = np.inf
    for j, sample in closest_samples:
        if m_dist is not None:
            m_col = j * m_span  # Adjusted matrix column index
            dist = np.sum(
                m_dist[m_row+1:m_row+m_span, m_col+1:m_col+m_span]
                .diagonal()
            )
        else:
            dist = sum([
                _clean_dist(inst["answers"][choice], sample["answers"][choice])
                for choice in inst["answers"]
            ])
        if dist <= closest:
            closest = dist
            result = sample
    return result, q_dist + closest


def test_find_closest():
    corpus_path = "data/train.json"
    extra_data_path = "data/medshake-year-and-topic.json"
    matrix_path = corpus_path.replace(".json", "_dist.bin.xz")

    corpus = _load_corpus(corpus_path)

    extra_list = _load_medshake_data(extra_data_path)

    # Load distance matrix
    m_dist = _load_distance_matrix(
        matrix_path, len(corpus), len(extra_list), file_sep="")

    # print(m_dist[1410:1416, 234:240])
    # print(m_dist[1410:1416, 2802:2808])
    # print(m_dist[2286:2292, 234:240])
    # print(m_dist[2286:2292, 2802:2808])
    # return

    for i in (235, 381):
        inst = corpus[i]
        print(inst)
        sample, dist = _find_closest(
            inst, extra_list=extra_list, inst_idx=i, m_dist=m_dist)
        print(dist)
        print(sample)
        print()


def merge_corpus_with_year_topic():
    corpus_path = "data/train.json"
    new_corpus_path = corpus_path.replace(".json", "-MERGED.json")
    extra_data_path = "data/medshake-year-and-topic.json"
    matrix_path = "data/train_dist.bin.xz"

    corpus = _load_corpus(corpus_path)
    extra_list = _load_medshake_data(extra_data_path)
    m_dist = _load_distance_matrix(matrix_path, len(corpus), len(extra_list))

    NEW_COLS = ["year", "year_txt", "question_nbr", "topics"]
    DEBUG_COLS = ["missing", "incorrect_answers",
                  "use_closest", "closest", "closest_dist", "closest_answers"]
    ALL_COLS = [
        "id",
        "question",
        "answers",
        *DEBUG_COLS,
        "correct_answers",
        "subject_name",
        "nbr_correct_answers",
        "medshake",
        "medshake_difficulty",
        *NEW_COLS,
    ]

    # Add extra data to corpus
    for i, inst in enumerate(corpus):
        match = _find_exact_match(inst, extra_list)
        if not match:
            print(f"No data found for {inst['id']}", file=sys.stderr)
            inst["missing"] = True

            # Find closest question
            if not inst.get("closest"):
                closest, dist = _find_closest(
                    inst, extra_list=extra_list, inst_idx=i, m_dist=m_dist)
                inst["use_closest"] = False
                inst["closest_dist"] = dist
                inst["closest"] = closest

            match = inst["closest"]

        # Handle duplicated questions (find that which shares the same answers)
        elif type(match) == list:
            print(f"No exact match for {inst['id']}", file=sys.stderr)
            inst["missing"] = True
            # Calculate distances between answers
            closest, dist = _find_closest(
                inst, closest_samples=match, inst_idx=i, m_dist=m_dist)
            inst["use_closest"] = False
            inst["closest_dist"] = dist
            inst["closest"] = closest
            match = closest

        # Add extra data found, including the student responses and medshake
        # difficulty if not existent
        inst.update({col: match[col] for col in NEW_COLS})
        if "medshake" not in inst:
            inst["medshake"] = match["medshake"]
        if "medshake_difficulty" not in inst:
            correct = " ".join(inst["correct_answers"])
            total_nb = sum(x["nb_answer"] for x in inst["medshake"].values())
            # There are errors in our data: some questions in "train" don't
            # have the correct value for "correct_answers"
            if correct not in inst["medshake"]:
                correct_nb = 0
                inst["incorrect_answers"] = True
            else:
                correct_nb = inst["medshake"][correct]["nb_answer"]
            inst["medshake_difficulty"] = 1 - (correct_nb / total_nb)

    # Save to new corpus file
    corpus = [
        {
            k: inst[k]
            for k in ALL_COLS
            if k in inst
        }
        for inst in corpus
    ]
    with open(new_corpus_path, "w") as fp:
        json.dump(corpus, fp, indent=4)
        fp.write("\n")


def apply_fix_merged_corpus(with_synonym: bool = True):
    """
    Use this after running `merge_corpus_with_year_topic` and manually tagging
    the question with "use_closest" True. This funcction will apply the data
    from the closest question to each applicable sample, and remove all
    intermediate data ("missing", "closest", etc.).

    If `with_synonym` is True (default) the closest question text is kept for
    future reference in a new attribute "synonym".
    """
    corpus_path = "data/train-MERGED.json"
    new_corpus_path = corpus_path.replace(".json", "-FIXED.json")

    corpus = _load_corpus(corpus_path)

    # Add extra data to corpus
    for inst in corpus:
        # Use data from closest question
        if inst.get("use_closest"):
            print(f"Fixing {inst['id']}: '{inst['question']}'", file=sys.stderr)
            if with_synonym:
                inst["synonym"] = inst["closest"]["question"]
            del inst["missing"]
            del inst["use_closest"]
            del inst["closest"]
            del inst["closest_dist"]

    with open(new_corpus_path, "w") as fp:
        json.dump(corpus, fp, indent=4)
        fp.write("\n")


def clean_fixed_merged_corpus():
    """
    Use this after running "apply_fix_merged_corpus" to remove all intermediate
    merge data, keeping only the necessary attributes to run the analysis.
    """
    corpus_path = "data/train-MERGED-FIXED.json"
    new_corpus_path = corpus_path.replace(".json", "-CLEAN.json")

    corpus = _load_corpus(corpus_path)

    for inst in corpus:
        if inst.get("missing"):
            print(f"Cleaning {inst['id']}: '{inst['question']}'", file=sys.stderr)
            del inst["use_closest"]
            del inst["closest"]
            del inst["closest_dist"]
            del inst["medshake"]
            del inst["medshake_difficulty"]
            del inst["year"]
            del inst["year_txt"]
            del inst["question_nbr"]
            del inst["topics"]
            del inst["missing"]     # Attribute moved to the bottom
            inst["missing"] = True

    with open(new_corpus_path, "w") as fp:
        json.dump(corpus, fp, indent=4)
        fp.write("\n")


def extract_missing_corpus():
    corpus_path = "data/train-MERGED-FIXED-CLEAN.json"
    new_corpus_path = "data/train-MISSING.json"

    corpus = _load_corpus(corpus_path)

    missing = []
    for inst in corpus:
        if inst.get("missing"):
            del inst["missing"]
            missing.append(inst)

    with open(new_corpus_path, "w") as fp:
        json.dump(missing, fp, indent=4)
        fp.write("\n")


def concat_corpus():
    """
    Concat multiple splits of the corpus into one file.
    """
    basedir = "data"
    files = [
        "train-with-medshake.json",
        "test-medshake-score.json",
    ]
    output_file = "train+test-with-medshake.json"

    # Concat corpus and tags files
    for prefix in ("", "tags-"):
        content = []
        if prefix:
            content = {}
        for f in files:
            with open(f"{basedir}/{prefix}{f}", encoding="utf-8") as fp:
                if prefix:
                    content.update(json.load(fp))
                else:
                    # Exclude records without MedShake data (applies to "test")
                    content.extend([
                        x for x in json.load(fp)
                        if "medshake" in x
                    ])
        output = f"{basedir}/{prefix}{output_file}"
        with open(output, "w", encoding="utf-8") as fp:
            json.dump(content, fp, indent=2, ensure_ascii=False)


def main(method_name: str, *args, **kwargs):
    import inspect
    module = inspect.getmodule(main)
    method = getattr(module, method_name)
    if not method:
        raise f"Method '{method_name}' not found"
    return method(*args, **kwargs)


if __name__ == "__main__":
    import fire
    fire.Fire(invert_medshake_difficulty)
    # fire.Fire(student_rates)
    # fire.Fire(calc_question_distances)
    # fire.Fire(test_load_distance_matrix)
    # fire.Fire(test_find_closest)
    # fire.Fire(merge_corpus_with_year_topic)
    # fire.Fire(apply_fix_merged_corpus)
    # fire.Fire(clean_fixed_merged_corpus)
    # fire.Fire(extract_missing_corpus)
