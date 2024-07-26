"""
Helper script to classify questions in equal-sized classes:
 - very easy
 - easy
 - medium
 - hard
 - very hard
"""

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


LABELS = ["very hard", "hard", "medium", "easy", "very easy"]


def load_corpus(corpus: str | object) -> pd.DataFrame:
    if isinstance(corpus, str):
        df = pd.read_json(corpus)
    else:
        df = pd.DataFrame(corpus)
    # print("\\nCorpus sample:")
    # print(df.head())
    # print("\n\nCorpus info:")
    # print(df.info())
    
    # Cut in N balanced classes
    # print("\n\nEqual-sized classes (pandas.qcut):")
    bins_qcut = pd.qcut(df.medshake_difficulty, len(LABELS), labels=LABELS)
    # print(bins_qcut.head())
    # print(bins_qcut.value_counts())

    # print("\n\nEqual-ranged classes (pandas.cut):")
    bins_cut = pd.cut(df.medshake_difficulty, len(LABELS), right=True,
                      labels=LABELS)
    # print(bins_cut.head())
    # print(bins_cut.value_counts())

    # colours = np.linspace(0, 1, len(LABELS))
    # colourdict = dict(zip(LABELS, colours)) 
    colourdict = {
        "very easy": "#FDE725",
        "easy": "#5EC962",
        "medium": "#21918C",
        "hard": "#3B528B",
        "very hard": "#440154",
    }

    df["medshake_class_qcut"] = bins_qcut
    df["colour_qcut"] = df["medshake_class_qcut"].apply(lambda x: colourdict[x])

    df["medshake_class_cut"] = bins_cut
    df["colour_cut"] = df["medshake_class_cut"].apply(lambda x: colourdict[x])

    return df


def plot_question_answer_rate(corpus_path: str, save_path: str) -> None:
    df = load_corpus(corpus_path)

    print("\n\nCorpus with classes (sample):")
    print(df.head())
    print("\n\nCorpus info:")
    print(df.info())

    # Sort by rate
    # df.sort_values(by="medshake_difficulty", inplace=True)

    # Plot with classes
    fig, ax = plt.subplots(3, sharex=True, sharey=True, figsize=(8, 10))
    fig.suptitle("MCQ Correct Answer Rate")
    fig.text(0.5, 0.02, "Questions", horizontalalignment="center")
    fig.text(0.02, 0.5, "% Exams Answered Correctly", rotation="vertical",
             verticalalignment="center")

    ax[0].set_title("Equal-sized classes (pandas.qcut)")
    for label in reversed(LABELS):
        _df = df[df["medshake_class_qcut"] == label]
        ax[0].scatter(_df.index, _df["medshake_difficulty"], c=_df["colour_qcut"], label=label)
    ax[0].legend()

    ax[1].set_title("Equal-range classes (pandas.cut)")
    ax[1].scatter(
        range(len(df.index)),
        [s["medshake_difficulty"] for _, s in df.iterrows()],
        c=df["colour_cut"].values,
    )
    ax[2].set_title("Equal-sized classes (with nbr_correct_answers)")
    ax[2].scatter(
        range(len(df.index)),
        [s["medshake_difficulty"] for _, s in df.iterrows()],
        c=df["colour_qcut"].values,
    )
    ax[2].plot(
        [s["nbr_correct_answers"] * 0.2 for _, s in df.iterrows()],
        "rx",
    )
    fig.savefig(save_path)


def get_average_by_difficulty(
        corpus: str | object, match_results: list[bool],
        hamming_results: list[float],
    ) -> tuple[dict[str, float], dict[str, float]]:

    df = load_corpus(corpus)
    matches_by_class = {k: [] for k in LABELS}
    hamming_by_class = {k: [] for k in LABELS}
    for i, sample in df.iterrows():
        matches_by_class[sample.medshake_class_qcut].append(match_results[i])
        hamming_by_class[sample.medshake_class_qcut].append(hamming_results[i])

    matches_avg = {
        k: np.average(v)
        for k, v in matches_by_class.items()
    }
    hamming_avg = {
        k: np.average(v)
        for k, v in hamming_by_class.items()
    }
    return matches_avg, hamming_avg


if __name__ == "__main__":
    # plot_question_answer_rate(
    #   "data/dev-medshake-score.json",
    #   "output/llama3/plots/questions_medshake_classes.png",
    # )
    emr, ham = get_average_by_difficulty(
        "data/dev-medshake-score.json",
        np.random.choice(a=[False, True], size=(1000,)),
        np.random.rand(1000),
    )
    print(emr)
    print(ham)
