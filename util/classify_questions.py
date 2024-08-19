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


# LABELS = ["very hard", "hard", "medium", "easy", "very easy"]
LABEL_COLOURS = {
    "very easy": "#FDE725",
    "easy": "#5EC962",
    "medium": "#21918C",
    "hard": "#3B528B",
    "very hard": "#440154",
}


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
    labels = list(reversed(LABEL_COLOURS.keys()))
    bins_qcut = pd.qcut(df.medshake_difficulty, len(labels), labels=labels)
    # print(bins_qcut.head())
    # print(bins_qcut.value_counts())

    # print("\n\nEqual-ranged classes (pandas.cut):")
    # bins_cut = pd.cut(df.medshake_difficulty, len(LABELS), right=True,
    #                   labels=LABELS)
    # print(bins_cut.head())
    # print(bins_cut.value_counts())

    df["medshake_class"] = bins_qcut
    df["colour_qcut"] = df["medshake_class"].apply(lambda x: LABEL_COLOURS[x])

    # df["medshake_class_cut"] = bins_cut
    # df["colour_cut"] = df["medshake_class_cut"].apply(lambda x: colourdict[x])

    return df


def plot_question_answer_rate(
        corpus_path: str, scatter_path: str,
        bars_path: str, bars_stack_by_class: bool = True,
) -> None:
    df = load_corpus(corpus_path)

    # print("\n\nCorpus with classes (sample):")
    # print(df.head())
    # print("\n\nCorpus info:")
    # print(df.info())

    # Sort by rate
    # df.sort_values(by="medshake_difficulty", inplace=True)

    # Plot with classes
    fig, ax = plt.subplots(figsize=(8, 4))
    # fig.suptitle("MCQ Correct Answer Rate")
    # fig.text(0.5, 0.02, "Questions", horizontalalignment="center")
    # fig.text(0.02, 0.5, "% Exams Answered Correctly", rotation="vertical",
    #          verticalalignment="center")
    ax.set_xlabel("Questions")
    ax.set_ylabel("% Exams Answered Correctly")

    # ax.set_title("Equal-sized classes (pandas.qcut)")
    ax.set_title("MCQ Correct Answer Rate")
    for label in LABEL_COLOURS.keys():
        _df = df[df["medshake_class"] == label]
        ax.scatter(_df.index, _df["medshake_difficulty"], c=_df["colour_qcut"], label=label)
    ax.legend()

    fig.savefig(scatter_path)


    # Bars plot to display distributuion of nbr_correct_answers per class
    if bars_stack_by_class:
        # One bar per number of answers, with stacked classes
        bars_data = {label: [] for label in LABEL_COLOURS}
        bars_index = []
        for num_answers, _df in df.groupby(by="nbr_correct_answers",
                                           observed=False):
            bars_index.append(num_answers)
            for label, count in _df["medshake_class"].value_counts().items():
                bars_data[label].append(count)
        xlabel = "Correct Answers"
        bar_colours = LABEL_COLOURS
        legend_data = {}
        path_suffix = "stacked_classes"
    else:
        # One bar per class, with stacked number of answers
        all_answers = sorted(df["nbr_correct_answers"].unique())
        bars_data = {n: [0] * len(LABEL_COLOURS.keys()) for n in all_answers}
        bars_index = []
        for idx, label in enumerate(LABEL_COLOURS.keys()):
            _df = df[df["medshake_class"] == label]
            bars_index.append(label)
            for n, count in _df["nbr_correct_answers"].value_counts().items():
                bars_data[n][idx] = count
        xlabel = "MedShake Class"
        bar_colours = None
        legend_data = {
            "labels": [
                f"{n} answer{'s' if n > 1 else ''}" for n in all_answers
            ],
            "loc": "lower left"
        }
        path_suffix = "stacked_answers"

    bars_df = pd.DataFrame(data=bars_data, index=bars_index)
    ax = bars_df.plot.bar(stacked=True, rot=0, color=bar_colours)
    ax.set_title("MCQ Correct Answers per Class")
    ax.set_ylabel("Number of Questions")
    ax.set_xlabel(xlabel)
    ax.legend(**legend_data)
    fig = ax.get_figure()
    fig.savefig(bars_path.replace(".png", f"_{path_suffix}.png"))


def get_average_by_difficulty(
        corpus: str | object, match_results: list[bool],
        hamming_results: list[float],
        medshake_results: list[float],
    ) -> tuple[dict[str, float], dict[str, float], dict[str, float]]:

    df = load_corpus(corpus)
    matches_by_class = {k: [] for k in LABEL_COLOURS}
    hamming_by_class = {k: [] for k in LABEL_COLOURS}
    medshake_by_class = {k: [] for k in LABEL_COLOURS}
    for i, sample in df.iterrows():
        matches_by_class[sample.medshake_class].append(match_results[i])
        hamming_by_class[sample.medshake_class].append(hamming_results[i])
        medshake_by_class[sample.medshake_class].append(medshake_results[i])

    matches_avg = {
        k: np.average(v)
        for k, v in matches_by_class.items()
    }
    hamming_avg = {
        k: np.average(v)
        for k, v in hamming_by_class.items()
    }
    medshake_avg = {
        k: np.average(v)
        for k, v in medshake_by_class.items()
    }
    return matches_avg, hamming_avg, medshake_avg


def main() -> None:
    plot_question_answer_rate(
      "data/dev-medshake-score.json",
      "output/llama3/plots/questions_medshake_classes.png",
      "output/llama3/plots/num_answers.png", bars_stack_by_class=True,
    )

    # emr, ham, med = get_average_by_difficulty(
    #     "data/dev-medshake-score.json",
    #     np.random.choice(a=[False, True], size=(1000,)),
    #     np.random.rand(1000),
    # )
    # print(emr)
    # print(ham)
    # print(med)


if __name__ == "__main__":
    import fire
    fire.Fire(main)
