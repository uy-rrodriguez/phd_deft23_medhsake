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


def shannon_entropy(instance: dict[str, any] | pd.Series,
                    deduplicate_answers: bool = False,
                    use_natural_log: bool = False) \
        -> float:
    """
    Returns the Shannon entropy for the given question instance calculated from
    the MedShake response rates.

    We define a normalised entropy, where values will be between 0 and 1, as
    follows:

        H_norm(P) = (- ∑ p_i * log(p_i)) / log(n), i ∈ {1..n}

        Where n is the number of possible answers and p_i is the proportion of
        students that replied the answer i.

    The MedShake data provides the number of responses per combination of
    answers (e.g., "a" = 15, "b" = 10, "a b" = 30), so the total n varies for
    each question.

    (NOTE: Deduplicating answers does not seem to properly represent the
    difficulty of questions.)
    However, we can deduplicate the responses and extract the number for each
    individual letter to calculate the entropy; in this case n would always be
    equal to 5 (letters range from a to e). This behaviour is disabled by
    default but can be enabled by using `deduplicate_answers` equal to True.

    (NOTE: There is little difference in the final result using one log
    function or the other.)
    The log is calculated on base 10 unless `use_natural_log` is True, which
    will use base e.
    """
    log_fn = np.log if use_natural_log else np.log10
    data: dict[str, int] = {
        k: v["nb_answer"]
        for k, v in instance["medshake"].items()
    }
    if deduplicate_answers:
        data = {
            letter: sum([v for k, v in data.items() if letter in k])
            for letter in instance["answers"].keys()
        }
    total_answers = sum([v for v in data.values()])
    sum_answers = sum([
        v / total_answers * log_fn(v / total_answers)
        for v in data.values()
    ])
    entropy = -1 * sum_answers / log_fn(len(data))
    # print(data, total_answers, sum_answers, len(data), entropy)
    return entropy


def test_shannon_entropy(id: str, deduplicate_answers: bool,
                         use_natural_log: bool) -> None:
    """
    Prints the Shannon entropy for a given instance id.
    """
    print(id, deduplicate_answers, use_natural_log)
    with open("data/test-medshake-score.json") as f:
        import json
        corpus = json.load(f)
    instance = next(filter(lambda x: x["id"] == id, corpus))
    if not instance:
        print("Instance not found")
    print(shannon_entropy(instance, deduplicate_answers, use_natural_log))


def load_corpus(corpus: str | dict[str, any]) -> pd.DataFrame:
    if isinstance(corpus, str):
        print(f"Loading corpus '{corpus}'")
        df = pd.read_json(corpus)
    else:
        df = pd.DataFrame(corpus)
    # print("\\nCorpus sample:")
    # print(df.head())
    # print("\n\nCorpus info:")
    # print(df.info())

    # >>> ADJUSTED IN DATA FILE <<<
    #
    # Adjust MedShake difficulty
    # Source data contains correct response rate, e.g. 0.6 if 60% of students
    # replied to the correct answer, so the higher the value the easier the
    # question. We want to invert this value, so higher values of
    # "medshake_difficulty" represent harder questions.
    # df["medshake_difficulty"] = df["medshake_difficulty"].apply(lambda x: 1 - x)
    # print(df.info())

    # Calculate Shannon entropy
    df["shannon_difficulty"] = df.apply(shannon_entropy, axis=1)
    # print("Min H", df["shannon_difficulty"].min(),
    #       df[df["shannon_difficulty"]
    #          == df["shannon_difficulty"].min()]["id"].values)
    # print("Avg H", df["shannon_difficulty"].mean())
    # print("Max H", df["shannon_difficulty"].max(),
    #       df[df["shannon_difficulty"]
    #          == df["shannon_difficulty"].max()]["id"].values)

    # Cut in N balanced classes
    # print("\n\nEqual-sized classes (pandas.qcut):")
    labels = LABEL_COLOURS.keys()
    # labels = list(reversed(LABEL_COLOURS.keys()))
    med_bins = pd.qcut(df["medshake_difficulty"], len(labels), labels=labels)
    # print(bins_qcut)
    sh_bins_qcut = pd.qcut(df["shannon_difficulty"], len(labels), labels=labels)
    # print(sh_bins_qcut)

    # print("\n\nEqual-ranged classes (pandas.cut):")
    # bins_cut = pd.cut(df.medshake_difficulty, len(LABELS), right=True,
    #                   labels=LABELS)
    # print(bins_cut.head())
    # print(bins_cut.value_counts())
    sh_bins_cut = pd.cut(
        df["shannon_difficulty"], len(labels), right=True, labels=labels)

    df["medshake_class"] = med_bins
    df["medshake_colour"] = \
        df["medshake_class"].apply(lambda x: LABEL_COLOURS[x])

    df["shannon_class_qcut"] = sh_bins_qcut
    df["shannon_colour_qcut"] = \
        df["shannon_class_qcut"].apply(lambda x: LABEL_COLOURS[x])

    df["shannon_class_cut"] = sh_bins_cut
    df["shannon_colour_cut"] = \
        df["shannon_class_cut"].apply(lambda x: LABEL_COLOURS[x])

    return df


def plot_question_classification(df: pd.DataFrame, scatter_path: str) -> None:
    # Plot with classes
    fig, ax = plt.subplots(nrows=3, sharex=True, figsize=(8, 12))
    fig.suptitle("MCQ Question Classification")
    # fig.text(0.5, 0.02, "Questions", horizontalalignment="center")
    # fig.text(0.02, 0.5, "% Exams Answered Correctly", rotation="vertical",
            #  verticalalignment="center")

    # ax.set_title("Equal-sized classes (pandas.qcut)")
    ax[0].set_title("Correct Response Rate (medshake_class)")
    ax[0].set_ylabel("% Exams NOT Answered Correctly")
    for label in LABEL_COLOURS.keys():
        _df = df[df["medshake_class"] == label]
        ax[0].scatter(
            _df.index, _df["medshake_difficulty"], c=_df["medshake_colour"],
            label=label)

    # Save the MedShake classification alone
    # ax[0].set_xlabel("Questions")
    # ax[0].legend()
    # ax0_bbox = ax[0].get_window_extent().transformed(
    #     fig.dpi_scale_trans.inverted())
    # fig.savefig(scatter_path.replace("_shannon", "_extract"),
    #             bbox_inches=ax0_bbox.expanded(1.23, 1.37))
    # ax[0].set_xlabel("")

    # Add Shannon classification
    ax[1].set_title("All Response Rates (shannon_class, qcut)")
    # ax[1].set_xlabel("Questions")
    ax[1].set_ylabel("Normalised Shannon Entropy")
    for label in LABEL_COLOURS.keys():
        _df = df[df["shannon_class_qcut"] == label]
        ax[1].scatter(
            _df.index, _df["shannon_difficulty"], c=_df["shannon_colour_qcut"],
            label=label)

    ax[2].set_title("All Response Rates (shannon_class, cut)")
    ax[2].set_xlabel("Questions")
    ax[2].set_ylabel(ax[1].get_ylabel())
    for label in LABEL_COLOURS.keys():
        _df = df[df["shannon_class_cut"] == label]
        ax[2].scatter(
            _df.index, _df["shannon_difficulty"], c=_df["shannon_colour_cut"],
            label=label)

    # Save all the classifications together
    fig.legend(labels=LABEL_COLOURS.keys(), loc=(0.82, 0.895))
    fig.savefig(scatter_path, bbox_inches="tight")


def plot_num_answers_distribution(
        df: pd.DataFrame, bars_path: str,
        bars_stack_by_class: bool = True,
        class_col: str = "medshake_class",
) -> None:
    """
    Generate bar plots to display distributuion of nbr_correct_answers per class
    """
    path_suffix = f"_{class_col.replace('_class', '')}"

    if bars_stack_by_class:
        # One bar per number of answers, with stacked classes
        bars_data = {label: [] for label in LABEL_COLOURS}
        bars_index = []
        for num_answers, _df in df.groupby(by="nbr_correct_answers"):
            bars_index.append(num_answers)
            for label, count in _df[class_col].value_counts().items():
                bars_data[label].append(count)
        xlabel = "Correct Answers"
        bar_colours = LABEL_COLOURS
        legend_data = {}
        path_suffix += "_stacked_classes"
    else:
        # One bar per class, with stacked number of answers
        all_answers = sorted(df["nbr_correct_answers"].unique())
        bars_data = {n: [0] * len(LABEL_COLOURS.keys()) for n in all_answers}
        bars_index = []
        for idx, label in enumerate(LABEL_COLOURS.keys()):
            _df = df[df[class_col] == label]
            bars_index.append(label)
            for n, count in _df["nbr_correct_answers"].value_counts().items():
                bars_data[n][idx] = count
        xlabel = {
            "medshake_class": "MedShake Class",
            "shannon_class_qcut": "Shannon Class (qcut)",
            "shannon_class_cut": "Shannon Class (cut)",
        }[class_col]
        bar_colours = None
        legend_data = {
            "labels": [
                f"{n} answer{'s' if n > 1 else ''}" for n in all_answers
            ],
            "loc":
                "upper left" if class_col == "shannon_class_cut"
                else "lower left"
        }
        path_suffix += "_stacked_answers"

    bars_df = pd.DataFrame(data=bars_data, index=bars_index)
    ax = bars_df.plot.bar(stacked=True, rot=0, color=bar_colours)
    ax.set_title(f"MCQ Correct Answers per Class ({class_col})")
    ax.set_ylabel("Number of Questions")
    ax.set_xlabel(xlabel)
    ax.legend(**legend_data)

    fig = ax.get_figure()
    fig.savefig(bars_path.replace(".png", f"{path_suffix}.png"))


def get_average_by_difficulty(
        corpus: str | dict[str, any] | pd.DataFrame,
        match_results: list[bool],
        hamming_results: list[float],
        medshake_results: list[float],
) -> tuple[dict[str, float], dict[str, float], dict[str, float]]:

    if not isinstance(corpus, pd.DataFrame):
        df = load_corpus(corpus)
    else:
        df = corpus

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
    df = load_corpus("data/test-medshake-score.json")

    # Scatter plot of questions and difficulties (MedShake, Shannon)
    #
    plot_question_classification(
        df,
        "output/plots/questions_classes_medshake_and_shannon.png",
    )

    # Bar plots of question difficulty vs number of correct answers
    #
    class_cols = ("medshake_class", "shannon_class_qcut", "shannon_class_cut")
    for class_col in class_cols:
        for bars_stack in (True, False):
            plot_num_answers_distribution(
                df,
                "output/plots/num_answers.png",
                bars_stack_by_class=bars_stack,
                class_col=class_col,
            )

    # emr, ham, med = get_average_by_difficulty(
    #     df,
    #     np.random.choice(a=[False, True], size=(1000,)),
    #     np.random.rand(1000),
    # )
    # print(emr)
    # print(ham)
    # print(med)


if __name__ == "__main__":
    import fire
    fire.Fire(main)
    # fire.Fire(test_shannon_entropy)
