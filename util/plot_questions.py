"""
Script to generate multiple plots and tables describing the contents of the
datasets.
"""

import json
import os
import re
import sys

from itertools import chain

import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from transformers import AutoModel, AutoTokenizer

# Trick to import local packages when this script is run from the terminal
sys.path.append(os.path.abspath("."))

from util.analyse_questions import calc_first_last_words, calc_qa_lengths
from util.classify_questions import load_corpus


# LABELS = ["very hard", "hard", "medium", "easy", "very easy"]
LABEL_COLOURS = {
    "very easy": "#FDE725",
    "easy": "#5EC962",
    "medium": "#21918C",
    "hard": "#3B528B",
    "very hard": "#440154",
}


# def load_corpus(corpus: str | dict[str, any]) -> pd.DataFrame:
#     if isinstance(corpus, str):
#         df = pd.read_json(corpus)
#     else:
#         df = pd.DataFrame(corpus)
#     return df


def plot_nbr_correct_answers(dfs: dict[str, pd.DataFrame], fig_path: str) -> None:
    # Plot with classes
    fig, ax = plt.subplots(figsize=(6, 4))

    label_colours = {
        "train": "#5EC962",
        "dev": "#EA5346",
        "test": "#3B528B",
    }

    for split, df in dfs.items():
        _df = df.groupby("nbr_correct_answers")["id"].count()  # Count answers
        print(_df)
        print(_df.sum())
        _df = _df / len(df.index)  # Convert to %
        _df.plot(
            kind="line", fig=fig, ax=ax, xticks=[1, 2, 3, 4, 5],
            label=split, color=[label_colours[split]])

    ax.set_title("Distribution of Correct Answers")
    ax.set_ylabel("% Questions in corpus")
    ax.set_xlabel("Number of answers")

    # Save all the classifications together
    ax.legend()
    fig.savefig(fig_path, bbox_inches="tight")


def plot_question_length(df: pd.DataFrame, figure_path: str) -> None:
    rate_col = "medshake_difficulty"
    rate_col = "med"

    # Calculate lengths
    df = calc_qa_lengths(df)

    # Plot with classes
    fig, ax = plt.subplots(nrows=3, sharex=True, figsize=(8, 12))
    fig.suptitle("MCQ Question Classification by Length")
    ax[-1].set_xlabel("Relative Length")
    # fig.text(0.5, 0.02, "Questions", horizontalalignment="center")
    # fig.text(0.02, 0.5, "% Exams Answered Correctly", rotation="vertical",
            #  verticalalignment="center")

    cols = {
        "q_len": "Question Length",
        "qa_len": "Question + Answers Length",
        "a_avg_len": "Average Answer Length",
    }
    for i, col in enumerate(cols):
        ax[i].set_title(cols[col])
        ax[i].set_ylabel("MedShake Difficulty")
        for label in LABEL_COLOURS.keys():
            _df = df[df["medshake_class"] == label]
            ax[i].scatter(
                _df[col], _df[rate_col], c=_df["medshake_colour"],
                label=label)

    # Save the figure
    fig.legend(labels=LABEL_COLOURS.keys(), loc=(0.82, 0.895))
    fig.savefig(figure_path, bbox_inches="tight")


def plot_embeddings(
        df: pd.DataFrame,
        algorithm: str,
        embeddings_path: str,
        figure_path: str,
        force_reload: bool = False,
        model_checkpoint: str = "BAAI/bge-m3",
):
    """
    Executes a dimension reduction algorithm (PCA, tSNE, UMAP) from question
    embeddings and plots the result.

    https://huggingface.co/BAAI/bge-m3
    https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
    https://sbert.net/docs/sentence_transformer/pretrained_models.html

    :param algorithm: Algorithm between "pca", "tsne", or "umap".
    """
    if not force_reload and os.path.exists(embeddings_path):
        print(f"Loading embeddings from '{embeddings_path}'")
        embeddings = torch.load(embeddings_path, weights_only=True)
    else:
        print(f"Generating embeddings with '{model_checkpoint}'")

        # Maybe extract keywords instead of full question?
        # - spaCy
        # - https://github.com/vgrabovets/multi_rake
        questions = [q for q in df["question"]]

        tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
        model = AutoModel.from_pretrained(model_checkpoint)

        encoded_input = tokenizer(
            questions, padding=True, truncation=True, return_tensors="pt")
        with torch.no_grad():
            model_output = model(**encoded_input)

        # Mean Pooling - Take attention mask into account for correct averaging
        def mean_pooling(model_output, attention_mask):
            # First element of model_output contains all token embeddings
            token_embeddings = model_output[0]
            input_mask_expanded = \
                attention_mask.unsqueeze(-1) \
                    .expand(token_embeddings.size()) \
                    .float()
            return (
                torch.sum(token_embeddings * input_mask_expanded, 1)
                / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
            )
        embeddings = mean_pooling(model_output, encoded_input["attention_mask"])
        print(f"Saving embeddings to '{embeddings_path}'")
        torch.save(embeddings, embeddings_path)



    # PCA algorithm
    if algorithm == "pca":
        # Apply PCA to reduce the embeddings to 2 dimensions
        torch.manual_seed(0)
        u, s, v = torch.pca_lowrank(embeddings)
        result = torch.matmul(embeddings, v[:, :2])
        df_pca = pd.DataFrame(result, columns=["pca0", "pca1"])
        extra_cols = ["id", "medshake_class", "medshake_colour"]
        df_pca[extra_cols] = df[extra_cols]
        # print(df_pca)

        # Plot with classes
        fig, ax = plt.subplots(figsize=(8, 6))
        fig.suptitle("MCQ 2D Question Embeddings with Classification")
        ax.set_xlabel("Principal Comp 1")
        ax.set_ylabel("Principal Comp 2")
        for label in LABEL_COLOURS.keys():
            _df = df_pca[df_pca["medshake_class"] == label]
            ax.scatter(
                _df["pca0"], _df["pca1"],
                c=_df["medshake_colour"], label=label)
        ax.legend()

    # tSNE algorithm
    elif algorithm == "tsne":
        from sklearn.manifold import TSNE
        fig = plt.figure(figsize=(12, 6))
        fig.suptitle(f"MCQ Question Embeddings tSNE with Classes")

        for n_comp in (2, 3):
            tsne_out_path = figure_path.replace(".png", f"_{n_comp}d.json")
            if not force_reload and os.path.exists(tsne_out_path):
                df_tsne = pd.read_json(tsne_out_path, orient="records")
            else:
                print("Running tSNE algorithm")
                tsne = TSNE(
                    n_components=n_comp, learning_rate="auto", init="random")
                result = tsne.fit_transform(embeddings)
                # print(result.shape)

                df_tsne = pd.DataFrame(
                    result, columns=tsne.get_feature_names_out())
                extra_cols = ["id", "medshake_class", "medshake_colour"]
                df_tsne[extra_cols] = df[extra_cols]
                # print(df_tsne)

                # Save projection to file
                with open(tsne_out_path, "w") as fp:
                    json.dump(df_tsne.to_dict(orient="records"), indent=4, fp=fp)

            # Plot with classes
            if n_comp == 3:
                ax = fig.add_subplot(119 + n_comp, projection="3d")
                ax.set_zlabel("tSNE Comp 3")
                ax.view_init(40, -10)
            else:
                ax = fig.add_subplot(119 + n_comp)
            ax.set_xlabel("tSNE Comp 1")
            ax.set_ylabel("tSNE Comp 2")
            for label in LABEL_COLOURS.keys():
                _df = df_tsne[df_tsne["medshake_class"] == label]
                ax.scatter(
                    *[_df[f"tsne{i}"] for i in range(n_comp)],
                    c=_df["medshake_colour"], label=label)

        fig.legend(labels=LABEL_COLOURS.keys())

    # UMAP algorithm
    elif algorithm == "umap":
        umap_out_path = figure_path.replace(".png", ".json")
        if not force_reload and os.path.exists(umap_out_path):
            df_umap = pd.read_json(umap_out_path, orient="records")
        else:
            print("Running UMAP algorithm")
            import umap
            from sklearn.preprocessing import StandardScaler

            # M = np.loadtxt(input_file, skiprows=1)
            M = StandardScaler().fit_transform(embeddings)
            reducer = umap.UMAP()
            result = reducer.fit_transform(M)
            # print(result.shape)

            df_umap = pd.DataFrame(result, columns=["umap0", "umap1"])
            extra_cols = ["id", "medshake_class", "medshake_colour"]
            df_umap[extra_cols] = df[extra_cols]
            # print(df_umap)

            # Save projection to file
            with open(umap_out_path, "w") as fp:
                json.dump(df_umap.to_dict(orient="records"), indent=4, fp=fp)

        # Plot with classes
        fig, ax = plt.subplots(figsize=(8, 6))
        fig.suptitle(f"MCQ Question Embeddings UMAP with Classification")
        ax.set_xlabel("UMAP Comp 1")
        ax.set_ylabel("UMAP Comp 2")
        for label in LABEL_COLOURS.keys():
            _df = df_umap[df_umap["medshake_class"] == label]
            ax.scatter(
                _df["umap0"], _df["umap1"],
                c=_df["medshake_colour"], label=label)
        ax.legend()

    # Save the figure
    fig.savefig(figure_path, bbox_inches="tight")


def main_plot_embeddings(algorithm: str, force_reload: bool = False) -> None:
    """
    Scatter plot of questions embeddings in 2D or 3D with MedShake class, using
    PCA, tSNE, or UMAP.
    """
    supported = ("pca", "tsne", "umap")
    assert algorithm in supported, f"Algorithm '{algorithm}' not in {supported}"

    print(f"Plot Embeddings with {algorithm}")
    corpus_path = "data/test-medshake-score.json"
    df = load_corpus(corpus_path)
    plot_embeddings(
        df,
        algorithm,
        "output/plots/questions/questions_embed_BGE.pt",
        f"output/plots/questions/questions_embed_BGE_{algorithm}.png",
        force_reload=force_reload,
        # model_checkpoint="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    )


def plot_question_negations(df: pd.DataFrame, figure_path: str) -> None:
    """
    Plot questions with their class, highlighting negations.
    """
    rate_column = "medshake_difficulty"
    rate_column = "med"

    # TODO: Change to usage of manually tagged negations

    # Add a column indicating negation
    colours = {
        "affirmative": "#5EC962",
        "negative": "#EE1212",
    }
    negation_re = re.compile(r" (n'|ne )")
    df["negation"] = df["question"].apply(
        lambda x: 1 if negation_re.search(x) else 0
    )
    df["negation_colour"] = df["negation"].apply(
        lambda x: colours["negative"] if x else colours["affirmative"]
    )

    # Plot with classes
    fig, ax = plt.subplots(figsize=(8, 4))
    fig.suptitle("MCQ Questions Negation Classification")
    ax.set_xlabel("Question")
    ax.set_ylabel("MedShake Difficulty")

    # Plot affirmative questions
    for label in LABEL_COLOURS.keys():
        _df = df[df["negation"] == 0]
        _df = _df[_df["medshake_class"] == label]
        ax.scatter(
            _df.index, _df[rate_column],
            c=_df["medshake_colour"], label=label)

    # Plot highlighted negative questions
    _df = df[df["negation"] == 1]
    ax.scatter(
        _df.index, _df[rate_column],
        c=_df["negation_colour"], label="Negation")

    # Save the figure
    fig.legend()
    fig.savefig(figure_path, bbox_inches="tight")


def plot_questions_by_words(df: pd.DataFrame, figure_path: str) -> None:
    """
    Plot questions classified by the top first words and the last character.
    """
    n_top_words = 10

    # Identify most common first few words and last char
    df = calc_first_last_words(df)

    questions_by_first = df \
        .groupby(by="first_words", sort=False) \
        .size() \
        .sort_values(ascending=False)
    questions_by_last = df.groupby(by="last_char", sort=False)

    # Keep only the most meaningful words
    # (most readable solution I found is to recreate "questions_by_first")
    top_first_words = questions_by_first.head(n=n_top_words).index
    df["first_words"] = df["first_words"].apply(
        lambda x: x if x in top_first_words else "<other>"
    )

    questions_by_top_first = df.groupby(by="first_words", sort=False)

    print(questions_by_top_first)
    print(questions_by_last)

    plot_first_instead_of_last = False
    transposed = True

    if plot_first_instead_of_last:
        # One bar per MedShake class, with stacked count of start/end categories
        first_words = [k for k, _ in questions_by_top_first]
        bars_data = {}

        for group, _df in questions_by_top_first:
            by_class = _df.groupby(by="medshake_class", sort=False)
            bars_data[group] = {k: len(v) for k, v in by_class}
        print(bars_data)

        bars_df = pd.DataFrame(data=bars_data)  #, index=bars_index)
        bar_colours = None
        rotation = 0
        legend_data = {"labels": first_words}
        xlabel = "MedShake Difficulty"
        ylabel = "# Questions per First Words"
        if transposed:
            bars_df = bars_df.T
            bar_colours = LABEL_COLOURS
            rotation = 30
            legend_data = {}
            xlabel = "Question First Words"
            ylabel = "# Questions per MedShake Difficulty"

        ax = bars_df.plot.bar(stacked=True, rot=rotation, color=bar_colours)
        ax.set_title("MCQ First Words and Last Character")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.legend(**legend_data)
        fig = ax.get_figure()

        # Save the figure
        fig.savefig(figure_path, bbox_inches="tight")

    else:
        # One bar per MedShake class, with stacked count of start/end categories
        last_chars = [k for k, _ in questions_by_last]
        bars_data = {}

        for group, _df in questions_by_last:
            by_class = _df.groupby(by="medshake_class", sort=False)
            bars_data[group] = {k: len(v) for k, v in by_class}

        bars_df = pd.DataFrame(data=bars_data)  #, index=bars_index)
        bar_colours = None
        rotation = 0
        legend_data = {"labels": last_chars}
        xlabel = "MedShake Difficulty"
        ylabel = "# Questions per Last Chars"
        if transposed:
            bars_df = bars_df.T
            bar_colours = LABEL_COLOURS
            rotation = 30
            legend_data = {}
            xlabel = "Question Last Chars"
            ylabel = "# Questions per MedShake Difficulty"

        ax = bars_df.plot.bar(stacked=True, rot=rotation, color=bar_colours)
        ax.set_title("MCQ First Words and Last Character")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.legend(**legend_data)
        fig = ax.get_figure()

        # Save the figure
        fig.savefig(figure_path, bbox_inches="tight")


def main_all() -> None:
    test_corpus_path = "data/test-medshake-score.json"
    print(f"Loading corpus '{test_corpus_path}")
    test_df = load_corpus(test_corpus_path)
    all_dfs = {
        "train": load_corpus("data/train.json"),
        "dev": load_corpus("data/dev.json"),
        "test": test_df,
    }

    plot_nbr_correct_answers(all_dfs, "output/plots/nbr_correct_answers.png")


    # Enrich data with rates based on model output
    # force_reload = False
    # suffix = gen_output_suffix(
    #     regex_prompt_nbr=2,
    #     regex_finetuned=True,
    #     regex_answer_txt=True,
    # )
    # basedir = "output/llama3/tuned_002_20240731"
    # presaved_results = f"{basedir}/raw_rates_outputs{suffix}.json"

    # if not force_reload and os.path.exists(presaved_results):
    #     with open(presaved_results) as f:
    #         results: list[dict] = json.load(f)
    # else:
    #     pattern = get_filename_pattern(
    #         regex_prompt_nbr=2,
    #         regex_finetuned=True,
    #         regex_answer_txt=False,
    #     )
    #     print("Filename pattern:", pattern.pattern)
    #     paths = [
    #         os.path.join(basedir, f)
    #         for f in os.listdir(basedir)
    #         if pattern.match(f)
    #     ]
    #     print(f"Files found ({len(paths)}):", *paths, sep="\n")
    #     results = load_output_files(paths, corpus_path, pattern)

    #     if not results:
    #         raise "No output files were found when loading results data."
    #     with open(presaved_results, "w") as f:
    #         json.dump(results, f)

    # results_df = pd.DataFrame(results, index=df.index)
    # print(results_df.head())

    # df["emr"] = results_df["emr"]
    # df["ham"] = results_df["hamming"]
    # df["med"] = results_df["medshake"]
    # print(df.head())

    # Scatter plot of question length and MedShake difficulty
    #
    plot_question_length(
        test_df,
        "output/plots/questions_classes_by_length__med.png",
    )

    # Scatter plot of negations (0 or 1) with MedShake class
    #
    plot_question_negations(
        test_df,
        "output/plots/questions_negations_with_class__med.png",
    )

    # Scatter plot of questions classified by first and last few words, with
    # MedShake difficulty
    #
    # plot_questions_by_words(
    #     test_df,
    #     "output/plots/questions_classes_by_words.png",
    # )


def plot_tags(
        df: pd.DataFrame,
        tags_path: str,
        figure_path_tpl: str,
):
    """
    Plot the distribution of tags amongst the MedShake classes.
    """
    # Load tags
    print(f"Loading tags from '{tags_path}'")
    df_tags = pd.read_json(tags_path, orient="index")
    df_tags.drop("tag_highlight", axis=1, inplace=True)

    # Join DataFrames
    df_tags = df_tags.merge(
        df[["id", "medshake_class"]], on="id", validate="one_to_one",
        suffixes=("", "--orig"), copy=False)

    # Plot with classes
    # One bar per class, with stacked tag choices
    class_col = "medshake_class"
    for tag_col in df_tags.filter(regex="^tag_*").columns:
        tag_values = sorted(df_tags[tag_col].unique())
        print("Processing", tag_col, tag_values)
        bars_data = {n: [0] * len(LABEL_COLOURS.keys()) for n in tag_values}
        bars_index = []
        for idx, label in enumerate(LABEL_COLOURS.keys()):
            _df = df_tags[df_tags[class_col] == label]
            bars_index.append(label)
            for n, count in _df[tag_col].value_counts().items():
                bars_data[n][idx] = count
        bar_colours = None
        legend_data = {
            "labels": tag_values,
            "loc": "lower left",
        }
        path_suffix = f"_{tag_col}"
        figure_path = figure_path_tpl.replace(".png", f"{path_suffix}.png")

        bars_df = pd.DataFrame(data=bars_data, index=bars_index)
        ax = bars_df.plot.bar(stacked=True, rot=0, color=bar_colours)
        ax.set_xlabel("MedShake Class")
        ax.set_ylabel(tag_col)
        ax.legend(**legend_data)

        # Save the figure
        fig = ax.get_figure()
        fig.suptitle(f"MCQ {tag_col} per Class ({class_col})")
        fig.savefig(figure_path, bbox_inches="tight")


def main_plot_tags():
    corpus_path = "data/test-medshake-score.json"
    df = load_corpus(corpus_path)
    plot_tags(
        df,
        "data/tags-test-medshake-score.json",
        "output/plots/tags/plot.png",
    )


def plot_topics(
        df: pd.DataFrame,
        figure_path: str,
):
    """
    Plot the distribution of topics amongst the MedShake classes.
    """
    class_col = "medshake_class"
    diff_col = "medshake_difficulty"
    topics = sorted(
        set(
            chain.from_iterable(
                df["topics"].tolist())))

    # Bar plot using MedShake class
    # One bar per topic, with stacked classes
    bars_data = {n: [0] * len(LABEL_COLOURS.keys()) for n in topics}
    bars_index = []
    for idx, label in enumerate(LABEL_COLOURS.keys()):
        _df = df[df[class_col] == label]
        bars_index.append(label)
        cls_topics, cls_count = np.unique(
            list(
                chain.from_iterable(
                    _df["topics"].tolist())),
            return_counts=True)
        for n, count in zip(cls_topics, cls_count):
            bars_data[n][idx] = count
    bars_data = {
        k.split("/")[0].strip(): v
        for k, v in bars_data.items()
    }
    bars_df = pd.DataFrame(data=bars_data, index=bars_index).T
    bars_df.sort_index(inplace=True, key=lambda x: x.str.normalize("NFKD"))

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 4))
    bars_df.plot.bar(stacked=True, rot=0, color=LABEL_COLOURS.values(), ax=ax)
    ax.set_xlabel("Topics")
    ax.set_ylabel("MedShake Class")
    ax.xaxis.set_tick_params(rotation=80)
    ax.legend(labels=LABEL_COLOURS.keys(), loc="upper left")

    # Save the figure
    fig = ax.get_figure()
    fig.suptitle(f"MCQ MedShake Class by Topic ({class_col})")
    fig.savefig(figure_path.replace(".", "_class."), bbox_inches="tight")

    ############################################################################

    # Line plot using MedShake difficulty
    for plot_type in ("all", "by_class"):
        fig, ax = plt.subplots(figsize=(10, 4))
        suffix = "_difficulty"
        line_data_df = pd.DataFrame(
            [
                [topic.split("/")[0].strip(), inst[diff_col], inst[class_col]]
                for _, inst in df.iterrows()
                for topic in inst["topics"]
            ],
            columns=["topic", diff_col, class_col]
        )

        # Plot the mean by topic without splitting by class
        if plot_type == "all":
            ax.plot(
                line_data_df
                    .groupby(by="topic")[diff_col]
                    .mean()
                    .sort_index(key=lambda x: x.str.normalize("NFKD")),
                marker=".", label=label, c=LABEL_COLOURS["hard"])
            fig.suptitle(f"MCQ Avg MedShake Difficulty by Topic")
            suffix += "_all"

        # Plot means by topic and class
        elif plot_type == "by_class":
            # Use pivot table with columns sorted based on class order
            pivot = pd.pivot_table(
                line_data_df,
                index="topic",
                columns=class_col,
                values=diff_col,
                observed=True,
            )[LABEL_COLOURS.keys()]
            pivot.sort_index(
                inplace=True, key=lambda x: x.str.normalize("NFKD"))
            pivot.plot(
                ax=ax, color=LABEL_COLOURS, legend=False, marker=".",
                xticks=range(len(topics)))
            fig.suptitle(f"MCQ Avg MedShake Difficulty by Topic and Class")
            fig.legend(loc="upper right")
            suffix += "_class"

        # Save the figure
        ax.set_xlabel("Topic")
        ax.set_ylabel("MedShake Difficulty")
        ax.xaxis.set_tick_params(
            rotation=80, gridOn=True, grid_color="#EEEEEE", grid_dashes=(1, 2),
            grid_linewidth=1.5)
        fig.savefig(figure_path.replace(".", f"{suffix}."), bbox_inches="tight")


def main_plot_topics():
    corpus_path = "data/test-medshake-score.json"
    df = load_corpus(corpus_path)
    plot_topics(
        df,
        "output/plots/topics/topics_plot.png",
    )


def plot_years(
        df: pd.DataFrame,
        figure_path: str,
):
    """
    Plot the distribution of years amongst the MedShake classes.
    """
    class_col = "medshake_class"
    diff_col = "medshake_difficulty"
    years = sorted(df["year_txt"].unique())

    # Bar plot using MedShake class,
    # One bar per year, with stacked classes
    bars_data = {n: [0] * len(LABEL_COLOURS.keys()) for n in years}
    bars_index = []
    for idx, label in enumerate(LABEL_COLOURS.keys()):
        _df = df[df[class_col] == label]
        bars_index.append(label)
        for n, count in _df["year_txt"].value_counts().items():
            bars_data[n][idx] = count
    bars_df = pd.DataFrame(data=bars_data, index=bars_index).T

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 4))
    bars_df.plot.bar(stacked=True, rot=0, color=LABEL_COLOURS.values(), ax=ax)
    ax.set_xlabel("Years")
    ax.set_ylabel("MedShake Class")
    ax.xaxis.set_tick_params(rotation=80)
    ax.legend(labels=LABEL_COLOURS.keys(), loc="upper left")

    # Save the figure
    fig = ax.get_figure()
    fig.suptitle(f"MCQ MedShake Class by Year ({class_col})")
    fig.savefig(figure_path.replace(".", "_class."), bbox_inches="tight")

    ############################################################################

    # Line plot using MedShake difficulty
    for plot_type in ("all", "by_class"):
        fig, ax = plt.subplots(figsize=(10, 4))
        df = df.sort_values(by="year_txt")
        suffix = "_difficulty"

        # Plot the mean by year without splitting by class
        if plot_type == "all":
            ax.plot(
                df.groupby(by="year_txt")[diff_col].mean(),
                marker=".", label=label, c=LABEL_COLOURS["hard"])
            fig.suptitle(f"MCQ Avg MedShake Difficulty by Year")
            suffix += "_all"

        # Plot means by year and class
        elif plot_type == "by_class":
            pd.pivot_table(
                df.reset_index(),
                index="year_txt",
                columns=class_col,
                values=diff_col,
                observed=True,
            ).plot(
                ax=ax, color=LABEL_COLOURS, legend=False, marker=".",
                xticks=range(len(years)))
            fig.suptitle(f"MCQ Avg MedShake Difficulty by Year and Class")
            fig.legend(loc="upper right")
            suffix += "_class"

        # Save the figure
        ax.set_xlabel("Year")
        ax.set_ylabel("MedShake Difficulty")
        ax.xaxis.set_tick_params(
            rotation=80, gridOn=True, grid_color="#EEEEEE", grid_dashes=(1, 2),
            grid_linewidth=1.5)
        fig.savefig(figure_path.replace(".", f"{suffix}."), bbox_inches="tight")


def main_plot_years():
    corpus_path = "data/test-medshake-score.json"
    df = load_corpus(corpus_path)
    plot_years(
        df,
        "output/plots/years/years_plot.png",
    )


def main(method_name: str, *args, **kwargs):
    from util import plot_questions
    method = getattr(plot_questions, method_name)
    if not method:
        raise f"Method '{method_name}' not found"
    return method(*args, **kwargs)


if __name__ == "__main__":
    import fire
    # fire.Fire(main_all)
    # fire.Fire(main_plot_tags)
    # fire.Fire(main_plot_topics)
    # fire.Fire(main_plot_years)
    fire.Fire(main)
