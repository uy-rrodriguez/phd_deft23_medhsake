"""
Scripts to compare Human vs LLM answers, generate plots, tables, etc.
"""

import json
import os
import sys

from matplotlib import pyplot as plt
import matplotlib.text as mtext
import numpy as np
import pandas as pd
import torch

# Trick to import local packages when this script is run from the terminal
sys.path.append(os.path.abspath("."))

from util.classify_questions import load_corpus, LABEL_COLOURS
from util.analyse_questions import merge_with_metadata
from util.process_output import (
    gen_output_suffix,
    get_filename_pattern,
    load_output_files_df,
)
from st_tagging_tool import config as tags_config


class LegendTitle(object):
    """
    Custom handler to include subtitles in plot legends.

    https://stackoverflow.com/a/38486135
    """
    def __init__(self, text_props=None):
        self.text_props = text_props or {}
        super().__init__()

    def legend_artist(self, legend, orig_handle, fontsize, handlebox):
        x0, y0 = handlebox.xdescent, handlebox.ydescent
        title = mtext.Text(
            x0, y0,
            # r'\underline{' + orig_handle + '}',
            orig_handle,
            usetex=False, **self.text_props)
        handlebox.add_artist(title)
        return title


def load_model_results_output(
        corpus_with_tags: pd.DataFrame,
        corpus_path: str = "data/test-medshake-score.json",
        llm_output_dir: str = "output/llama3/tuned_002_20240731",
        class_col = "medshake_class",
        force_reload: bool = True,
) -> pd.DataFrame:
    """
    Helper to load model output files and calculate average scores.
    The "medshake" score is calculated from the actual responses given by the
    LLM to each question.
    """
    # Arguments used to load LLM output files
    llm_kwargs = {
        "regex_prompt_nbr": 2,
        "regex_shots_nbr": 2,
        "regex_finetuned": True,
        "regex_answer_txt": False,
    }
    suffix = gen_output_suffix(**llm_kwargs)
    presaved_llm_results = \
        f"{llm_output_dir}/raw_rates_outputs{suffix}_details.json"

    if not force_reload and os.path.exists(presaved_llm_results):
        llm_results_df = pd.read_json(presaved_llm_results, orient="records")
    else:
        pattern = get_filename_pattern(**llm_kwargs)
        print("Filename pattern:", pattern.pattern)
        paths = [
            os.path.join(llm_output_dir, f)
            for f in os.listdir(llm_output_dir)
            if pattern.match(f)
        ]
        print(f"Files found ({len(paths)}):", *paths, sep="\n")
        llm_results_df = load_output_files_df(paths, corpus_path, pattern)

        if not len(llm_results_df):
            raise "No output files were found when loading results data."
        with open(presaved_llm_results, "w") as f:
            llm_results_df.to_json(f, orient="records")

    # Group results by ID (join all result files) and calculate average
    # Note: This DataFrame is indexed by ID
    llm_results_df = llm_results_df.groupby(by="id").mean()
    # Add MedShake class to LLM results for later use
    llm_results_df = llm_results_df.join(
        corpus_with_tags.groupby("id").first()[class_col])
    return llm_results_df


def load_model_scores(
        corpus_path: str = "data/test-medshake-score.json",
        model_scores_path: str = "output/model_scores/test-model-scores.json",
        model_score_col = "medshake",
        class_col = "medshake_class",
        apply_softmax: bool = True,
        softmax_temp: float = 0.6,  # Default in LLaMa-3-8b AutoModelForCausalLM.generate
) -> pd.DataFrame:
    """
    Helper to load the model probabilities for each question and answer.

    This was extracted from the internal model logits, the probability that the
    model generated "<|end-of-text|>" given each possible combination of
    answers.

    The "medshake difficulty" for LLMs is based on the probability the model
    gives to the correct answer.

    `apply_softmax` as True is required when the scores file contains logits
    instead of probabilities.
    """
    with open(model_scores_path) as fp:
        model_probs = json.load(fp)

    # Pre-process Log-P output
    if "-logp" in model_scores_path:
        for v in model_probs.values():
            for k, d in v.items():
                v[k] = d["seq_log_prob"]
    # print(pd.read_json(model_scores_path, orient="index"))

    corpus = load_corpus(corpus_path)

    llm_results = []
    for _, inst in corpus.iterrows():
        _id = inst["id"]
        correct_answers = " ".join(inst["correct_answers"])
        model_inst = model_probs.get(_id)
        if not model_inst:
            print(f"Probabilities not found for '{_id}'", file=sys.stderr)
            model_score = 0
        else:
            if apply_softmax:
                # Convert str "-inf" to a float
                for k, v in model_inst.items():
                    if type(v) == str:
                        model_inst[k] = np.float16(v)
                # Get softmax probability
                probs = torch.softmax(
                    torch.tensor(list(model_inst.values())) / softmax_temp,
                    dim=0,
                ).tolist()
                model_inst = {k: v for k, v in zip(model_inst.keys(), probs)}
            model_score = model_inst[correct_answers]
        llm_results.append({
            "id": _id,
            model_score_col: model_score,
            class_col: inst[class_col],
        })

    # Note: This DataFrame is indexed by ID
    llm_results_df = pd.DataFrame(
        llm_results,
        index=[inst["id"] for inst in llm_results])
    # print(llm_results_df)
    return llm_results_df


def plot_tags_topics(
        corpus_path: str = "data/test-medshake-score.json",
        data_output_path: str = "output/analysis/regression-data.json",
        model_scores_path: str = "output/model_scores/llama3/llama-3-8b-deft_002_20240731-logp.json",
        figure_path: str = "output/compare/model_scores/logp/compare.png",
        # model_output_dir: str = "output/llama3/tuned_002_20240731",
        # figure_path: str = "output/compare/model_outputs/compare.png",
        plot_all: bool = False,
) -> None:
    """
    Plot MedShake score of Humans vs LLM, splitting data by tags and topics.

    :param plot_all: (Default False). If True, all tags and columns are exported
    instead of a selected few (the more relevant).
    """
    # Definition of relevant columns
    diff_col = "medshake_difficulty"
    class_col = "medshake_class"
    # llm_score_col = "medshake"
    llm_score_col = "emr"
    colours = ("#1f77b4", "#ff7f0e")  # Human: blue, LLM: orange

    # Load data source with tags
    df = merge_with_metadata(
        data_output_path=data_output_path,
        result_ignored_cols=None,
    )

    if not plot_all:
        # Manually select tags and topics to plot
        columns_config = {
            # Load tag values from streamlit app config, ignoring the option N/A
            "tag_negation": tags_config.TAGS_OPTS_NEGATION[1:],
            "tag_composition": tags_config.TAGS_OPTS_COMPOSITION[1:],
            "tag_positive": tags_config.TAGS_OPTS_POSITIVE[1:],
            "tag_answer": tags_config.TAGS_OPTS_SINGLE[1:],
            "topic": ("galénique", "chimieanalytique", "immunologie",
                      "physiologie")
        }
    else:
        # Extract all tags and topics to plot from data source
        columns_config = {}
        for col in df.filter(regex=r"(tag|topic)_.+").columns:
            col_parts = col.split("_")
            base, val = "_".join(col_parts[:-1]), col_parts[-1]
            if not base in columns_config:
                columns_config[base] = [val]
            else:
                columns_config[base].append(val)

    # Load LLM rates from model output
    # llm_results_df = load_model_results_output(
    #     df,
    #     corpus_path=corpus_path,
    #     llm_output_dir=model_output_dir,
    #     class_col=class_col,
    # )

    # Load LLM rates from model scores
    llm_results_df = load_model_scores(
        corpus_path=corpus_path,
        model_scores_path=model_scores_path,
        model_score_col=llm_score_col,
        class_col=class_col,
        apply_softmax=True,
        softmax_temp=1,
    )

    # Plot score by tag value, for each tag of interest
    for base_col, col_values in columns_config.items():
        figsize_bar = figsize_line = None
        if base_col == "topic" and plot_all:
            figsize_bar = (16, 4)
            figsize_line = (16, 11)
        fig_bar, ax_bar = plt.subplots(figsize=figsize_bar)
        fig_line, ax_line = plt.subplots(figsize=figsize_line)

        # Add a couple of bars per column possible value
        # E.g.: tag_negation no/yes
        for val in col_values:
            tag_col = f"{base_col}_{val}"
            print(f"*** {tag_col} ***")

            _df = df[df[f"{tag_col}"] == 1]
            _llm_df = llm_results_df[llm_results_df.index.isin(_df["id"])]

            # Bar plots of average for all classes
            human_score = 1 - _df[diff_col].mean()
            llm_score = _llm_df[llm_score_col].mean()
            print("Difficulty", _df[diff_col].mean())
            print("Human score", human_score)
            print("LLM score", llm_score)
            ax_bar.bar(
                [f"Human {val.capitalize()}", f"LLM {val.capitalize()}"],
                [human_score, llm_score],
                label=("Human", "LLM"), color=colours)

            # Line plots of average by class
            # ax_line.plot(
            #     _df.groupby(by=class_col)[diff_col]
            #         .mean()
            #         .apply(lambda v: 1 - v)
            #         .sort_index(key=lambda x: [
            #             list(LABEL_COLOURS.keys()).index(k)
            #             for k in x
            #         ]),
            #     label=f"Human {val.capitalize()}",
            # )
            _df = _df.groupby(by=class_col)[diff_col] \
                    .mean().apply(lambda v: 1 - v)
            ax_line.plot(
                LABEL_COLOURS.keys(),
                [
                    _df.loc[k] if k in _df.index else None
                    for k in LABEL_COLOURS
                ],
                label=f"Human {val.capitalize()}",
            )
            # ax_line.plot(
            #     _llm_df.groupby(by=class_col)[llm_score_col]
            #         .mean()
            #         .sort_index(key=lambda x: [
            #             list(LABEL_COLOURS.keys()).index(k)
            #             for k in x
            #         ]),
            #     label=f"LLM {val.capitalize()}",
            # )
            _llm_df = _llm_df.groupby(by=class_col)[llm_score_col].mean()
            ax_line.plot(
                LABEL_COLOURS.keys(),
                [
                    _llm_df.loc[k] if k in _llm_df.index else None
                    for k in LABEL_COLOURS
                ],
                label=f"LLM {val.capitalize()}",
            )

        # Setup plot axes
        for ax in (ax_bar, ax_line):
            ax.set_title(f"Humans vs LLM ({base_col})")
            ax.set_ylabel("MedShake score")

        # Bar plot
        ax_bar.legend(("Human", "LLM"))
        if base_col in ("tag_answer", "tag_mode"):
            ax_bar.xaxis.set_tick_params(rotation=40)
        elif base_col == "topic":
            ax_bar.xaxis.set_tick_params(rotation=80)

        # Line plot
        ax_line.set_xlabel("MedShake Class")
        ax_line.legend()
        # Sort labels in legend and add group titles
        handles, labels = fig_line.gca().get_legend_handles_labels()
        order = [labels.index(l) for l in sorted(labels)]
        handles=[handles[idx] for idx in order]
        labels=[
            " ".join(labels[idx].split()[1:])
            for idx in order
        ]
        handles.insert(0, "Humans:")
        labels.insert(0, "")
        handles.insert(len(handles) // 2 + 1, "LLM:")
        labels.insert(len(labels) // 2 + 1, "")
        ax_line.legend(
            handles, labels, loc=(1.05, 0),
            handler_map={str: LegendTitle()},
        )

        # Save figures
        fig_path = figure_path.replace(".", f"_{base_col}.")
        if base_col == "topic" and plot_all:
            fig_path = fig_path.replace(".", f"_all.")
        fig_bar.savefig(fig_path, bbox_inches="tight")
        fig_line.savefig(fig_path.replace(".", f"_cls."), bbox_inches="tight")


def main(method_name: str, *args, **kwargs):
    from util import compare_human_llm
    method = getattr(compare_human_llm, method_name)
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
