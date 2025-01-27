"""
Scripts to compare Human vs LLM answers, generate plots, tables, etc.
"""

import os
import sys

import pandas as pd
from matplotlib import pyplot as plt
import matplotlib.text as mtext

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
            usetex=True, **self.text_props)
        handlebox.add_artist(title)
        return title


def plot_tags_topics(
        corpus_path: str = "data/test-medshake-score.json",
        # tags_path: str = "data/tags-test-medshake-score.json",
        data_output_path: str = "output/analysis/regression-data.json",
        llm_output_dir: str = "output/llama3/tuned_002_20240731",
        figure_path: str = "output/compare/compare.png",
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
    llm_score_col = "medshake"
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

    # Enrich data with rates based on model output
    force_reload = True
    # arguments used to load LLM output files
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
    llm_results_df = llm_results_df.join(df.groupby("id").first()[class_col])

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
