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

from util.classify_questions import (
    load_corpus,
    CLASS_COL,
    LABEL_COLOURS,
)
from util.analyse_questions import corpus_with_metadata
from util.markdown import save_params
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
    # Add MedShake∕Shannon class to LLM results for later use
    llm_results_df = llm_results_df.join(
        corpus_with_tags.groupby("id").first()[CLASS_COL])
    return llm_results_df


def load_model_scores(
        corpus_path: str = "data/test-medshake-score.json",
        model_scores_path: str = "output/model_scores/test-model-scores.json",
        params_out_dir: str = "output/compare/model_scores/logp",
        model_score_col: str = "medshake",
        class_col: str = CLASS_COL,
        score_field: str = None,
        length_field: str = None,
        softmax: bool = None,
        softmax_temp: float = None,  # 0.6 == Default in LLaMa-3-8b AutoModelForCausalLM.generate
        normalise_softmax: bool = None,  # Normalise with seq. length before softmax
        normalise_letters: bool = None,  # Normalise scores with the length of the choices
        normalise: bool = None,  # Normalise scores (v_i = v_i / sum_i_N(v_i))
        perplexity: bool = None,  # Calculate Perplexity (Pxty_i = e^(-v_i/len seq i))
) -> pd.DataFrame:
    """
    Helper to load the model probabilities for each question and answer.

    This was extracted from the internal model logits, the probability that the
    model generated "<|end-of-text|>" given each possible combination of
    answers.

    The "medshake difficulty" for LLMs is based on the probability the model
    gives to the correct answer.

    `softmax` is used when the scores file contains logits instead of
    probabilities.

    `normalise` is used when the file contains log probabilities, to bring them
    to the range 0..1, normalising by the sum of all values.

    `normalise_softmax` is used when the length of answer sequence is available,
    and normalises using this length to then apply softmax.

    `normalise_letters` is used when the file contains log probabilities of the
    choice letters only, and normalises using the number of letters in the
    combination.

    `perplexity` is used when the file contains log probabilities and the length
    of each sequence. The log prob. is already the sum of log P of each token in
    the sequence. Dividing by -1/N (number of tokens) and calculating the
    exponential gives us the perplexity.
    """
    # Save a README with some key execution parameters
    save_params(
        os.path.join(params_out_dir, "README.md"),
        "load_model_scores",
        model_scores_path=model_scores_path,
        score_field=score_field,
        length_field=length_field,
        softmax=softmax,
        softmax_temp=softmax_temp,
        normalise_softmax=normalise_softmax,
        normalise_letters=normalise_letters,
        normalise=normalise,
        perplexity=perplexity,
    )

    # Default temperature is 1 if not given
    softmax_temp = softmax_temp or 1

    with open(model_scores_path) as fp:
        model_probs = json.load(fp)

    # Extract score and sequence length from given fields
    seq_lengths = {}
    for _id, v in model_probs.items():
        seq_lengths[_id] = {}
        for k, d in v.items():
            if score_field is not None:
                v[k] = sum(d[f] for f in score_field.split("+"))
            if length_field is not None:
                seq_lengths[_id][k] = sum(d[f] for f in length_field.split("+"))
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
            # Convert str "-inf" to a float
            for k, v in model_inst.items():
                if type(v) == str:
                    model_inst[k] = np.float16(v)

            # Get softmax probability
            if softmax:
                probs = torch.softmax(
                    torch.tensor(list(model_inst.values())) / softmax_temp,
                    dim=0,
                ).tolist()
                model_inst = {k: v for k, v in zip(model_inst.keys(), probs)}

            # Get softmax probability after normalising with the length of the
            # sequence
            elif normalise_softmax:
                _len = seq_lengths[_id]
                normalised = [v/_len[k] for k, v in model_inst.items()]
                # _sum = sum(normalised)
                # probs = [p/_sum for p in normalised]
                probs = torch.softmax(
                    torch.tensor(normalised) / softmax_temp,
                    dim=0,
                ).tolist()
                model_inst = {k: v for k, v in zip(model_inst.keys(), probs)}

            # Get softmax probability after normalising with the number of
            # letters in each combination
            elif normalise_letters:
                normalised = [v/(len(k.split()) + 1) for k, v in model_inst.items()]
                # _sum = sum(normalised)
                # probs = [p/_sum for p in normalised]
                probs = torch.softmax(
                    torch.tensor(normalised) / softmax_temp,
                    dim=0,
                ).tolist()
                model_inst = {k: v for k, v in zip(model_inst.keys(), probs)}

            # Normalise log probabilities
            elif normalise:
                _sum = sum(model_inst.values())
                probs = [v/_sum for v in model_inst.values()]
                model_inst = {k: v for k, v in zip(model_inst.keys(), probs)}

            # Normalised Perplexity based on log P and sequence length.
            #
            # Our method is analogous to calculating the softmax of the average
            # log P of a token in the sequence:
            #   Avg_Log_P(S) = 1/N * Sum_1_N[logP(w_i|w1, ..., w_i-1)]
            #   Perplexity(S) = e^Avg_Log_P(S)
            #   => Normalised_Perp(S) = Perp(S) / Sum_1_M[Perp(S_j)]
            #
            # Softmax is calculated as follows:
            #   Softmax(x) = e^x / Sum_1_N[e^x_i]
            elif perplexity:
                # Calculate perplexity(x) = e^(-1/N * x)
                _len = seq_lengths[_id]
                probs = [np.exp(-1/_len[k] * v) for k, v in model_inst.items()]
                # Normalise to [0..1] ~ softmax
                _sum = sum(probs)
                probs = [p/_sum for p in probs]
                # probs = torch.softmax(
                #     torch.tensor(probs) / softmax_temp,
                #     dim=0,
                # ).tolist()
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
        model_scores_path: str = "output/model_scores/llama3/llama-3-8b-deft_002_20240731-logp_20250218.json",
        # model_scores_path: str = "output/model_scores/llama3/llama-3-8b-deft_002_20240731-hf_perp_20250220.json",
        figure_path: str = "output/compare/model_scores/llama-3-8b-deft_002_20240731/logp/compare.png",
        # model_output_dir: str = "output/llama3/tuned_002_20240731",
        # figure_path: str = "output/compare/model_outputs/20250218/compare.png",
        plot_all: bool = False,
        scores_boxplot_only: bool = False,
) -> None:
    """
    Plot MedShake score of Humans vs LLM, splitting data by tags and topics.

    :param plot_all: (Default False). If True, all tags and columns are exported
    instead of a selected few (the more relevant).
    """
    # Definition of relevant columns
    diff_col = "medshake_difficulty"
    # llm_score_col = "medshake"
    llm_score_col = "emr"
    colours = ("#1f77b4", "#ff7f0e")  # Human: blue, LLM: orange
    classes = list(LABEL_COLOURS.keys())

    # Load data source with tags
    df = corpus_with_metadata(
        data_output_path=data_output_path,
        result_ignored_cols=None,
    )

    if not plot_all:
        # Manually select tags and topics to plot
        columns_config = {
            # Load tag values from streamlit app config, ignoring the option N/A
            "tag_negation": tags_config.TAGS_OPTS_NEGATION[1:],
            "tag_mode": tags_config.TAGS_OPTS_MODE[1:],
            "tag_composition": tags_config.TAGS_OPTS_COMPOSITION[1:],
            "tag_intruder": tags_config.TAGS_OPTS_INTRUDER[1:],
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
    # )

    # Load LLM rates from model scores
    llm_results_df = load_model_scores(
        corpus_path=corpus_path,
        model_scores_path=model_scores_path,
        params_out_dir=os.path.dirname(figure_path),
        model_score_col=llm_score_col,
        score_field="seq_logp",
        # score_field="prompt_logp+seq_logp",
        # score_field="letters_logp",
        length_field="seq_len",
        # softmax=True,
        # softmax_temp=0.01,
        # normalise_letters=True,
        # normalise_softmax=True,
        # normalise=True,
        # perplexity=True,  # Perplexity with normalisation to 0..1
    )

    # Boxplot of scores, to visualise variation
    fig, ax = plt.subplots()
    fig.suptitle("Humas vs LLM (scores variation)")
    ax.set_ylabel("EMR Score")
    human_scores = 1 - df[diff_col]
    llm_scores = llm_results_df[llm_score_col]
    ax.boxplot(
        human_scores, positions=(1,), tick_labels=("Human",),
        boxprops={"color": colours[0]}, medianprops={"color": "#000"})
    ax.boxplot(llm_scores, positions=(2,), tick_labels=("LLM",),
        boxprops={"color": colours[1]}, medianprops={"color": "#000"})
    fig.savefig(figure_path.replace(".", f"_box."), bbox_inches="tight")

    # Boxplots by class
    _df = df.groupby(by=CLASS_COL)[diff_col].apply(lambda v: 1 - v)
    ax.clear()
    ax.boxplot(
        [_df.loc[k] if k in _df.index else None for k in classes],
        positions=[i*3+1 for i in range(len(classes))],
        tick_labels=LABEL_COLOURS.keys(),
        boxprops={"color": colours[0]}, medianprops={"color": "#000"})
    _df = llm_results_df.groupby(by=CLASS_COL)[llm_score_col].apply(lambda x: x)
    ax.boxplot(
        [_df.loc[k] if k in _df.index else None for k in classes],
        positions=[i*3+2 for i in range(len(classes))],
        tick_labels=classes,
        boxprops={"color": colours[1]}, medianprops={"color": "#000"})
    ax.xaxis.set_tick_params(rotation=40)
    for i in range(len(classes) - 1):
        ax.axvline(x=i*3+3, color="#EEEEEE", linestyle="--")
    fig.savefig(figure_path.replace(".", f"_box_cls."), bbox_inches="tight")

    if scores_boxplot_only:
        return

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
            _df = _df.groupby(by=CLASS_COL)[diff_col] \
                    .mean().apply(lambda v: 1 - v)
            ax_line.plot(
                classes,
                [
                    _df.loc[k] if k in _df.index else None
                    for k in classes
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
            _llm_df = _llm_df.groupby(by=CLASS_COL)[llm_score_col].mean()
            ax_line.plot(
                classes,
                [
                    _llm_df.loc[k] if k in _llm_df.index else None
                    for k in classes
                ],
                label=f"LLM {val.capitalize()}",
            )

        # Setup plot axes
        for ax in (ax_bar, ax_line):
            ax.set_title(f"Humans vs LLM ({base_col})")
            ax.set_ylabel("EMR Score")

        # Bar plot
        ax_bar.legend(("Human", "LLM"))
        if base_col in ("tag_answer", "tag_mode"):
            ax_bar.xaxis.set_tick_params(rotation=40)
        elif base_col == "topic":
            ax_bar.xaxis.set_tick_params(rotation=80)

        # Line plot
        ax_line.set_xlabel("Class")
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
