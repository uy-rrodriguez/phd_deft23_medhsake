"""
Scripts to compare Human vs LLM answers, generate plots, tables, etc.
"""

import json
import os
import re
import sys

from matplotlib import pyplot as plt
import matplotlib.text as mtext
import numpy as np
import pandas as pd
import torch

# Trick to import local packages when this script is run from the terminal
sys.path.append(os.path.abspath("."))

from deft import hamming, medshake_rate, generate_medshake_scores
from util.classify_questions import (
    load_corpus,
    CLASS_COL,
    LABEL_COLOURS,
    get_average_by_difficulty,
    load_corpus,
)
from util.markdown import save_params
from util.preprocess_data import corpus_with_metadata, student_rates
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


def get_model_params(
        model_name: str,
        model_logits_filename: str,
        output_basedir: str,
        output_filename: str,
) -> tuple[str, str, dict, str]:
    """
    Helper function to generate common parameters used to load model inference
    results, and to store output files in model-dependent directories.

    :returns:
        Tuple with path to load model scores, path to find model inference
        results, kwargs to filter model inference files, and path to store the
        methods output (graphs, README, etc.)
    """
    model_output_kwargs = {
        "regex_prompt_nbr": 2,
        "regex_shots_nbr": 2,
        "regex_finetuned": True,
        "regex_answer_txt": False,
    }
    base_logits_path = "output/model_scores/%s/" + model_logits_filename
    base_output = f"{output_basedir}/%s/{output_filename}"

    # LLM parameters depending on chosen model
    if (model_name == "llama3-8b"):
        model_output_dir = "output/llama3/tuned_002_20240731/"
        model_dir = "llama-3-8b-deft_002_20240731"
        model_output_kwargs["regex_shots_nbr"] = 3

    elif (model_name == "mistral-7b_full"):
        model_output_dir = "output/mistral/tuned_008_20241031/"
        model_dir = "mistral-7b-deft_008_20241031"
        model_output_kwargs["regex_shots_nbr"] = 4

    elif (model_name == "mistral-7b_letters"):
        model_output_dir = "output/mistral/tuned_017_20250304/base"
        model_dir = "mistral-7b-deft_017_20250304"
        model_output_kwargs["regex_shots_nbr"] = 3

    elif (model_name == "mistral-7b_300"):
        model_output_dir = "output/mistral/tuned_017_20250304/varying_300"
        model_dir = "mistral-7b-deft_017_20250304"
        model_output_kwargs["regex_shots_nbr"] = 3

    elif (model_name == "biomistral-7b"):   # <== RELANCER
        model_output_dir = "output/biomistral/tuned_011_20240912/"
        model_dir = "biomistral-7b-deft_011_20240912"
        model_output_kwargs["regex_prompt_nbr"] = 4

    elif (model_name == "apollo-7b"):
        model_output_dir = "output/apollo/tuned_013_20240920/"
        model_dir = "apollo-7b-deft_013_20240920"
        model_output_kwargs["regex_prompt_nbr"] = 4
        model_output_kwargs["regex_shots_nbr"] = 3

    else:
        raise ValueError(f"Model '{model_name}' not recognised.")

    model_logits_path = base_logits_path % model_dir
    output_path = base_output % model_dir

    return model_logits_path, model_output_dir, model_output_kwargs, output_path


def load_model_results_output(
        corpus_df: pd.DataFrame,
        model_output_dir: str,
        model_output_kwargs: dict = None,
        force_reload: bool = False,
) -> pd.DataFrame:
    """
    Helper to load model output files and calculate average scores.
    The "medshake" score is calculated from the actual responses given by the
    LLM to each question.
    """
    # Arguments used to load LLM output files
    model_kwargs = {
        # "regex_prompt_nbr": 2,
        # "regex_shots_nbr": 3,
        # "regex_finetuned": True,
        # "regex_answer_txt": False,
    }
    model_kwargs.update(model_output_kwargs or {})
    model_results_df = load_output_files_df(
        basedir=model_output_dir,
        corpus=corpus_df,
        pattern_kwargs=model_kwargs,
        force_reload=force_reload,
    )

    # Group results by ID (join all result files) and calculate average
    # Note: This DataFrame is indexed by ID
    model_results_df = model_results_df.groupby(by="id").mean()
    # Add MedShake∕Shannon class to LLM results for later use
    model_results_df = model_results_df.join(
        corpus_df.groupby("id").first()[CLASS_COL])
    return model_results_df


def _process_field_value(field: str, data: dict) -> float:
    """
    Helper to process model score fields, possibly applying additions or
    substractions over multiple fields.

    E.g.: "a+b-c" will calculate the result of adding the value of field "a" to
    "b", and then substract "c".
    """
    # Handle special value, a field that does not exist in the data
    if field == "LETTERS_ONLY":
        return sum(np.float16(v) for v in data["letters"].values())

    # Handle existent fields
    fields = re.split(r"[+-]", field)
    vals = []
    for f in fields:
        # Conversion to float to handle "-Inf"
        val = np.float16(data[f])
        is_substraction = re.match(f".*-{f}.*", field)
        if is_substraction:
            val = -1 * val
        vals.append(val)
    return sum(vals)


def load_model_logits(
        corpus: str | pd.DataFrame = "data/test-medshake-score.json",
        model_scores_path: str = "output/model_scores/test-model-scores.json",
        class_col: str = CLASS_COL,
        score_field: str = "letters_logp",
        readme_out_dir: str = None,
        length_field: str = None,
        softmax: bool = None,
        softmax_temp: float = None,  # 0.6 == Default in LLaMa-3-8b AutoModelForCausalLM.generate
        normalise_by_length: bool = None,  # Normalise with seq. length
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

    `normalise_by_length` is used when the length of the sequence is available,
    and normalises dividing the score by the length.

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
    if readme_out_dir is not None:
        save_params(
            os.path.join(readme_out_dir, "README.md"),
            "load_model_scores",
            model_scores_path=model_scores_path,
            score_field=score_field,
            length_field=length_field,
            softmax=softmax,
            softmax_temp=softmax_temp,
            normalise_by_length=normalise_by_length,
            normalise_softmax=normalise_softmax,
            normalise_letters=normalise_letters,
            normalise=normalise,
            perplexity=perplexity,
        )

    # Default temperature is 1 if not given
    softmax_temp = softmax_temp or 1

    with open(model_scores_path) as fp:
        model_probs = json.load(fp)

    # Make sure a field for the length is given when required
    requires_length = (normalise_by_length or normalise_softmax or perplexity)
    assert requires_length and length_field or not requires_length

    # Extract score and sequence length from given fields
    seq_lengths = {}
    for _id, v in model_probs.items():
        seq_lengths[_id] = {}
        for k, d in v.items():
            if score_field is not None:
                v[k] = _process_field_value(score_field, d)
            if length_field is not None:
                seq_lengths[_id][k] = _process_field_value(length_field, d)
    # print(pd.read_json(model_scores_path, orient="index"))

    corpus = load_corpus(corpus)

    llm_results = []
    for _, inst in corpus.iterrows():
        _id = inst["id"]
        correct_answers = " ".join(inst["correct_answers"])
        sample_probs = model_probs.get(_id)
        if not sample_probs:
            print(f"Probabilities not found for '{_id}'", file=sys.stderr)
            sample_probs = {}
            model_score = 0
        else:
            # Convert str "-inf" to a float
            for k, v in sample_probs.items():
                if type(v) == str:
                    sample_probs[k] = np.float16(v)

            # Get softmax probability
            if softmax:
                probs = torch.softmax(
                    torch.tensor(list(sample_probs.values())) / softmax_temp,
                    dim=0,
                ).tolist()
                sample_probs = {k: v for k, v in zip(sample_probs.keys(), probs)}

            # Normalise with the length of the sequence
            elif normalise_by_length:
                _len = seq_lengths[_id]
                sample_probs = {k: v/_len[k] for k, v in sample_probs.items()}

            # Get softmax probability after normalising with the length of the
            # sequence
            elif normalise_softmax:
                _len = seq_lengths[_id]
                normalised = [v/_len[k] for k, v in sample_probs.items()]
                # _sum = sum(normalised)
                # probs = [p/_sum for p in normalised]
                probs = torch.softmax(
                    torch.tensor(normalised) / softmax_temp,
                    dim=0,
                ).tolist()
                sample_probs = {k: v for k, v in zip(sample_probs.keys(), probs)}

            # Get softmax probability after normalising with the number of
            # letters in each combination
            elif normalise_letters:
                normalised = [v/(len(k.split()) + 1) for k, v in sample_probs.items()]
                # _sum = sum(normalised)
                # probs = [p/_sum for p in normalised]
                probs = torch.softmax(
                    torch.tensor(normalised) / softmax_temp,
                    dim=0,
                ).tolist()
                sample_probs = {k: v for k, v in zip(sample_probs.keys(), probs)}

            # Normalise log probabilities
            elif normalise:
                _sum = sum(sample_probs.values())
                probs = [v/_sum for v in sample_probs.values()]
                sample_probs = {k: v for k, v in zip(sample_probs.keys(), probs)}

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
                probs = [np.exp(-1/_len[k] * v) for k, v in sample_probs.items()]
                # Normalise to [0..1] ~ softmax
                _sum = sum(probs)
                probs = [p/_sum for p in probs]
                # probs = torch.softmax(
                #     torch.tensor(probs) / softmax_temp,
                #     dim=0,
                # ).tolist()
                sample_probs = {k: v for k, v in zip(sample_probs.keys(), probs)}

            model_score = sample_probs[correct_answers]

        llm_results.append({
            "id": _id,
            "emr": model_score,
            class_col: inst[class_col],
            "probs": sample_probs,
        })

    # Note: This DataFrame is indexed by ID
    llm_results_df = pd.DataFrame(
        llm_results,
        index=[inst["id"] for inst in llm_results])
    # print(llm_results_df)
    return llm_results_df


def plot_tags_topics(
        model_name: str = "mistral-7b_letters",
        score_field: str = "letters_logp",
        length_field: str = None,
        include_inference: bool = False,
        include_argmax: bool = False,
        plot_all_tags: bool = False,
) -> None:
    """
    Plot MedShake score of Humans vs LLM, splitting data by tags and topics.

    :param plot_all: (Default False). If True, all tags and columns are exported
    instead of a selected few (the more relevant).
    """
    # Source data can be based on "test" or "train+test"
    corpus_path = "data/train+test-with-medshake.json"
    data_output_path = "output/regression/train+test-regression-data.json"
    model_scores_filename = "train+test-logp_20250318.json"
    figure_basedir = "output/compare/model_scores"
    figure_filename = "train+test/letters_logp_soft/compare.png"
    if include_inference:
        corpus_path = "data/test-medshake-score.json"
        data_output_path = "output/regression/test-regression-data.json"
        model_scores_filename = "logp_20250306.json"
        figure_filename = "test/letters_logp_soft/compare.png"

    diff_col = "medshake_difficulty"
    bar_config = {
        "humans": {"label": "Humans", "colour": "#1F77B4"},
        "logits": {"label": "LLM logits", "colour": "#FF7F0E"},
    }
    if include_inference:
        bar_config["inference"] = {
            "label": "LLM inference", "colour": "#5EC962",
        }
    if include_argmax:
        bar_config["argmax"] = {
            "label": "LLM argmax", "colour": "#AB1F90",
        }
    bar_width = 0.2 if len(bar_config) > 2 else 0.3

    # LLM parameters depending on chosen model
    model_logits_path, model_output_dir, model_output_kwargs, output_path = \
        get_model_params(
            model_name, model_scores_filename, figure_basedir, figure_filename)

    # Create output directories
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Load data source with tags
    df = corpus_with_metadata(
        corpus_path=corpus_path,
        data_output_path=data_output_path,
        result_ignored_cols=None,
    )

    if not plot_all_tags:
        # Manually select tags and topics to plot
        columns_config = {
            # Load tag values from streamlit app config, ignoring the option N/A
            "tag_negation": tags_config.TAGS_OPTS_NEGATION[1:],
            "tag_mode": tags_config.TAGS_OPTS_MODE[1:],
            "tag_composition": tags_config.TAGS_OPTS_COMPOSITION[1:],
            "tag_intruder": tags_config.TAGS_OPTS_INTRUDER[1:],
            "tag_answer": tags_config.TAGS_OPTS_SINGLE[1:],
            "topic": ["chimieanalytique", "immunologie", "galénique",
                      "physiologie", "parasitologie", "mycologie",
                      "enzymologie", "bactériologie", "virologie"]
        }
        if include_inference:
            columns_config["topic"].remove("enzymologie")
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
    infer_scores_df = load_model_results_output(
        corpus_df=df,
        model_output_dir=model_output_dir,
        model_output_kwargs=model_output_kwargs,
    )

    # Load LLM rates from model scores
    logits_scores_df = load_model_logits(
        corpus=df,
        model_scores_path=model_logits_path,
        score_field=score_field,
        readme_out_dir=os.path.dirname(output_path),
        length_field=length_field,
        softmax=True,
        # softmax_temp=0.6,
        # normalise_letters=True,
        # normalise_softmax=True,
        # normalise=True,
        # perplexity=True,  # Perplexity with normalisation to 0..1
    )
    if include_argmax:
        with open(corpus_path) as fp:
            corpus_df = load_corpus(corpus_path)
        logits_argmax_df: pd.Series = logits_scores_df.apply(axis=1, func=
            lambda x:
                list(x["probs"].keys())[
                    np.array([v for v in x["probs"].values()]).argmax()
                ] == " ".join(corpus_df[corpus_df["id"] == x["id"]]["correct_answers"].item())
        )

    # Bar plots by class with average EMR
    scores = {k: [] for k in bar_config}
    for label in LABEL_COLOURS.keys():
        _df = df[df[CLASS_COL] == label]
        _logs_df = logits_scores_df[logits_scores_df.index.isin(_df["id"])]
        scores["humans"].append(1 - _df[diff_col].mean())
        scores["logits"].append(_logs_df["emr"].mean())
        if include_inference:
            _infer_df = infer_scores_df[infer_scores_df.index.isin(_df["id"])]
            scores["inference"].append(_infer_df["emr"].mean())
        if include_argmax:
            _argmax_df = logits_argmax_df[logits_argmax_df.index.isin(_df["id"])]
            scores["argmax"].append(_argmax_df.mean())
    fig, ax = plt.subplots()
    ax.set_title(f"EMR Humans vs LLM ({CLASS_COL})")
    ax.set_ylabel("Score")
    x = np.arange(len(LABEL_COLOURS))
    for i, (_bar, config) in enumerate(bar_config.items()):
        ax.bar(
            x + bar_width*i, scores[_bar], label=config["label"],
            color=config["colour"],
            width=bar_width, align="center")
    ax.set_xticks(x + (len(bar_config)-1) / 2 * bar_width)
    ax.set_xticklabels(list(LABEL_COLOURS.keys()))
    ax.legend()
    fig.savefig(output_path.replace(".", f"_by_class."), bbox_inches="tight")

    # Boxplot of scores, to visualise variation
    fig, ax = plt.subplots()
    fig.suptitle("EMR variance Humans vs LLM")
    ax.set_ylabel("Score")
    scores = {
        "humans": 1 - df[diff_col],
        "logits": logits_scores_df["emr"],
        "inference": infer_scores_df["emr"],
    }
    for i, (k, config) in enumerate(bar_config.items()):
        if k != "argmax":
            ax.boxplot(
                scores[k], positions=(i+1,), tick_labels=(config["label"],),
                boxprops={"color": config["colour"]},
                medianprops={"color": "#000"})
    fig.savefig(output_path.replace(".", f"_box."), bbox_inches="tight")

    # Plot score by tag value, for each tag of interest
    for base_col, col_values in columns_config.items():
        figsize = None
        if base_col == "topic" and plot_all_tags:
            figsize = (16, 4)
        fig, ax = plt.subplots(figsize=figsize)

        # Add a couple of bars per column possible value
        # E.g.: tag_negation no/yes
        scores = {k: [] for k in bar_config}
        for val in col_values:
            tag_col = f"{base_col}_{val}"
            print(f"*** {tag_col} ***")
            _df = df[df[f"{tag_col}"] == 1]
            _logs_df = logits_scores_df[logits_scores_df.index.isin(_df["id"])]
            scores["humans"].append(1 - _df[diff_col].mean())
            scores["logits"].append(_logs_df["emr"].mean())
            if include_inference:
                _infer_df = infer_scores_df[infer_scores_df.index.isin(_df["id"])]
                scores["inference"].append(_infer_df["emr"].mean())
            if include_argmax:
                _argmax_df = logits_argmax_df[logits_argmax_df.index.isin(_df["id"])]
                scores["argmax"].append(_argmax_df.mean())
            print("Difficulty", _df[diff_col].mean())
            for k in scores:
                print(f"{bar_config[k]['label']} score", scores[k][-1])

        # Bar plots of average for all classes
        x = np.arange(len(col_values))
        for i, (_bar, config) in enumerate(bar_config.items()):
            ax.bar(
                x + bar_width*i, scores[_bar], label=config["label"],
                color=config["colour"],
                width=bar_width, align="center")
        ax.set_xticks(x + (len(bar_config)-1) / 2 * bar_width)
        ax.set_xticklabels([val.capitalize() for val in col_values])
        if base_col == "topic":
            ax.xaxis.set_tick_params(rotation=40)

        # Setup plot axes
        for ax in (ax,):  #  (ax_bar, ax_line):
            ax.set_title(f"EMR Humans vs LLM ({base_col})")
            ax.set_ylabel("Score")

        # Bar plot
        ax.legend()

        # Save figures
        fig_path = output_path.replace(".", f"_{base_col}.")
        if base_col == "topic" and plot_all_tags:
            fig_path = fig_path.replace(".", f"_all.")
        fig.savefig(fig_path, bbox_inches="tight")


def plot_regressions(
        include_inference: bool = False,
        selected_only: bool = False,
):
    """
    Plots to compare linear regressions of Humans vs LLMs.
    """
    model_name = "mistral-7b-deft_017_20250304"
    figures_dir = "logits+inference" if include_inference else "logits"
    figure_name = "sel_feats" if selected_only else "all_feats"
    figure_path = f"output/compare/regression/{model_name}/{figures_dir}/{figure_name}.png"
    os.makedirs(os.path.dirname(figure_path), exist_ok=True)

    basedir = "output/regression/linear/20250320_train+test"
    bar_config = {
        "humans": {
            "label": "Humans", "colour": "#1F77B4",
            "coefs": f"{basedir}/humans/full_no_best/regression_coefs.json",
        },
        "logits": {
            "label": "LLM logits", "colour": "#FF7F0E",
            "coefs": f"{basedir}/logits/full_no_best/regression_coefs.json",
        },
        # "argmax": {"label": "LLM argmax", "colour": "#AB1F90"},
    }
    if include_inference:
        bar_config["inference"] = {
            "label": "LLM inference", "colour": "#5EC962",
            # Inference scores can only be calculated for the "test" corpus
            "coefs": "output/regression/linear/20250325_test/inference/full_no_best/regression_coefs.json",
        }

    bar_width = 0.3

    sel_feats_names = None
    # sel_feats_names = [
    #     "tag_answer_single", "tag_answer_undefined", "tag_answer_multiple",

    #     "topic_chimieanalytique", "topic_immunologie", "topic_galénique",
    #     "topic_physiologie", "topic_parasitologie", "topic_mycologie",
    #     "topic_enzymologie", "topic_bactériologie", "topic_virologie",
    #     "topic_statistiques",

    #     "np_count", "q_len", "avg_vp_words", "qa_len", "vp_count",
    #     "avg_word_count", "word_count",
    # ]

    feats_df = None
    for k, config in bar_config.items():
        coefs_output_path = config["coefs"]
        print(f"Loading coefficients '{coefs_output_path}'")
        regression_df = pd.read_json(
            coefs_output_path, orient="index", encoding="utf-8")
        coefs_cols = regression_df.filter(regex=r"coef_.+", axis=1).columns
        cv_coefs_df = regression_df[coefs_cols]
        cv_coefs_df = cv_coefs_df.drop("intercept", axis=0)

        # Calculate avg and std of results from cross-validation
        cv_coefs_avg = cv_coefs_df.mean(axis=1).rename(k)
        avg_df = pd.DataFrame(cv_coefs_avg)

        if feats_df is None:
            feats_df = avg_df
        else:
            feats_df = feats_df.merge(
                avg_df, left_index=True, right_index=True)

    if selected_only:
        means = feats_df.abs().mean()
        if not sel_feats_names:
            sel_feats_names = set()

            # Ignore years
            feats_df = feats_df.filter(regex=r"^(?!year_)", axis=0)
            # Ignore choice letters
            feats_df = feats_df.filter(regex=r"^(?!answer_[a-e])", axis=0)

            for col in feats_df.columns:
                # print(feats_df[feats_df[col] >= means[col]].index)
                sel_feats_names.update(
                    feats_df[feats_df[col].abs() >= means[col]].index)
            print(len(sel_feats_names), sel_feats_names)

        feats_df = feats_df.filter(sel_feats_names, axis=0)

    feats_df["order"] = feats_df.apply(
        lambda x:
            max([abs(x[col]) for col in feats_df.columns]),
        axis=1,
    )
    feats_df = feats_df.sort_values(by="order", ascending=False)
    print(feats_df)

    # Plot selected features
    fig, ax = plt.subplots(figsize=(12, 3) if selected_only else (26, 4))
    fig.suptitle(
        f"Humans vs LLMs: {'Selected' if selected_only else 'All'} features")
    ax.set_ylabel("Coefficient")
    ax.xaxis.set_tick_params(rotation=60 if selected_only else 80)
    ax.axhline(y=0, color="black", linestyle="-")
    ax.yaxis.set_tick_params(
        gridOn=True, grid_color="#333333", grid_dashes=(1, 2),
        grid_linewidth=0.5)
    ax.margins(x=0.01)

    x = np.arange(len(feats_df.index))
    for i, (k, config) in enumerate(bar_config.items()):
        ax.bar(
            x + bar_width*i, feats_df[k].values,
            label=config["label"], color=config["colour"],
            width=bar_width, align="center")
    ax.set_xticks(x + (len(bar_config)-1) / 2 * bar_width)
    ax.set_xticklabels(feats_df.index.tolist())
    ax.set_yticks(np.arange(-0.25, 0.25, step=0.05))

    ax.legend()
    fig.savefig(figure_path, bbox_inches="tight")


def calc_argmax_rates(
        corpus_path: str = "data/test-medshake-score.json",
) -> None:
    """
    Calculates the evaluation scores (EMR, MedShake, Hamming) for all models,
    based on their logp files, and taking the argmax as response chosen by the
    model.

    Stores a summary of scores per model, together with human and inference
    scores.
    """
    # Definition of relevant parameters
    summary_output_path = "output/compare/model_scores/eval_rates_summary.json"
    models = [
        # model_name            model_scores_filename
        ("llama3-8b",           "logp_20250218.json"),
        ("mistral-7b_full",     "logp_20250226.json"),
        ("mistral-7b_letters",  "logp_20250306.json"),
        ("mistral-7b_300",      "logp_20250306.json"),
        ("biomistral-7b",       "logp_20250227.json"),
        ("apollo-7b",           "logp_20250227.json"),
    ]

    summary_scores = {}
    for model_name, model_scores_filename in models:
        # LLM parameters depending on chosen model
        model_scores_path, llm_output_dir, llm_output_kwargs, output_path = \
            get_model_params(
                model_name, model_scores_filename,
                "output/compare/model_scores", "eval_rates/eval_rates.json",
            )

        # Load data source with tags
        corpus_df = load_corpus(corpus_path)

        # Include scores of human answers
        results = {
            "human": student_rates(corpus=corpus_path, print_results=False),
        }

        # Create output directories
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Load LLM rates from model output
        infer_scores_df = load_model_results_output(
            corpus_df=corpus_df,
            model_output_dir=llm_output_dir,
            model_output_kwargs=llm_output_kwargs,
        )

        # Include scores of LLM inference
        all_matches = []
        all_hamming = []
        all_medshake = []
        for _, inst in corpus_df.iterrows():
            sample_scores = infer_scores_df.loc[inst["id"]]
            all_matches.append(sample_scores["emr"])
            all_hamming.append(sample_scores["hamming"])
            all_medshake.append(sample_scores["medshake"])
        emr_by_class, hamming_by_class, medshake_by_class = \
                get_average_by_difficulty(
                    corpus_df, all_matches, all_hamming, all_medshake)
        results["inference"] = {
            "emr": np.average(all_matches),
            "hamming": np.average(all_hamming),
            "medshake": np.average(all_medshake),
            "emr_by_class": emr_by_class,
            "hamming_by_class": hamming_by_class,
            "medshake_by_class": medshake_by_class,
        }

        # Different model "probability" fields to calculate model accuracy
        # The answer choice corresponding to the argmax of each probability
        # field is used as the model response to the question
        score_fields = [
            "letters_logp",
            "seq_logp",
            "eos_logp",
            "prompt_logp+seq_logp",            # Full prompt
            "seq_logp-letters_logp+eos_logp",  # Sequence without choice letters
        ]

        for score_field in score_fields:
            # Load LLM rates from model scores
            llm_scores_df = load_model_logits(
                corpus=corpus_df,
                model_scores_path=model_scores_path,
                score_field=score_field,
                readme_out_dir=os.path.dirname(output_path),
            )

            all_matches = []
            all_hamming = []
            all_medshake = []
            for _, inst in corpus_df.iterrows():
                _id = inst["id"]
                expected = inst["correct_answers"]
                medshake_data = generate_medshake_scores(
                    correct_answers=expected,
                    medshake_scores=inst["medshake"],
                )

                # Choose the would-be "predicted" answer based on argmax of
                # scores
                sample_scores = llm_scores_df.loc[_id]
                sample_probs = pd.Series(sample_scores["probs"])
                predicted = sample_probs.index[sample_probs.argmax()].split()

                all_matches.append(predicted == expected)
                all_hamming.append(hamming(predicted, expected))
                all_medshake.append(medshake_rate(predicted, medshake_data))

            emr_by_class, hamming_by_class, medshake_by_class = \
                get_average_by_difficulty(
                    corpus_df, all_matches, all_hamming, all_medshake)

            llm_results = {
                "emr": np.average(all_matches),
                "hamming": np.average(all_hamming),
                "medshake": np.average(all_medshake),
                "emr_by_class": emr_by_class,
                "hamming_by_class": hamming_by_class,
                "medshake_by_class": medshake_by_class,
            }

            results[score_field] = llm_results

        summary_scores[model_name] = {
            field: {
                k: v for k, v in scores.items()
                if k in ("emr", "hamming", "medshake")
            }
            for field, scores in results.items()
        }

        # Save model scores
        with open(output_path, "w") as fp:
            json.dump(results, fp, indent=2)

    # Save summary of all models
    with open(summary_output_path, "w") as fp:
        json.dump(summary_scores, fp, indent=2)


def eval_rates_summary():
    """
    Generates LaTeX tables with the pre-calculated summary of scores per model.
    """
    summary_path = "output/compare/model_scores/eval_rates_summary.json"
    with open(summary_path) as fp:
        summary = json.load(fp)
    for model, data in summary.items():
        print("\nModel:", model)
        df = pd.DataFrame(data).T
        deltas = df \
            .apply(lambda x: [x[k] - x["inference"] for k in x.index]) \
            .apply(lambda x: [v if v != 0 else "" for v in x]) \
            .rename(columns=lambda k: f"$\Delta$\_{k}")
        df = df.join(deltas) \
            .rename(
                index=lambda k: "seq\_logp (no letters)"
                if k == "seq_logp-letters_logp+eos_logp"
                else k.replace("_", "\_"))
        df.columns.name = " " * 22
        print(
            df.style
            .format_index("{: <22}")
            .format_index("{: <18}", axis=1)
            .format("{: <18.3}")
            .to_latex(
                environment="table*",
                column_format="l|cccccc",
                caption=f"Scores of logits argmax for model {model}",
            )
        )


def main(method_name: str, *args, **kwargs):
    import inspect
    module = inspect.getmodule(main)
    method = getattr(module, method_name)
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
