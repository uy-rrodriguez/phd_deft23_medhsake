"""
Utility function to process and reformat the output of tests.
"""

import json
import os
import re
import subprocess
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from classify_questions import get_average_by_difficulty


# Hack to import deft when this script is run from the terminal
try:
    import deft
except ImportError:
    sys.path.append(os.path.abspath("."))
    import deft


GENERIC_RE = \
    r"(?P<model>.+(?P<finetuned>_tuned){tuned_modifier}.+?)" \
    r"_prompt(?P<prompt>{prompt_modifier})" \
    r".*" \
    r"_shots(?P<shots>\d+)" \
    r"(?P<with_answer_txt>_answertxt){answertxt_modifier}" \
    r"_(?P<run>\d+)" \
    r"\.txt"


RATE_TITLES = {
    "emr": "EMR",
    "hamming": "Hamming rate",
    "medshake": "MedShake rate",
}


RE_DEFAULT_ARGS = {
    "tuned_modifier": "?",
    "prompt_modifier": "\d+",
    "answertxt_modifier": "?",
}
FILENAME_REGEXS = {
    "default":
        GENERIC_RE.format(**RE_DEFAULT_ARGS),
    "finetuned":
        GENERIC_RE.format(**{**RE_DEFAULT_ARGS, "tuned_modifier": ""}),
    "with_prompt_nbr":
        GENERIC_RE.format(**{**RE_DEFAULT_ARGS,
                             "prompt_modifier": "{prompt_nbr}"}),
    "with_answer_txt":
        GENERIC_RE.format(**{**RE_DEFAULT_ARGS, "answertxt_modifier": ""}),
}


def get_filename_pattern(regex_key: str = "default", **regex_args: str):
    """
    Returns the filename pattern for log and output files.

    Parameters:
        - filename_regex: Key representing the regex to use, from those defined
            in `FILENAME_REGEXS`.
        - regex_args: (Optional) Parameters to build the regex if the template
            found is a string template that expects named parameters.
    """
    path_re = FILENAME_REGEXS[regex_key].format(**regex_args)
    return re.compile(path_re)


def parse_path(path: str, pattern: re.Pattern = None) -> dict:
    """
    Parses file path and extracts run attributes form the file name.
    """
    pattern = pattern or get_filename_pattern()
    path_data = pattern.search(path)
    if not path_data:
        return {}
    return {
        k: 0 if v is None else (int(v) if v.isdigit() else 1)
        for k, v in path_data.groupdict().items()
        if k not in ["model"]
    }


def load_logs(paths: list[str], pattern: str = None) -> list[dict]:
    """
    Loads log files and returns the list of extracted results found at the end.
    """
    data = []
    pattern = pattern or get_filename_pattern()

    for path in paths:
        print(f"\n==> {path} <==", file=sys.stderr)

        # Read run parameters from file name
        path_data = parse_path(path, pattern)

        # Read last lines of the file to find rates
        tail = subprocess.run(
            ["tail", "-n", "6", path], capture_output=True, text=True)
        print(tail.stdout, file=sys.stderr)
        lines = tail.stdout.splitlines()
        try:
            emr = float(lines[0].split(": ")[1])
            hamming = float(lines[1].split(": ")[1])
            medshake = float(lines[2].split(": ")[1])
            emr_by_class = json.loads(
                "{"
                + lines[3]
                .split("{")[1]
                .replace("\'", "\"")
            )
            hamming_by_class = json.loads(
                "{"
                + lines[4]
                .split("{")[1]
                .replace("\'", "\"")
            )
            medshake_by_class = json.loads(
                "{"
                + lines[5]
                .split("{")[1]
                .replace("\'", "\"")
            )
        except Exception as e:
            raise RuntimeError(
                f"Could not load rates at the end of file '{path}'", e)

        results = {
            **path_data,
            "emr": emr,
            "hamming": hamming,
            "medshake": medshake,
            "emr_by_class": emr_by_class,
            "hamming_by_class": hamming_by_class,
            "medshake_by_class": medshake_by_class,
        }
        print(json.dumps(results, indent=2), file=sys.stderr)
        data.append(results)
    return data


def load_outputs(paths: list[str], corpus_path: str,
                 pattern: str = None) -> list[dict]:
    """
    Loads output files and returns the list of extracted results.
    """
    pattern = pattern or get_filename_pattern()
    data = []
    with open(corpus_path, "r") as f:
        corpus = json.load(f)
    for path in paths:
        print(f"\n==> {path} <==", file=sys.stderr)

        # Read run parameters from file name
        path_data = parse_path(path, pattern)

        all_match = []
        all_hamming = []
        all_medshake = []
        with open(path, "r") as f:
            for i, line in enumerate(f.readlines()):
                try:
                    generated = line.strip().split(";")[1].split("|")
                except IndexError as e:
                    raise IndexError(f"Parsing failed in line '{line}'", e)
                instance = corpus[i]
                expected = instance["correct_answers"]
                is_match = set(generated) == set(expected)
                hamming_rate = deft.hamming(generated, expected)
                medshake_rate = deft.medshake_rate(generated, instance)
                all_match.append(is_match)
                all_hamming.append(hamming_rate)
                all_medshake.append(medshake_rate)

        emr_by_class, hamming_by_class, medshake_by_class = \
            get_average_by_difficulty(
                corpus,
                all_match,
                all_hamming,
                all_medshake,
            )

        results = {
            **path_data,
            "emr": np.average(all_match),
            "hamming": np.average(all_hamming),
            "medshake": np.average(all_medshake),
            "emr_by_class": emr_by_class,
            "hamming_by_class": hamming_by_class,
            "medshake_by_class": medshake_by_class,
        }
        print(json.dumps(results, indent=2), file=sys.stderr)
        data.append(results)
    return data


def get_results_dataframe(results: list[dict], group_by_shots=False) \
        -> pd.DataFrame:
    """
    Returns a DataFrame after pre-processing the given results.
    """
    df = pd.DataFrame(results)

    df_emr = pd.DataFrame(df["emr_by_class"].tolist())
    df_emr.rename(inplace=True, columns=lambda k: "emr_" + "_".join(k.split()))
    df = df.join(df_emr)

    df_ham = pd.DataFrame(df["hamming_by_class"].tolist())
    df_ham.rename(inplace=True, columns=lambda k: "hamming_" + "_".join(k.split()))
    df = df.join(df_ham)

    df_med = pd.DataFrame(df["medshake_by_class"].tolist())
    df_med.rename(inplace=True, columns=lambda k: "medshake_" + "_".join(k.split()))
    df = df.join(df_med)

    df.drop(inplace=True, columns=["emr_by_class", "hamming_by_class",
                                   "medshake_by_class"])

    # print(df.head())
    return df


def group_results_by_shots(df: pd.DataFrame) -> pd.DataFrame:
    """
    Groups the results by number of shots, calculating average and standard
    deviation.
    """
    df_groups = df.groupby(by="shots")
    df = df_groups.mean()

    # Include standard deviation
    df_std = df_groups.std()

    for i, (k, series) in enumerate(df_std.items()):
        df.insert(i * 2 + 1 , f"{k}_std", series)
    return df


def split_results_by_rate(df: pd.DataFrame, exclude_std_by_class: bool = True) \
        -> dict[str, pd.DataFrame]:
    """
    Splits the results in multiple DataFrames, one per measured rate (emr,
    hamming, medshake).

    The returned frames exclude columns not related to rates.

    Parameters:
        df: DataFrame with results to split.
        exclude_std_by_class: If True (default), the columns with standard
            deviation by MedShake class are also excluded.
    """
    prefixes = "|".join(RATE_TITLES.keys())
    # df = df.filter(regex=prefixes)
    # Ignore columns with standard deviation by class
    if exclude_std_by_class:
        df = df.drop(columns=list(df.filter(regex=f"({prefixes})(_.+)+_std")))
    return {
        # This will keep only columns with rates
        prefix: df.filter(regex=prefix)
        for prefix in RATE_TITLES.keys()
    }


def print_results(df: pd.DataFrame, head_only=True, split_rates=False) -> None:
    """
    Prints results from the given DataFrame, optionally split by rate
    (emr, hamming, medshake).
    """
    # Print full table
    with pd.option_context(
            "display.max_rows", None,
            "display.max_columns", None,
            "expand_frame_repr", None,
    ):
        if split_rates:
            df_by_prefix = split_results_by_rate(df)
            for prefix, _df in df_by_prefix.items():
                print(f"\n{RATE_TITLES[prefix]}:")
                if head_only:
                    print(_df.head())
                else:
                    print(_df)
        else:
            if head_only:
                print(df.head())
            else:
                print(df)


def plot_results(df: pd.DataFrame, suffix: str = "") -> None:
    """
    Create plots for EMR and Hamming rate of data grouping by shots.
    """
    for prefix, title in RATE_TITLES.items():
        # Plot rate grouped by shot
        fig, ax = plt.subplots()
        df.boxplot(by="shots", column=prefix, ax=ax)
        fig.suptitle(None)
        ax.set_title(title)
        save_path = f"output/llama3/plots/{prefix}_by_shot{suffix}.png"
        fig.savefig(save_path)
        print(f"Plot saved in {save_path}", file=sys.stderr)

        # Plot all rates for each number of shots
        # df_groups = df.groupby(by="shots")
        # for shots, _ in df_groups:
        #     fig, ax = plt.subplots()
        #     _df = df[df["shots"] == shots].filter(regex=prefix)
        #     def renamer(prefix: str) -> None:
        #         parts = prefix.split("_")
        #         return prefix if len(parts) == 1 else "_".join(parts[1:])
        #     _df.rename(inplace=True, columns=renamer)
        #     _df.plot(ax=ax, kind="box", grid=True)
        #     ax.set_title(f"{title} {shots}-shot")
        #     fig.savefig(f"output/llama3/plots/{prefix}_by_class_shots{shots}{suffix}.png")


def main(
        basedir: str = "output/llama3/paper",
        force_reload: bool = False,
        regex_key: str = "default",
        **regex_args: str,
) -> None:
    suffix = f"_{regex_key}" if regex_key != "default" else ""
    presaved_results = f"{basedir}/pre_processed_outputs{suffix}.json"

    if not force_reload and os.path.exists(presaved_results):
        with open(presaved_results) as f:
            results = json.load(f)
    else:
        pattern = get_filename_pattern(regex_key=regex_key, **regex_args)
        print("Filename pattern:", pattern)
        paths = [
            os.path.join(basedir, f)
            for f in os.listdir(basedir)
            if pattern.match(f)
        ]
        print("Files to process:", *paths, sep="\n")
        # results = load_logs(paths, pattern)
        results = load_outputs(paths, "data/dev-medshake-score.json", pattern)
        print("Results:", results)

        if not results:
            return
        with open(presaved_results, "w") as f:
            json.dump(results, f)

    df = get_results_dataframe(results, group_by_shots=True)

    plot_results(df, suffix=suffix)

    df = group_results_by_shots(df)
    print_results(df, split_rates=True, head_only=False)


if __name__ == "__main__":
    import fire
    fire.Fire(main)
