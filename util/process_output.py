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


def get_filename_pattern(finetuned_modifier: str = "?"):
    """
    Returns the filename pattern for log and output files.

    Parameters:
     - finetuned_modifier: Use "?" (default) to match all files, "{0}" to
       exclude output from finetuned models, or "" to only match those files.
    """
    path_re = r"(?P<model>.+?)" \
        r"_shots(?P<shots>\d+)" \
        r"_medshake" \
        r"(?P<finetuned>_finetuned)" f"{finetuned_modifier}" \
        r"_(?P<run>\d+)" \
        r"\.txt"
        # r"_prompt(?P<prompt>\d+)" \
    return re.compile(path_re)


def parse_path(path: str, pattern: re.Pattern = None) -> dict:
    """
    Parses file path and extracts run attributes form the file name.
    """
    pattern = pattern or get_filename_pattern()
    path_data = pattern.search(path)
    if not path_data:
        shots_nbr = is_finetuned = run_nbr = 0
    else:
        shots_nbr = int(path_data.group("shots"))
        run_nbr = int(path_data.group("run"))
        is_finetuned = int(path_data.group("finetuned") is not None)
    return {
        "shots": shots_nbr,
        "run": run_nbr,
        "finetuned": is_finetuned,
    }


def load_logs(paths: list[str]) -> list[dict]:
    """
    Loads log files and returns the list of extracted results found at the end.
    """
    data = []
    pattern = get_filename_pattern()

    for path in paths:
        print(f"\n==> {path} <==", file=sys.stderr)

        # Read run parameters from file name
        path_data = parse_path(path, pattern)

        # Read last lines of the file to find rates
        tail = subprocess.run(
            ["tail", "-n", "4", path], capture_output=True, text=True)
        print(tail.stdout, file=sys.stderr)
        lines = tail.stdout.splitlines()
        try:
            emr = float(lines[0].split(": ")[1])
            hamming = float(lines[1].split(": ")[1])
            emr_by_class = json.loads(
                "{"
                + lines[2]
                .split("{")[1]
                .replace("\'", "\"")
            )
            hamming_by_class = json.loads(
                "{"
                + lines[3]
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
            "emr_by_class": emr_by_class,
            "hamming_by_class": hamming_by_class,
        }
        print(json.dumps(results, indent=2), file=sys.stderr)
        data.append(results)
    return data


def load_outputs(paths: list[str], corpus_path: str) -> list[dict]:
    """
    Loads output files and returns the list of extracted results.
    """
    data = []
    with open(corpus_path, "r") as f:
        corpus = json.load(f)
    for path in paths:
        print(f"\n==> {path} <==", file=sys.stderr)

        # Read run parameters from file name
        path_data = parse_path(path)

        all_match = []
        all_hamming = []
        with open(path, "r") as f:
            for i, line in enumerate(f.readlines()):
                generated = line.strip().split(";")[1].split("|")
                expected = corpus[i]["correct_answers"]
                is_match = set(generated) == set(expected)
                hamming_rate = deft.hamming(generated, expected)
                all_match.append(is_match)
                all_hamming.append(hamming_rate)

        emr_by_class, hamming_by_class = get_average_by_difficulty(
            corpus,
            all_match,
            all_hamming,
        )

        results = {
            **path_data,
            "emr": np.average(all_match),
            "hamming": np.average(all_hamming),
            "emr_by_class": emr_by_class,
            "hamming_by_class": hamming_by_class,
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

    df.drop(inplace=True, columns=["emr_by_class", "hamming_by_class"])

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


def print_results(df: pd.DataFrame, head_only=True, split_rates=False) -> None:
    """
    Prints results from the given DataFrame, optionally split by rate
    (emr, hamming).
    """
    # Print full table
    with pd.option_context(
            "display.max_rows", None,
            "display.max_columns", None,
            "expand_frame_repr", None,
    ):
        # Keep only columns with rates
        df = df.filter(regex="emr|hamming")
        # Ignore columns with standard deviation by class
        df = df.drop(columns=list(df.filter(regex="(emr|hamming)(_.+)+_std")))

        if split_rates:
            for col, title in [("emr", "EMR:"), ("hamming", "\nHamming rate:")]:
                print(title)
                _df = df.filter(regex=col)
                if head_only:
                    print(_df.head())
                else:
                    print(_df)
        else:
            if head_only:
                print(df.head())
            else:
                print(df)


def plot_results(df: pd.DataFrame, use_finetuned: bool = False) -> None:
    """
    Create plots for EMR and Hamming rate of data grouping by shots.
    """
    suffix = "_finetuned" if use_finetuned else ""
    for col, title in [("emr", "EMR"), ("hamming", "Hamming rate")]:
        # Plot rate grouped by shot
        fig, ax = plt.subplots()
        df.boxplot(by="shots", column=col, ax=ax)
        fig.suptitle(None)
        ax.set_title(title)
        save_path = f"output/llama3/plots/{col}_by_shot{suffix}.png"
        fig.savefig(save_path)
        print(f"Plot saved in {save_path}", file=sys.stderr)

        # Plot all rates for each number of shots
        # df_groups = df.groupby(by="shots")
        # for shots, _ in df_groups:
        #     fig, ax = plt.subplots()
        #     _df = df[df["shots"] == shots].filter(regex=col)
        #     def renamer(col: str) -> None:
        #         parts = col.split("_")
        #         return col if len(parts) == 1 else "_".join(parts[1:])
        #     _df.rename(inplace=True, columns=renamer)
        #     _df.plot(ax=ax, kind="box", grid=True)
        #     ax.set_title(f"{title} {shots}-shot")
        #     fig.savefig(f"output/llama3/plots/{col}_by_class_shots{shots}{suffix}.png")


def main(use_finetuned: bool = True, force_reload: bool = False) -> None:
    basedir = "output/llama3/paper"
    suffix = "_finetuned" if use_finetuned else ""
    presaved_results = f"{basedir}/pre_processed_outputs{suffix}.json"

    if not force_reload and os.path.exists(presaved_results):
        with open(presaved_results) as f:
            results = json.load(f)
    else:
        finetuned_modifier = "{0}" if not use_finetuned else ""
        pattern = get_filename_pattern(finetuned_modifier=finetuned_modifier)
        paths = [
            os.path.join(basedir, f)
            for f in os.listdir(basedir)
            if pattern.match(f)
        ]
        # results = load_logs(paths)
        results = load_outputs(paths, "data/dev-medshake-score.json")
        with open(presaved_results, "w") as f:
            json.dump(results, f)

    df = get_results_dataframe(results, group_by_shots=True)

    plot_results(df, use_finetuned=use_finetuned)

    df = group_results_by_shots(df)
    print_results(df, split_rates=True, head_only=False)


if __name__ == "__main__":
    import fire
    fire.Fire(main)
