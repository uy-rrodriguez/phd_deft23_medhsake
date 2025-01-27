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
    r"(?P<model>[^_]+(?P<finetuned>_tuned.*?){tuned_modifier})" \
    r"(_prompt(?P<prompt>{prompt_nbr})){prompt_modifier}" \
    r".*" \
    r"(_shots(?P<shots>{shots_nbr})){shots_modifier}" \
    r"(?P<with_answer_txt>_answertxt){answertxt_modifier}" \
    r"_(?P<run>\d+)" \
    r"\.txt"


RATE_TITLES = {
    "medshake": "MedShake rate",
    "emr": "EMR",
    "hamming": "Hamming score",
}


def get_filename_pattern(
        regex_prompt_nbr: int = None,
        regex_no_prompt: bool = None,
        regex_finetuned: bool = None,
        regex_shots_nbr: int = None,
        regex_no_shots: bool = None,
        regex_answer_txt: bool = None,
):
    """
    Returns the filename pattern for log and output files.

    Parameters:
        - regex_prompt_nbr: (Optional) Number to only select files with this
            prompt number.
        - regex_finetuned: (Optional) Whether to include or exclude files for
            finetuned models (contain "tuned"). If not given, no filtering is
            done.
        - regex_answer_txt: (Optional) Whether to include or exclude files for
            shots with full answer text (contain "answertxt"). If not given, no
            filtering is done.
    """
    path_re = GENERIC_RE.format(**{
        "prompt_nbr":
            r"\d+" if regex_prompt_nbr is None
            else regex_prompt_nbr,
        "prompt_modifier":
            "{0}" if regex_no_prompt
            else "",
        "tuned_modifier":
            "?" if regex_finetuned is None
            else "" if regex_finetuned
            else "{0}",
        "shots_nbr":
            r"\d+" if regex_shots_nbr is None
            else regex_shots_nbr,
        "shots_modifier":
            "{0}" if regex_no_shots
            else "",
        "answertxt_modifier":
            "?" if regex_answer_txt is None
            else "" if regex_answer_txt
            else "{0}",
    })
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


def load_log_files(paths: list[str], pattern: str = None) -> list[dict]:
    """
    Loads log files and returns the list of extracted results found at the end.
    """
    data = []
    pattern = pattern or get_filename_pattern()

    for path in paths:
        # print(f"\n==> {path} <==", file=sys.stderr)

        # Read run parameters from file name
        path_data = parse_path(path, pattern)

        # Read last lines of the file to find rates
        tail = subprocess.run(
            ["tail", "-n", "6", path], capture_output=True, text=True)
        # print(tail.stdout, file=sys.stderr)
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
        # print(json.dumps(results, indent=2), file=sys.stderr)
        data.append(results)
    return data


def load_output_files(paths: list[str], corpus_path: str,
                 pattern: str = None) -> list[dict]:
    """
    Loads output files and returns the list of extracted results.
    """
    pattern = pattern or get_filename_pattern()
    data = []
    with open(corpus_path, "r") as f:
        corpus = json.load(f)
    for path in paths:
        # print(f"\n==> {path} <==", file=sys.stderr)

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
                medshake_data = instance.get("medshake", {})
                medshake_rate = deft.medshake_rate(generated, medshake_data)
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
        # print(json.dumps(results, indent=2), file=sys.stderr)
        data.append(results)
    return data


def load_output_files_df(
        paths: list[str],
        corpus_path: str,
        pattern: str = None,
) -> pd.DataFrame:
    """
    Loads output files and returns the results as a DataFrame.

    MedShake scores per question are calculated, but no average nor any other
    calculation is done on the data.
    """
    pattern = pattern or get_filename_pattern()
    data = []
    with open(corpus_path, "r") as f:
        corpus = json.load(f)
    for path in sorted(paths):
        # Read run parameters from file name
        path_data = parse_path(path, pattern)

        with open(path, "r") as f:
            for i, line in enumerate(f.readlines()):
                try:
                    question_id, generated = line.strip().split(";")
                    generated = generated.split("|")
                except IndexError as e:
                    raise IndexError(f"Parsing failed in line '{line}'", e)
                instance = corpus[i]
                expected = instance["correct_answers"]
                is_match = set(generated) == set(expected)
                hamming_rate = deft.hamming(generated, expected)
                medshake_data = instance.get("medshake", {})
                medshake_rate = deft.medshake_rate(generated, medshake_data)
                data.append({
                    "id": question_id,
                    **path_data,
                    "emr": 1 if is_match else 0,
                    "hamming": hamming_rate,
                    "medshake": medshake_rate,
                })

    # print(json.dumps(results, indent=2), file=sys.stderr)
    return pd.DataFrame(data)


def gen_output_suffix(
        regex_prompt_nbr: int = None,
        regex_shots_nbr: int = None,
        regex_finetuned: bool = None,
        regex_answer_txt: bool = None,
) -> str:
    """
    Builds a suffix for generated files (results cache, plots, etc.) based on
    the regex parameters of the output files loaded.
    """
    suffix = ""
    if regex_prompt_nbr is not None:
        suffix += f"_prompt{regex_prompt_nbr}"
    if regex_shots_nbr is not None:
        suffix += f"_shots{regex_shots_nbr}"
    if regex_finetuned is not None:
        suffix += f"_tuned{1 if regex_finetuned else 0}"
    if regex_answer_txt is not None:
        suffix += f"_answertxt{1 if regex_answer_txt else 0}"
    return suffix


def load_results(
        basedir: str,
        corpus_path: str,
        force_reload: bool = False,
        regex_prompt_nbr: int = None,
        regex_no_prompt: bool = None,
        regex_finetuned: bool = None,
        regex_no_shots: bool = None,
        regex_answer_txt: bool = None,
) -> list[dict]:
    """
    Loads and returns result data from output files found in the given
    directory, filtering by the regex parameters.

    If `force_reload` is False (default), results are loaded from a cached file,
    if available. Otherwise, results are loaded from the raw output files
    created when running the experiments.
    """
    suffix = gen_output_suffix(
        regex_prompt_nbr=regex_prompt_nbr,
        regex_finetuned=regex_finetuned,
        regex_answer_txt=regex_answer_txt,
    )
    presaved_results = f"{basedir}/pre_processed_outputs{suffix}.json"

    if not force_reload and os.path.exists(presaved_results):
        with open(presaved_results) as f:
            results: list[dict] = json.load(f)
    else:
        pattern = get_filename_pattern(
            regex_prompt_nbr=regex_prompt_nbr,
            regex_no_prompt=regex_no_prompt,
            regex_finetuned=regex_finetuned,
            regex_no_shots=regex_no_shots,
            regex_answer_txt=regex_answer_txt,
        )
        print("Filename pattern:", pattern.pattern, file=sys.stderr)
        paths = [
            os.path.join(basedir, f)
            for f in os.listdir(basedir)
            if pattern.match(f)
        ]
        print(f"Files found ({len(paths)}):", *paths, sep="\n", file=sys.stderr)
        # results = load_log_files(paths, pattern)
        results = load_output_files(paths, corpus_path, pattern)
        # print("Results:", results, file=sys.stderr)

        if not results:
            raise "No output files were found when loading results data."
        with open(presaved_results, "w") as f:
            json.dump(results, f)

    return results


def get_results_dataframe(results: list[dict]) -> pd.DataFrame:
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


def print_results(df: pd.DataFrame, head_only=True, split_rates=True) -> None:
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


def latex_print_results(
        df: pd.DataFrame, table_title: str = "",
        single_table: bool = False, highlight_top: bool = True,
) -> None:
    """
    Prints resulting rates from the given DataFrame as multiple LaTeX tables.

    If single_table is True, all rates are printed in a single table, split by
    an empty row.
    """
    template_table = """\
\\begin{{table}}[H]
\\centering
\\begin{{tabular}}{{@{{}}{alignments}@{{}}}}
{title}\
{content}\
\\end{{tabular}}
\\caption{{Rate results...}}
\\label{{table:res_...}}
\\end{{table}}
"""
    template_title = """\
\\\\
\\multicolumn{{8}}{{c}}{{\\textbf{{{title}}}}} \\\\
\\\\
"""
    template_rate = """\
\\multicolumn{{{num_columns}}}{{c}}{{{title}}} \\\\
\\toprule
{headers} \\midrule
{content} \\bottomrule
"""
    template_sep = """\
\\\\
"""

    def latex_clean_header(header: str) -> str:
        """
        Returns the given header cleaned to escape special characters.
        """
        header = re.sub(r"^\w+?_", r"", header)
        header = header.replace("_", "\\_")
        return header

    def latex_df_format_column(
            column: pd.Series, highlight_top: bool) -> pd.Series:
        """
        Given a Series corresponding to a column with numeric values, returns
        LaTeX code to format the numbers and highlight first and second highest
        values.
        """
        # Do not highlight shots and standard deviation
        if re.match(r"shots|.*std", column.name):
            # return column.apply(lambda v: f"{v:.6f}".rstrip("0").rstrip("."))
            return column.apply(lambda v: f"{v:.3g}")
        # Highlight the highest two if there are more than 2 rows, otherwise
        # highlight only the first
        n = 2 if len(column) > 2 else 1
        largest = column.nlargest(n).values
        def value_to_latex(v: any) -> str:
            # v_str = f"{v:.6f}".rstrip("0")
            v_str = f"{v:.3g}"
            if highlight_top:
                if v == largest[0]:
                    return f"{{\\bf|{v_str}}}"
                if n > 1 and v == largest[1]:
                    return f"{{\\ul|{v_str}}}"
            return f"{v_str}"
        return column.apply(value_to_latex)

    def latex_df_to_str(df: pd.DataFrame,
                        min_col_widths: list | None = None) -> str:
        """
        Returns a DataFrame with numeric content as LaTeX rows, highlighting
        first and second highest values, and cells with equal width per column.
        """
        df = df.rename(latex_clean_header, axis="columns")
        df = df.apply(latex_df_format_column, args=(highlight_top,))

        def build_formatter(idx, col):
            def formatter(v):
                max_len = df[col].str.len().max()
                if min_col_widths is not None:
                    max_len = max(max_len, min_col_widths[idx])
                    min_col_widths[idx] = max_len
                return f"{v: <{max_len}s}"
            return formatter

        formatters = {}
        for idx, col in enumerate(df.select_dtypes("object")):
            formatters[col] = build_formatter(idx, col)

        df_str = df.to_string(
            formatters=formatters,
            index=False,
            justify="left",
        )

        df_str = re.sub(r" ([^\s])", r" & \1", df_str)
        df_str = re.sub(r"$", r" \\\\", df_str, flags=re.MULTILINE)
        df_str = re.sub(r"\|", " ", df_str)
        return df_str

    def latex_rate_to_str(df: pd.DataFrame, rate_title: str,
                          min_col_widths: list | None = None) -> str:
        """
        Returns LaTeX rows with the data for a single rate.
        """
        headers = ["shots"] + [h for h in df]
        content = []
        for shots, rates_series in df.iterrows():
            content.append([shots] + [r for r in rates_series])

        df = pd.DataFrame(columns=headers, data=content)
        tabular = latex_df_to_str(df, min_col_widths)
        tabular_lines = tabular.splitlines()

        return template_rate.format(
            num_columns=len(headers),
            title=rate_title,
            headers=tabular_lines[0],
            content="\n".join([l for l in tabular_lines[1:]]),
        )

    # Process results rate by rate and print them as LaTeX tables
    df_by_rate = split_results_by_rate(df)
    num_columns = len(next(iter(df_by_rate.values())).columns) + 1

    # When printing a single table, to make the columns of all rates the same
    # width, we pass the widths of the last loop to the next
    min_col_widths = [0] * num_columns if single_table else None
    # A second pass is needed to re-generate tables adjusting the column width
    num_generations = 2 if single_table else 1

    latex_by_rate = {}
    for _ in range(num_generations):
        for prefix, _df in df_by_rate.items():
            latex_by_rate[prefix] = latex_rate_to_str(
                _df, RATE_TITLES[prefix], min_col_widths)

    default_args = {
        "title":
            template_title.format(title=table_title)
            if table_title else "",
        "alignments": "l" * num_columns,
    }
    if single_table:
        result = template_table.format(
            **default_args,
            content=template_sep.join([s for s in latex_by_rate.values()]),
        )
    else:
        result = "\n".join([
            template_table.format(
                **default_args,
                content=rate_content,
            )
            for rate_content in latex_by_rate.values()
        ])

    print("LaTeX table:")
    print(result)


def box_plot_results(df: pd.DataFrame, basedir: str, suffix: str) -> None:
    """
    Create plots for EMR and Hamming score of data grouping by shots.
    """
    for prefix, title in RATE_TITLES.items():
        # Plot rate grouped by shot
        fig, ax = plt.subplots()
        df.boxplot(by="shots", column=prefix, ax=ax)
        fig.suptitle(None)
        ax.set_title(title)
        save_path = f"{basedir}/plots/{prefix}_by_shot{suffix}.png"
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
        basedir: str,
        corpus_path: str = "data/test-medshake-score.json",
        force_reload: bool = False,
        regex_prompt_nbr: int = None,
        regex_no_prompt: bool = None,
        regex_finetuned: bool = None,
        regex_no_shots: bool = None,
        regex_answer_txt: bool = None,
        highlight_top: bool = True,
) -> None:
    results = load_results(
        basedir=basedir,
        corpus_path=corpus_path,
        force_reload=force_reload,
        regex_prompt_nbr=regex_prompt_nbr,
        regex_no_prompt=regex_no_prompt,
        regex_finetuned=regex_finetuned,
        regex_no_shots=regex_no_shots,
        regex_answer_txt=regex_answer_txt,
    )

    df = get_results_dataframe(results)

    # suffix = gen_output_suffix(
    #     regex_prompt_nbr=regex_prompt_nbr,
    #     regex_finetuned=regex_finetuned,
    #     regex_answer_txt=regex_answer_txt,
    # )
    # box_plot_results(df, basedir=basedir, suffix=suffix)

    df = group_results_by_shots(df)
    # print_results(df, split_rates=True, head_only=False)

    latex_print_results(df, single_table=True, table_title="Results of model X",
                        highlight_top=highlight_top)


if __name__ == "__main__":
    import fire
    fire.Fire(main)
