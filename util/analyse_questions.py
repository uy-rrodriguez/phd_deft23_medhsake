"""
Utility functions to analyse the difficulty of questions, calculated from human
answers or model responses, in terms of question length, lexical structure, etc.
"""

import json
import os
import sys

from itertools import chain, combinations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

# Trick to import local packages when this script is run from the terminal
sys.path.append(os.path.abspath("."))

import st_tagging_tool
from classify_questions import load_corpus, LABEL_COLOURS
# from process_output import get_filename_pattern
from st_tagging_tool.config import TAGS_CONFIG


CAT_BOOL = pd.CategoricalDtype(categories=[0, 1])


def diff_test_files():
    """
    Compares the files "test.json" an d "test-medshake-score.json" to find
    discrepancies  between the expected correct answers. Both files should
    contain the same questions and answers.
    """
    with open("data/test.json", "r") as f:
        data_test = json.load(f)
    with open("data/test-medshake-score.json", "r") as f:
        data_medshake = json.load(f)

    print("Length Test:", len(data_test), "vs MedShake:", len(data_medshake))

    # Compare questions
    id_test_set = set([x["id"] for x in data_test])
    q_test_set = set([x["question"] for x in data_test])
    id_med_set = set([x["id"] for x in data_medshake])
    q_med_set = set([x["question"] for x in data_medshake])
    id_test_minus_med = id_test_set - id_med_set
    id_med_minus_test = id_med_set - id_test_set
    # q_test_minus_med = q_test_set - q_med_set
    # q_med_minus_test = q_med_set - q_test_set

    print("IDs in Test but not in MedShake:", len(id_test_minus_med))
    if id_test_minus_med:
        for i in id_test_minus_med:
            print(i)
        print()
    print("IDs in MedShake but not in Test:", len(id_med_minus_test))
    if id_med_minus_test:
        for i in id_med_minus_test:
            print(i)
        print()

    # print("Questions in Test but not in MedShake:", len(q_test_minus_med))
    # if q_test_minus_med:
    #     for q in q_test_minus_med:
    #         print(q)
    #     print()
    # print("Questions in MedShake but not in Test:", len(q_med_minus_test))
    # if q_med_minus_test:
    #     for q in q_med_minus_test:
    #         print(q)
    #     print()

    # Compare answers of shared questions
    intersect = id_test_set & id_med_set
    diff_answers_count = 0
    for i in intersect:
        test = next(x for x in data_test if x["id"] == i)
        med = next(x for x in data_medshake if x["id"] == i)
        a_test = test["correct_answers"]
        a_med = med["correct_answers"]
        if a_test != a_med:
            diff_answers_count += 1
            print("Different answers for question:", i)
            print("Question:", med["question"])
            print("Answers Test:", a_test)
            print("Answers MedShake:", a_med)

    print("Total different answers found:", diff_answers_count)


def calc_qa_lengths(df: pd.DataFrame, percent: bool = True) -> pd.DataFrame:
    """
    Enriches the data with lengths of question, question and answers, and
    average answer length.

    If "percent" is True (default) a percentage relative to the largest length
    is returned instead of the real length.
    """
    # Calculate lengths (final length value is a percentage calculated relative
    # to the max length of all questions)
    df["q_len"] = df["question"].apply(lambda x: len(x))
    df["qa_len"] = df.apply(
        lambda x: \
            len(x["question"]) \
            + sum([len(a) for a in x["answers"].values()]),
        axis=1,
    )
    df["a_avg_len"] = df["answers"].apply(
        lambda x: np.mean([len(a) for a in x.values()])
    )
    if percent:
        max_q_len = df["q_len"].max()
        max_qa_len = df["qa_len"].max()
        max_a_avg_len = df["a_avg_len"].max()
        df["q_len"] = df["q_len"].apply(lambda x: x / max_q_len)
        df["qa_len"] = df["qa_len"].apply(lambda x: x / max_qa_len)
        df["a_avg_len"] = df["a_avg_len"].apply(lambda x: x / max_a_avg_len)
    return df


def calc_first_last_words(
        df: pd.DataFrame,
        n_first_words: int = 1,
) -> pd.DataFrame:
    """
    Enriches the data adding the first word of the question and the last char.
    """
    df[["first_word", "last_char"]] = df["question"].apply(
        lambda q: pd.Series([
            # " ".join(re.sub(r"[,;.]", "", q).split()[:n_first_words]),
            " ".join(q.lower().split()[:n_first_words]),
            q[-1] if q[-1] in ("?", ":") else "<other>",
        ])
    )
    return df


# def load_output_files(
#         paths: list[str],
#         corpus_path: pd.DataFrame,
#         pattern: str = None,
# ) -> list[dict]:
#     """
#     Loads output files and returns the list of extracted results.
#     """
#     pattern = pattern or get_filename_pattern()
#     with open(corpus_path, "r") as f:
#         corpus = json.load(f)
#     match = [[] for _ in range(len(corpus))]
#     hamming = [[] for _ in range(len(corpus))]
#     medshake = [[] for _ in range(len(corpus))]
#     for path in paths:
#         with open(path, "r") as f:
#             for i, line in enumerate(f.readlines()):
#                 try:
#                     generated = line.strip().split(";")[1].split("|")
#                 except IndexError as e:
#                     raise IndexError(f"Parsing failed in line '{line}'", e)
#                 instance = corpus[i]
#                 expected = instance["correct_answers"]
#                 is_match = set(generated) == set(expected)
#                 hamming_rate = deft.hamming(generated, expected)
#                 medshake_data = instance.get("medshake", {})
#                 medshake_rate = deft.medshake_rate(generated, medshake_data)
#                 match[i].append(is_match)
#                 hamming[i].append(hamming_rate)
#                 medshake[i].append(medshake_rate)

#     results = {
#         "emr": [np.average(x) for x in match],
#         "hamming": [np.average(x) for x in hamming],
#         "medshake": [np.average(x) for x in medshake],
#     }
#     return results


def extract_ngrams(
        df: pd.DataFrame,
        output_path: str,
        ngram_lens: tuple[str] = (1, 2, 3),
):
    """
    Extract n-grams from the questions and save them in a separate file.

    By default, generates k-grams for k = 1, 2, 3.
    """
    print(f"About to extract the following n-grams: {ngram_lens}")

    from nltk import ngrams
    from nltk.tokenize import word_tokenize

    def get_ngrams(text):
        # Normalise
        text = text.lower().replace('"', "")
        # Extract tokens
        tokens = word_tokenize(text, language="french")
        # Add token at end of question
        tokens.append("<eos>")
        n_grams = {
            n: [" ".join(v) for v in ngrams(tokens, n)]
            for n in ngram_lens
        }
        return n_grams

    result = {
        inst["id"]: {
            "id": inst["id"],
            **{
                gram: 1
                for k_grams in get_ngrams(inst["question"]).values()
                for gram in sorted(set(k_grams))
            }
        }
        for _, inst in tqdm(df.iterrows())
    }

    with open(output_path, "w") as fp:
        json.dump(result, fp, indent=4, ensure_ascii=False)


def main_extract_ngrams():
    print("\nN-gram extraction")
    corpus_path = "data/test-medshake-score.json"
    df = load_corpus(corpus_path)
    extract_ngrams(df, "data/ngrams-test-medshake-score.json")


def multiplex_column(
        df: pd.DataFrame, col: str, new_prefix: str, drop_col: bool = True,
) -> None:
    """
    Expand one column into multiple columns, one per possible value in the
    corpus, with values in {NaN, 1}.

    The new column for a given possible value will be equal to 1 when the value
    under the original column "col" is equal to it.

    E.g.: for column negation = "no", three columns will be created:
        tag_negation_na = NaN, tag_negation_no = 1, tag_negation_yes = NaN
    """
    values = sorted(df[col].unique())
    for v in values:
        df[f"{new_prefix}_{v}"] = df[col].map({v: 1}).astype(CAT_BOOL)
    if drop_col:
        df.drop(col, axis=1, inplace=True)


def multiplex_column_list(
        df: pd.DataFrame, col: str, new_prefix: str, drop_col: bool = True,
) -> None:
    """
    Expand one column into multiple columns. Equivalent to `multiplex_column`
    but for columns where the values are lists.

    The new column for a given possible value will be equal to 1 when the list
    under the original column "col" contains it.

    E.g.: for column topics = ["A", "B"]:
        topic_A = 1, topic_B = 1, topic_C = NaN
    """
    values = sorted(set(
        chain.from_iterable(df[col].tolist())
    ))
    for v in values:
        df[f"{new_prefix}_{v}"] = df[col].map(
            lambda x: 1 if v in x else np.nan
        ).astype(CAT_BOOL)
    if drop_col:
        df.drop(col, axis=1, inplace=True)


def merge_with_metadata(
        df: pd.DataFrame,
        tags_path: str,
        ignored_tags: tuple[str] = ["tag_highlight"],
        ngrams_path: str | None = None,
        include_qa_lengths: bool = False,
        include_first_last_words: bool = False,
        data_output_path: str | None = None,
        force_reload: bool = False,
        result_filter_cols: bool = True,
        result_ignored_cols: list[str] = ["question", "medshake_difficulty"],
):
    """
    Enriches the source data with the given tags and n-grams.

    Pass "clean_for_regression" = True if the data is used for algorthms such as
    tSNE, Random Forest, or Linear Regression. This will remove non numeric
    columns and MedShake and Shannon classes.
    """
    # Try to load existing file
    if not force_reload and data_output_path and os.path.exists(data_output_path):
        print(f"Loading from existent file '{data_output_path}'")
        df = pd.read_json(data_output_path)
    else:
        # Pre-process source data, add metadata and remove unnecessary columns
        df = df.copy()

        # Add metadata from questions and answers
        if include_qa_lengths:
            df = calc_qa_lengths(df, percent=False)

        # Extract first and last words and expand to multiple columns, one per
        # possible word, with values in {0, 1} where 1 means the question
        # starts/ends with the word.
        if include_first_last_words:
            df = calc_first_last_words(df)
            multiplex_column(df, "first_word", "first")
            multiplex_column(df, "last_char", "last")
            df.rename(
                inplace=True,
                columns=lambda k: "last_other" if k == "last_<other>" else k)

        # Transform correct_answers into a string, then into a categorical type
        multiplex_column_list(df, "correct_answers", "answer")

        # Expand year to multiple columns
        multiplex_column(df, "year_txt", "year")

        # Expand topics list to multiple columns
        df["topics"] = df["topics"].apply(
            # Clean topics and keep first when there are many in one line
            lambda x: [
                v.split("/")[0].replace(" ", "").replace(".", "")
                for v in x
            ]
        )
        multiplex_column_list(df, "topics", "topic")

        # Drop unecessary columns
        drop_cols_re = (
            r"^(answers|subject_name|year|medshake|.*_colour(_q?cut)?"
            r"|shannon_.*)$"
        )
        df.drop(df.filter(regex=drop_cols_re).columns, axis=1, inplace=True)

        # Load tags
        #
        # Create one column per possible tag value and apply a boolean value
        # from {0, 1}, where 1 indicates the tag's value for a given question is
        # equal to the corresponding option.
        # E.g.: if negation=no => negation_na=0, negation_no=1, negation_yes=0
        print(f"Loading tags from '{tags_path}'")
        df_tags = pd.read_json(tags_path, orient="index")
        df_tags.drop(ignored_tags, axis=1, inplace=True)
        tag_cols = df_tags.columns.drop("id").to_list()
        for col in tag_cols:
            df_tags[col] = df_tags[col].apply(
                lambda x: x.replace("/", "").replace(" ", ""))
            multiplex_column(df_tags, col, col)

        # Load n-grams
        if ngrams_path:
            print(f"Loading n-grams from '{ngrams_path}'")
            single_quote = "'"
            df_ngrams = pd.read_json(ngrams_path, orient="index")
            df_ngrams.rename(
                inplace=True,
                columns=lambda k: f"ngram_{k}" if k != "id" else k)

        # Join DataFrames
        df = df.merge(
            df_tags, on="id", validate="one_to_one",
            suffixes=("", "--tags"), copy=False)
        if ngrams_path:
            df = df.merge(
                df_ngrams, on="id", validate="one_to_one",
                suffixes=("", "--ngrams"), copy=False)
        df.drop(
            df.filter(regex="id--(tags|ngrams)").columns, axis=1, inplace=True)

        if data_output_path:
            with open(data_output_path, "w") as fp:
                print(f"Saving data to '{data_output_path}'")
                df_dict = [
                    {
                        k: v
                        for k, v in d.items()
                        if not pd.isna(v)
                    }
                    for d in df.to_dict(orient="records")
                ]
                # print(json.dumps(df_dict, indent=4))
                json.dump(df_dict, fp, indent=4, ensure_ascii=False)

    # Replace NA with zeros in columns where NA values are expected
    cols_na = df.filter(
        regex=r"(first|last|answer|year|topic|tag|ngram)_.+"
    ).columns
    df[cols_na] = df[cols_na].fillna(0).astype("int")

    # For selected feature groups (e.g. year), remove columns where there
    # are less than 3 samples with a value for it (e.g. year_2017 if only
    # one sample is for that year).
    if result_filter_cols:
        # cols_filter = df.filter(
        #     regex=r"(first|last|answer|year|topic|tag|ngram)_.+"
        # ).columns
        cols_filter = cols_na
        sums = df[cols_filter].astype("float").sum()
        cols_drop = sums[sums < 3].index
        # print(to_drop)
        df.drop(cols_drop, axis=1, inplace=True)

    # Apply appropriate dtypes
    df["medshake_class"] = df["medshake_class"].astype(str)

    # Update category of difficulty classes so the index starts at 1
    # class_cols = df.filter(regex=r"(medshake|shannon)_class(_q?cut)?").columns
    # df[class_cols] = df[class_cols].astype(
    #     pd.CategoricalDtype(
    #         categories=[0] + list(LABEL_COLOURS.keys()), ordered=True))

    # Correct answers combined into a string
    # choices = ["a", "b", "c", "d", "e"]
    # df["correct_answers"] = df["correct_answers"].astype(
    #     pd.CategoricalDtype(
    #         categories=[
    #             "".join(comb)
    #             for i in range(1, len(choices) + 1)
    #             for comb in combinations(choices, i)
    #         ],
    #         ordered=True,
    #     )
    # )

    # Convert categorical values to int equivalent
    for k, t in df.dtypes.items():
        if t == "category":
            df[k] = df[k].cat.codes

    # Remove columns not needed by the algorithm to follow
    if result_ignored_cols:
        df.drop(result_ignored_cols, axis=1, inplace=True)

    # print(df)
    # print(df.columns.tolist())
    return df


def main_merge_with_metadata(
        force_reload: bool = False,
):
    print("\nMerge with metadata")
    corpus_path = "data/test-medshake-score.json"
    df = load_corpus(corpus_path)
    df = merge_with_metadata(
        df,
        tags_path="data/tags-test-medshake-score.json",
        ngrams_path="data/ngrams-test-medshake-score.json",
        data_output_path="output/analysis/random-forests-data.json",
        force_reload=force_reload,
    )
    print(df.head())
    # print(df[df.filter(regex=r"(first)_.*").columns])


def tsne(
        df: pd.DataFrame,
        tags_path: str,
        ngrams_path: str,
        data_output_path: str,
        figure_path: str,
        force_reload: bool = False,
        force_reload_embeddings: bool = False,
):
    """
    Executes the tSNE algorithm for dimension reduction and clustering over the
    source data, enriched with the given tags and n-grams.
    """
    df_rich = merge_with_metadata(
        df,
        tags_path=tags_path,
        ngrams_path=ngrams_path,
        data_output_path=data_output_path,
        force_reload=force_reload,
        result_ignored_cols = ["id", "question", "medshake_difficulty",
                               "medshake_class"],
    )

    # tSNE algorithm
    print("Running tSNE algorithm")
    from sklearn.manifold import TSNE
    fig = plt.figure(figsize=(15, 8))
    fig.suptitle(f"MCQ tSNE with Classification")

    for n_comp in (2, 3):
        tsne_out_path = figure_path.replace(".png", f"_{n_comp}d.json")
        if not force_reload_embeddings and os.path.exists(tsne_out_path):
            df_tsne = pd.read_json(tsne_out_path, orient="records")
        else:
            tsne = TSNE(
                n_components=n_comp, learning_rate="auto", init="random")
            df_tsne = tsne.fit_transform(df_rich)

            # print(df_tsne.shape)
            df_tsne = pd.DataFrame(
                df_tsne, columns=tsne.get_feature_names_out())
            df_tsne[["id", "medshake_class", "medshake_colour"]] = \
                df[["id", "medshake_class", "medshake_colour"]]
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

    # Save the figure
    fig.legend(labels=LABEL_COLOURS.keys())
    fig.savefig(figure_path, bbox_inches="tight")


def main_tsne(
        force_reload: bool = False,
        force_reload_embeddings: bool = False,
):
    print("\ntSNE")
    corpus_path = "data/test-medshake-score.json"
    df = load_corpus(corpus_path)
    tsne(
        df,
        "data/tags-test-medshake-score.json",
        "data/ngrams-test-medshake-score.json",
        "output/analysis/random-forests-data.json",
        "output/analysis/tsne_results.png",
        force_reload=force_reload,
        force_reload_embeddings=force_reload_embeddings,
    )


def umap(
        df: pd.DataFrame,
        tags_path: str,
        ngrams_path: str,
        data_output_path: str,
        figure_path: str,
        force_reload: bool = False,
):
    """
    Executes the UMAP algorithm for dimension reduction and clustering over the
    source data, enriched with the given tags and n-grams.
    """
    df_rich = merge_with_metadata(
        df,
        tags_path=tags_path,
        ngrams_path=ngrams_path,
        data_output_path=data_output_path,
        force_reload=force_reload,
        result_ignored_cols = ["id", "question", "medshake_difficulty",
                               "medshake_class"],
    )

    # UMAP algorithm
    print("Running UMAP algorithm")
    import umap
    from sklearn.preprocessing import StandardScaler

    # M = np.loadtxt(input_file, skiprows=1)
    M = StandardScaler().fit_transform(df_rich)
    reducer = umap.UMAP()
    df_low_d = reducer.fit_transform(M)
    embeddings_path = figure_path.replace(".png", ".txt")
    # print(df_low_d.shape)
    # print(df_low_d)
    df[["umap_1", "umap_2"]] = df_low_d

    # Save projection to file
    np.savetxt(embeddings_path, df_low_d, header='%d %d' % df_low_d.shape)

    # Plot with classes
    fig, ax = plt.subplots(figsize=(15, 8))
    fig.suptitle(f"MCQ UMAP with Classification")
    ax.set_xlabel("UMAP Comp 1")
    ax.set_ylabel("UMAP Comp 2")

    for label in LABEL_COLOURS.keys():
        _df = df[df["medshake_class"] == label]
        ax.scatter(
            _df["umap_1"], _df["umap_2"],
            c=_df["medshake_colour"], label=label)

    # Save the figure
    # fig.legend(loc=(0.8, 0.62))
    # fig.legend(labels=LABEL_COLOURS.keys())
    fig.legend()
    fig.savefig(figure_path, bbox_inches="tight")


def main_umap(force_reload: bool = False):
    print("\nUMAP")
    corpus_path = "data/test-medshake-score.json"
    df = load_corpus(corpus_path)
    umap(
        df,
        "data/tags-test-medshake-score.json",
        "data/ngrams-test-medshake-score.json",
        "output/analysis/random-forests-data.json",
        "output/analysis/umap_results.png",
        force_reload=force_reload,
    )


def main(method_name: str, *args, **kwargs):
    from util import analyse_questions
    method = getattr(analyse_questions, method_name)
    if not method:
        raise f"Method '{method_name}' not found"
    return method(*args, **kwargs)


if __name__ == "__main__":
    import fire
    # fire.Fire(diff_test_files)
    # fire.Fire(main_extract_ngrams)
    # fire.Fire(main_merge_with_metadata)
    # fire.Fire(main_tsne)
    # fire.Fire(main_umap)
    fire.Fire(main)
