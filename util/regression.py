"""
Utility functions to analyse the difficulty of questions, calculated from human
answers or model responses, in terms of question length, lexical structure, etc.
"""

import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.feature_selection import SelectFromModel
import statsmodels.api as sm
from tqdm import tqdm

# Trick to import local packages when this script is run from the terminal
sys.path.append(os.path.abspath("."))

from util.classify_questions import (
    load_corpus,
    CLASS_COL, LABEL_COLOURS,
)
from util.compare_human_llm import load_model_scores
from util.linear_model import DataCVRidge
from util.preprocess_data import corpus_with_metadata
from util.process_output import inference_difficulty
from st_tagging_tool.config import TAGS_CONFIG


################################################################################
#   UTILITIES                                                                  #
################################################################################

def plot_regression_coefs(
        coefs_df: pd.DataFrame|pd.Series,
        figsize: tuple[int, int],
        suptitle: str,
        figure_path: str,
        one_per_class: bool = False,
        single_plot: bool = False,
):
    """
    Plot the coefficients of a linear/logistic regression.
    """
    def init_plot(figsize, suptitle, grid=True):
        fig, ax = plt.subplots(figsize=figsize)
        fig.suptitle(suptitle)
        ax.set_xlabel("Features")
        ax.set_ylabel("Coefficient")
        ax.xaxis.set_tick_params(
            rotation=80, gridOn=grid, grid_color="#EEEEEE", grid_dashes=(1, 2),
            grid_linewidth=1.5)
        ax.axhline(y=0, color="r", linestyle="-")
        # for feat in _coefs.index:
        #     ax.axvline(x=feat, color="#EEEEEE", linestyle='dotted')
        return fig, ax

    # One file per class
    if one_per_class:
        for _cls, _coefs in coefs_df.iterrows():
            # print(_cls)
            _coefs = _coefs[abs(_coefs) >= 0.1].sort_values(ascending=False)
            path_cls = figure_path.split("_")
            path_cls = "_".join(path_cls[:-1] + [_cls] + [path_cls[-1]])
            fig, ax = init_plot(
                figsize,
                suptitle.replace("<cls>", _cls)
            )
            ax.plot(_coefs, c=LABEL_COLOURS[_cls], linewidth=2, marker="o")
            # Save the figure
            fig.savefig(path_cls, bbox_inches="tight")
            plt.close(fig)

    # Single file with one plot per class
    elif not single_plot:
        fig, ax = init_plot(figsize, suptitle)
        for _cls, _coefs in coefs_df.iterrows():
            ax.plot(_coefs, label=_cls, c=LABEL_COLOURS[_cls], marker="o")

        # Save the figure
        fig.savefig(figure_path, bbox_inches="tight")
        plt.close(fig)

    # Single file with a single plot for the entire DataFrame
    else:
        fig, ax = init_plot(figsize, suptitle, grid=False)
        ax.bar(coefs_df.index, coefs_df.values, color=LABEL_COLOURS["hard"])
        fig.savefig(figure_path, bbox_inches="tight")
        plt.close(fig)


################################################################################
#   LINEAR REGRESSION                                                          #
################################################################################

def logits_difficulty(
        corpus_path: str = "data/train+test-with-medshake.json",
        model_scores_path: str = "output/model_scores/mistral-7b-deft_017_20250304/train+test-logp_20250318.json",
):
    """
    Loads model inference output and generates a difficulty score per question.
    """
    # Load LLM rates from model scores
    llm_scores_df = load_model_scores(
        corpus_path=corpus_path,
        model_scores_path=model_scores_path,
        # score_field="LETTERS_ONLY",
        score_field="letters_logp",
        softmax=True,
    )
    # print(llm_scores_df)
    # print(llm_scores_df["emr"].mean())

    # EMR stores the probability of the correct answer
    return llm_scores_df["emr"].apply(lambda x: 1 - x) \
        .rename("logits_difficulty")


def test_logits_difficulty(
        corpus_path: str = "data/train+test-with-medshake.json",
):
    diff = logits_difficulty(
        corpus_path=corpus_path,
    )
    print(diff)

    _ids = [
        "006e1bacf5401adafd5797448a7feed411a1b4d1ae2b5ace26c669c33fcc0100",
        "230bac49b0fe863b772410bc8d01a025f63c3c999065480131d6334abd2efeff",
    ]
    print(type(diff[_ids]), diff[_ids])

    corpus = pd.read_json(corpus_path, orient="records")
    x = corpus[corpus["id"].isin(_ids)]
    print(x)
    print(diff[x["id"]])


def linear_regression(
        corpus_path: str,
        tags_path: str,
        ngrams_path: str | None,
        regression_data_path: str | None,
        coefs_output_path: str,
        figure_path: str,
        # coef_retention_level: float = 0.8,
        with_inference_difficulty: bool = False,
        with_logits_difficulty: bool = False,
        feature_threshold: float | str = "mean",
        use_pvalues: bool = False,
        significance_level: float = 0.05,
        force_reload: bool = False,
        include_qa_lengths: bool = False,
        include_first_last_words: bool = False,
        include_linguistic: bool = False,
        include_year: bool = False,
        include_answer_choices: bool = False,
        normalise: bool = True,
        features_regex: str = None,
        cv_splits: int = 5,
        cv_balance_classes: bool = True,
        find_best: bool = False,
):
    """
    Executes a Linear Regression to determine the most important features that
    predict MedShake score. Utilises the source data enriched with the given
    tags and, optionally, n-grams.

    Data is split in `cv_splits` number of folds for cross-validation. If
    `balance_classes` is True, the class column is used to balance samples in
    each fold.

    Scikit-Learn's Ridge is used to calculate the coefficients.
    When `use_pvalues` is False (default), `feat_threshold` is used to
    determine the most important features with an absolute score at least as
    much (e.g., "mean" of all coefficients, or 0.1).

    However, when `use_pvalues` is True, StatsModels is used to determine
    p-values and eliminate features with a `significance_level` > 0.05. In this
    case `feat_threshold` is ignored.

    Results are plotted in multiple files.
    """
    do_reload = (
        force_reload
        or not coefs_output_path or not os.path.exists(coefs_output_path)
    )
    if not do_reload:
        print(f"Loading coefficients '{coefs_output_path}'")
        regression_df = pd.read_json(
            coefs_output_path, orient="index", encoding="utf-8")
        coefs_cols = regression_df.filter(regex=r"coef_.+", axis=1).columns
        cv_coefs_df = regression_df[coefs_cols]
        cv_coefs_df = cv_coefs_df.drop("intercept", axis=0)
        if use_pvalues:
            pvalues_cols = \
                regression_df.filter(regex=r"pvalue_.+", axis=1).columns
            cv_pvalues_df = regression_df[pvalues_cols]
    else:
        df = corpus_with_metadata(
            corpus_path=corpus_path,
            tags_path=tags_path,
            ngrams_path=ngrams_path,
            data_output_path=regression_data_path,
            force_reload=force_reload,
            include_qa_lengths=include_qa_lengths,
            include_first_last_words=include_first_last_words,
            include_linguistic=include_linguistic,
            normalise=normalise,
            result_ignored_cols=None,
        )

        if not include_year:
            df = df.filter(regex=r"^(?!year_)")
        if not include_answer_choices:
            df = df.filter(regex=r"^(?!answer_[a-e])")

        col_y = "medshake_difficulty"

        if features_regex:
            print(f"Use only features matching '{features_regex}'")
            keep_feats = ["id", CLASS_COL, col_y]
            filtered_df = df.filter(regex=features_regex, axis=1).copy(deep=True)
            filtered_df[keep_feats] = df[keep_feats].copy(deep=True)
            df = filtered_df

        df_class = df[CLASS_COL]
        X = df.drop(
            ["id", "medshake_class", "shannon_class",
             "medshake_difficulty", "shannon_difficulty"],
            axis=1,
            errors="ignore")

        # Different options to calculate difficulty (humans vs models)
        if with_inference_difficulty:
            inference = inference_difficulty(
                corpus_path=corpus_path,
                force_reload=False,
            )
            y = inference[df["id"]]
        elif with_logits_difficulty:
            logits = logits_difficulty(
                corpus_path=corpus_path,
            )
            y = logits[df["id"]]
        else:
            y = df[col_y]
        print(pd.concat([df[col_y], y]))

        ####### START REGRESSION ###############################################

        # Loop and reduce features until local optimum is found
        best_model = None
        best_score = -1
        score = 0
        i = 0
        max_loops = 10 if find_best else 1
        while best_score <= score and i < max_loops:
            i += 1
            model = DataCVRidge(
                splits=cv_splits,
                balance_classes=cv_balance_classes,
                log_steps=True,
            )
            model.fit(X, y, classes=df_class)
            score = np.average([r.r2 for r in model.cv_results])
            if score >= best_score:
                best_model = model
                best_score = score
                selector = SelectFromModel(
                    model, prefit=True, threshold=feature_threshold)
                selected_feat = selector.get_support(indices=True)
                rejected_feat = [
                    X.columns[i]
                    for i in range(len(X.columns))
                    if i not in selected_feat
                ]
                X = X.drop(rejected_feat, axis=1)
                print("=" * 80)
                print(f"New best score R-square: {score:2.3f}")
                print("=" * 80)

        model = best_model

        ####### END REGRESSION #################################################

        # Create DataFrame from each split result
        regression_df = \
        cv_coefs_df = pd.DataFrame(
            [r.coefs for r in model.cv_results],
            columns=model.feature_names_in_,
            index=[f"coef_{i}" for i, _ in enumerate(model.cv_results)]).T

        if use_pvalues:
            cv_pvalues_df = pd.DataFrame(
                [r.pvalues for r in model.cv_results],
                index=[f"pvalue_{i}" for i, _ in enumerate(model.cv_results)]).T
            regression_df = pd.merge(
                cv_coefs_df, cv_pvalues_df,
                left_index=True, right_index=True,
            )

        print(regression_df)

        # Save coefficients and p-values from cross-validation runs
        if coefs_output_path:
            data = regression_df.to_dict(orient="index")
            data["intercept"] = {
                f"coef_{i}": v
                for i, v in enumerate([r.intercept for r in model.cv_results])
            }
            with open(coefs_output_path, "w", encoding="utf-8") as fp:
                json.dump(data, indent=2, fp=fp, ensure_ascii=False)

        # Prediction scores during cross-validation
        metrics_config = (
            ("r2", "R-squared", "score_r2"),
            ("mse", "Mean Squared Error", "score_mse"),
            ("rmse", "Root Mean Squared Error", "score_rmse"),
            ("mae", "Mean Absolute Error", "score_mae"),
        )
        for (metric, title, suff) in metrics_config:
            scores = [getattr(r, metric, None) for r in model.cv_results]
            if scores:
                fig, ax = plt.subplots()
                fig.suptitle(f"MCQ cross-validation {title}")
                ax.set_xlabel("Cross-validation runs")
                ax.set_ylabel("Score")
                ax.axhline(y=np.average(scores), color="r", linestyle="-")
                ax.plot(
                    [i for i in range(1, cv_splits + 1)], scores,
                    c=LABEL_COLOURS["hard"])
                fig.savefig(
                    figure_path.replace("coefs.", f"{suff}."),
                    bbox_inches="tight")
                plt.close(fig)

    ############################################################################

    # Calculate avg and std of results from cross-validation
    cv_coefs_avg = cv_coefs_df.mean(axis=1).rename("coef")
    cv_coefs_std = cv_coefs_df.std(axis=1).rename("coef_std")
    if use_pvalues:
        cv_pvalues_avg = cv_pvalues_df.mean(axis=1).rename("pvalue")
        cv_pvalues_std = cv_pvalues_df.std(axis=1).rename("pvalue_std")
        merged_dfs = (cv_coefs_std, cv_pvalues_avg, cv_pvalues_std)
    else:
        merged_dfs = (cv_coefs_std,)

    avg_df = pd.DataFrame(cv_coefs_avg)
    for _df in merged_dfs:
        avg_df = avg_df.merge(_df, left_index=True, right_index=True)

    print("\nAverage coefficients and std. deviation")
    print(avg_df)

    if use_pvalues:
        selected_feat_df = (
            avg_df[avg_df["pvalue"] <= significance_level]
                .sort_values(by="coef", key=lambda x: abs(x), ascending=False)
        )
    else:
        fake_model = Ridge()
        fake_model.coef_ = cv_coefs_avg.values
        selector = SelectFromModel(
            fake_model, prefit=True, threshold=feature_threshold)
        selected_feat = selector.get_support(indices=True)
        selected_feat_df = (
            avg_df.iloc[list(selected_feat)]
                .sort_values(by="coef", key=lambda x: abs(x), ascending=False)
        )
    print("\nSelected features")
    print(selected_feat_df.round(5))

    ######## GENERATING PLOTS ##################################################

    # Plot features
    print("\nGenerating plots")

    # All features
    plot_regression_coefs(
        cv_coefs_avg.sort_values(key=lambda x: abs(x), ascending=False),
        (30, 4) if not find_best else (10, 4),
        "MCQ all feature coefficients",
        figure_path.replace(".", "_all."),
        single_plot=True)

    # Tags
    plot_regression_coefs(
        cv_coefs_avg.filter(regex=r"^tag_.*", axis=0)
            .sort_values(key=lambda x: abs(x), ascending=False),
        (6, 4) if not find_best else (4, 4),
        "MCQ tag features coefficients",
        figure_path.replace(".", "_tags."),
        single_plot=True)

    # Topics
    plot_regression_coefs(
        cv_coefs_avg.filter(regex=r"^topic_.*", axis=0)
            .sort_values(key=lambda x: abs(x), ascending=False),
        (10, 4)  if not find_best else (7, 4),
        "MCQ topic features coefficients",
        figure_path.replace(".", "_topics."),
        single_plot=True)

    # Selected features
    plot_regression_coefs(
        selected_feat_df["coef"],
        (10, 4) if not find_best else (5, 4),
        "MCQ selected features coefficients",
        figure_path.replace(".", "_sel."),
        single_plot=True)

    # Cross-validation boxplot of coefficients and std. deviation
    def box_plot(
            _df, filter_selected, sort_asc, suptitle, ylabel,
            figsize, figure_path
    ):
        hist_df = _df
        if filter_selected:
            hist_df = _df.filter(items=selected_feat_df.index, axis=0)
        hist_df = hist_df.sort_index(
            key=lambda x: hist_df.loc[x].T.mean().abs(),
            ascending=sort_asc
        )
        hist_df = hist_df.T
        fig, ax = plt.subplots(figsize=figsize)
        fig.suptitle(suptitle)
        ax.set_xlabel("Features")
        ax.set_ylabel(ylabel)
        ax.axhline(y=0, color="r", linestyle="-")
        hist_df.boxplot(ax=ax)
        ax.xaxis.set_tick_params(
            rotation=80, gridOn=True, grid_color="#EEEEEE", grid_dashes=(1, 2),
            grid_linewidth=1.5)
        fig.savefig(figure_path, bbox_inches="tight")

    box_plot(
        cv_coefs_df, True, False,
        "MCQ cross-validation coefficients",
        "Coefficient", (12, 4) if not find_best else (6, 4),
        figure_path.replace(".", "_hist."))
    box_plot(
        cv_coefs_df, False, False,
        "MCQ cross-validation all coefficients",
        "Coefficient", (40, 4) if not find_best else (12, 4),
        figure_path.replace(".", "_hist_all."))

    if use_pvalues:
        box_plot(
            cv_pvalues_df, True, True,
            "MCQ cross-validation p-values",
            "P-value", (12, 4) if not find_best else (6, 4),
            figure_path.replace("coefs.", "pvalues_hist."))
        box_plot(
            cv_pvalues_df, False, True,
            "MCQ cross-validation all p-values",
            "P-value", (40, 4) if not find_best else (12, 4),
            figure_path.replace("coefs.", "pvalues_hist_all."))


def main_linear_regression(
        force_reload: bool = False,
        with_inference: bool = False,
        with_logits: bool = False,
):
    print("\nLinear Regression")
    from datetime import datetime
    # _date = datetime.strftime(datetime.now(), "%Y%m%d_%H%M")
    _date = datetime.strftime(datetime.now(), "%Y%m%d")
    out_dir = f"output/analysis/lin_regression/{_date}_train+test"
    corpus_path = "data/train+test-with-medshake.json"
    tags_path = "data/tags-train+test-with-medshake.json"
    regression_data_path = "output/analysis/train+test-regression-data.json"

    if with_inference:
        out_dir += "/inference"
        # Inference is only generated for the test split
        corpus_path = "data/test-medshake-data.json"
        tags_path = "data/tags-test-with-medshake.json"
        regression_data_path = "output/analysis/test-regression-data.json"
    elif with_logits:
        out_dir += "/logits"
    else:
        out_dir += "/humans"

    def run_regression(
            coefs_output_path: str,
            features_regex: str = None,
            find_best: bool = False,
    ):
        os.makedirs(os.path.dirname(coefs_output_path), exist_ok=True)
        linear_regression(
            # Build or load data source for regression
            # corpus_path="data/test-medshake-score.json",
            # tags_path="data/tags-test-medshake-score.json",
            corpus_path=corpus_path,
            tags_path=tags_path,
            regression_data_path=regression_data_path,
            ngrams_path=None,  # "data/ngrams-test-medshake-score.json",
            include_qa_lengths=True,
            # include_first_last_words=True,
            include_linguistic=True,
            force_reload=force_reload,

            # How to select difficulty (by default, uses "medshake_difficulty")
            with_inference_difficulty=with_inference,
            with_logits_difficulty=with_logits,

            # Customise regression method
            features_regex=features_regex,
            # normalise=False,
            feature_threshold="mean",  # 0.1,
            use_pvalues=False,
            cv_splits=5,
            # cv_balance_classes=False,
            find_best=find_best,

            # Output path
            coefs_output_path=coefs_output_path,
            # coefs_output_path="output/analysis/lin_regression/20250122/lin_regression_MY_RIDGE_20250122_coefs",
            figure_path=coefs_output_path.replace(".json", ".png"),
        )

    # Regression with all features (both methods: find best model and normal)
    for best in (False, True):
        with_best_dir = f"{out_dir}/{'full_best' if best else 'full_no_best'}"
        with_best_path = f"{with_best_dir}/regression_coefs.json"
        run_regression(coefs_output_path=with_best_path, find_best=best)

    # Regressions per feature subset
    feat_subsets = {
        "linguistic": "|".join([
            "word_count", "avg_word_count", "avg_tree", "np_count", "avg_np_words",
            "vp_count", "avg_vp_words", "pp_count", "avg_pp_words",
        ]),
        "topics": r"topic_.+",
        # "years": r"year_.+",
        "tags": "|".join(TAGS_CONFIG.keys()),
        "answers": r"nbr_correct_answers|answer_.+"
    }
    for key, re in feat_subsets.items():
        print("\n" + "="*80 + f"\nRegression for feature set: {key}")
        feats_path = f"{out_dir}/{key}/{key}_coefs.json"
        run_regression(coefs_output_path=feats_path, features_regex=re)


################################################################################
#   TEST REGRESSION HYPOTHESES                                                 #
################################################################################

def plot_residuals(y: np.array, y_pred: np.array, figure_path: str):
    """
    Plot the residuals after fitting a regression model.
    """
    fig, ax = plt.subplots()
    fig.suptitle("MCQ Regression Residuals")
    ax.set_xlabel("Fitted Value")
    ax.set_ylabel("Residual")
    ax.axhline(y=0, color="r", linestyle="-")
    ax.scatter(y_pred, y_pred - y, marker="o", c="#3B528B")
    fig.savefig(figure_path, bbox_inches="tight")

def main_plot_residuals():
    load_results = False

    if load_results:
        result_path = (
            "output/analysis/lin_regression/"
            "lin_regression_MY_EL_20250115_results.txt"
        )
        results = {}
        with open(result_path) as fp:
            for line in fp.readlines():
                line = line[:-1]
                _id, values = line.split(";")
                results[_id] = [float(v) for v in values.split("|")]
        values = np.array(list(results.values())).T
        y = values[0]
        y_pred = values[0]

    else:
        data_path = "output/analysis/regression-data.json"
        print(f"Loading data from file '{data_path}'")
        df = corpus_with_metadata(
            data_output_path=data_path,
            result_ignored_cols=["question", "medshake_class", "shannon_class"],
        )

        coefs_path = (
            "output/analysis/lin_regression/20250122/"
            "lin_regression_MY_RIDGE_20250122_coefs.json"
        )
        print(f"Loading coefficients from file '{coefs_path}'")
        with open(coefs_path) as fp:
            regression_df = pd.read_json(coefs_path, orient="index")
            coefs_cols = regression_df.filter(regex=r"coef_.+", axis=1).columns
            coefs_df = regression_df[coefs_cols].T.mean()
            intercept = coefs_df["intercept"]
            coefs_df.drop("intercept", inplace=True)
            features = coefs_df.index.to_list()

        from util.linear_model import Ridge
        reg = Ridge()
        reg.coef_ = coefs_df.values.copy()
        reg.intercept_ = intercept
        reg.feature_names_in_ = features

        X = df[features]
        y = df["medshake_difficulty"],
        y_pred = reg.predict(X)

    plot_residuals(y, y_pred, "output/analysis/lin_regression/residuals.png")


def main(method_name: str, *args, **kwargs):
    import inspect
    module = inspect.getmodule(main)
    method = getattr(module, method_name)
    if not method:
        raise f"Method '{method_name}' not found"
    return method(*args, **kwargs)


if __name__ == "__main__":
    import fire
    fire.Fire(main)
