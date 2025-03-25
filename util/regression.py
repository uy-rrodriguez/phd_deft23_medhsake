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
import statsmodels.api as sm
from tqdm import tqdm

# Trick to import local packages when this script is run from the terminal
sys.path.append(os.path.abspath("."))

from analyse_questions import CLASS_COL, corpus_with_metadata
from classify_questions import load_corpus, LABEL_COLOURS


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
    def init_plot(figsize, suptitle):
        fig, ax = plt.subplots(figsize=figsize)
        fig.suptitle(suptitle)
        ax.set_xlabel("Features")
        ax.set_ylabel("Coefficient")
        ax.xaxis.set_tick_params(
            rotation=80, gridOn=True, grid_color="#EEEEEE", grid_dashes=(1, 2),
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

    # Single file with one plot per class
    elif not single_plot:
        fig, ax = init_plot(figsize, suptitle)
        for _cls, _coefs in coefs_df.iterrows():
            ax.plot(_coefs, label=_cls, c=LABEL_COLOURS[_cls], marker="o")

        # Save the figure
        fig.savefig(figure_path, bbox_inches="tight")

    # Single file with a single plot for the entire DataFrame
    else:
        fig, ax = init_plot(figsize, suptitle)
        ax.plot(coefs_df, marker="o", c="#3B528B")
        fig.savefig(figure_path, bbox_inches="tight")
        plt.close(fig)


################################################################################
#   LINEAR REGRESSION                                                          #
################################################################################

def linear_regression(
        corpus_path: str,
        tags_path: str,
        use_ngrams: bool,
        ngrams_path: str | None,
        data_output_path: str | None,
        coefs_output_path: str,
        figure_path: str,
        significance_level: float = 0.05,
        force_reload: bool = False,
        include_qa_lengths: bool = False,
        include_first_last_words: bool = False,
        normalise: bool = True,
        cv_splits: int = 5,
        cv_balance_classes: bool = True,
):
    """
    Executes a Linear Regression to determine the most important features that
    predict MedShake score. Utilises the source data enriched with the given
    tags and, optionally, n-grams.

    Data is split in `cv_splits` number of folds for cross-validation. If
    `balance_classes` is True, the class column is used to balance samples in
    each fold.

    First, StatsModels is used to determine p-values and eliminate features with
    a `significance_level` > 0.05.

    Then, Scikit-Learn's Ridge is used to calculate final coefficients.

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
        pvalues_cols = regression_df.filter(regex=r"pvalue_.+", axis=1).columns
        cv_pvalues_df = regression_df[pvalues_cols]

        cv_scores = None
    else:
        if use_ngrams:
            assert ngrams_path is not None
        else:
            ngrams_path = None
        df = corpus_with_metadata(
            corpus_path=corpus_path,
            tags_path=tags_path,
            ngrams_path=ngrams_path,
            data_output_path=data_output_path,
            force_reload=force_reload,
            result_ignored_cols = ["question"],
            include_qa_lengths=include_qa_lengths,
            include_first_last_words=include_first_last_words,
            normalise=normalise,
        )

        col_class = "medshake_class"
        col_y = "medshake_difficulty"
        df_class = df[CLASS_COL]
        df = df.drop(CLASS_COL, axis=1)

        ####### START REGRESSION ###############################################

        # Coefficients and p-values after cross-validation executions
        cv_scores: list[float] = []
        cv_coefs: list[list[float]] = []
        cv_intercepts: list[float] = []
        cv_pvalues: list[pd.Series] = []

        # Use cross-validation with leave-one-out, with the split left out being
        # used as test to calculate score
        if cv_balance_classes:
            kf = StratifiedKFold(n_splits=cv_splits)
        else:
            kf = KFold(n_splits=cv_splits)
        for (train_index, test_index) in kf.split(df, df_class):
            # Split X and y for the linear regression algorithm
            df_train = df.iloc[train_index]
            df_train_x = df_train.drop(["id", col_y], axis=1)
            df_train_y = df_train[col_y]

            df_test = df.iloc[test_index]
            df_test_x = df_test.drop(["id", col_y], axis=1)
            df_test_y = df_test[col_y]

            # Linear Regression
            print("\nRunning Linear Regression algorithm")

            # Scikit-learn lib
            # from util.linear_model import Ridge, FitPvalues
            from sklearn.linear_model import Ridge
            # reg = ElasticNet(alpha=1.0, l1_ratio=0)
            # reg = LinearRegression()
            reg = Ridge()
            reg.fit(df_train_x, df_train_y)

            # No score calculated on each cross-validation step
            cv_scores.append(reg.score(df_test_x, df_test_y))
            print("Score:", cv_scores[-1])

            # if isinstance(reg, FitPvalues):
            #     reg_sum = reg.summary()
            #     print(
            #         "\nRidge features retained, significance"
            #         f" {significance_level}:",
            #         reg_sum[reg_sum["p_values"] <= significance_level])

            # Statsmodels lib
            mod = sm.OLS(df_train_y, df_train_x, missing="raise")
            # mod = sm.RLM(df_train_y, df_train_x, missing="raise")
            res = mod.fit()
            print("")
            print(res.summary())
            # print(
            #     "\n\nFeatures to retain based on significance level"
            #     f" {significance_level}")
            # print(res.pvalues[res.pvalues <= significance_level].round(5))

            cv_coefs.append(reg.coef_.copy())
            cv_intercepts.append(reg.intercept_)
            cv_pvalues.append(res.pvalues)
            print("\n" + "*" * 80 + "\n")

        ####### END REGRESSION #################################################

        # Save coefficients and p-values from cross-validation runs
        cv_coefs_df = pd.DataFrame(
            cv_coefs, columns=reg.feature_names_in_,
            index=[f"coef_{i}" for i in range(cv_splits)]).T

        cv_pvalues_df = pd.DataFrame(
            cv_pvalues, index=[f"pvalue_{i}" for i in range(cv_splits)]).T

        regression_df = pd.merge(
            cv_coefs_df, cv_pvalues_df,
            left_index=True, right_index=True,
        )
        print(regression_df)

        if coefs_output_path:
            data = regression_df.to_dict(orient="index")
            data["intercept"] = {
                f"coef_{i}": v
                for i, v in enumerate(cv_intercepts)
            }
            with open(coefs_output_path, "w", encoding="utf-8") as fp:
                json.dump(data, indent=2, fp=fp, ensure_ascii=False)

    ############################################################################

    # Calculate avg and std of results from cross-validation
    cv_coefs_avg = cv_coefs_df.mean(axis=1).rename("coef")
    cv_coefs_std = cv_coefs_df.std(axis=1).rename("coef_std")
    cv_pvalues_avg = cv_pvalues_df.mean(axis=1).rename("pvalue")
    cv_pvalues_std = cv_pvalues_df.std(axis=1).rename("pvalue_std")

    avg_df = pd.DataFrame(cv_coefs_avg)
    for _df in (cv_coefs_std, cv_pvalues_avg, cv_pvalues_std):
        avg_df = avg_df.merge(_df, left_index=True, right_index=True)
    print("\nAverage coefficients and std. deviation")
    print(avg_df)

    selected_feat_df = (
        avg_df[avg_df["pvalue"] <= significance_level]
            .sort_values(by="coef", key=lambda x: abs(x), ascending=False)
    )
    print("\nSelected features")
    print(selected_feat_df.round(5))

    # Plot features
    print("\nGenerating plots")

    # All features
    plot_regression_coefs(
        cv_coefs_avg.sort_values(key=lambda x: abs(x), ascending=False),
        (50, 8),
        "MCQ all feature coefficients",
        figure_path.replace(".", "_all."),
        single_plot=True)

    # Tags
    plot_regression_coefs(
        cv_coefs_avg.filter(regex=r"^tag_.*", axis=0)
            .sort_values(key=lambda x: abs(x), ascending=False),
        (12, 8),
        "MCQ tag features coefficients",
        figure_path.replace(".", "_tags."),
        single_plot=True)

    # Topics
    plot_regression_coefs(
        cv_coefs_avg.filter(regex=r"^topic_.*", axis=0)
            .sort_values(key=lambda x: abs(x), ascending=False),
        (20, 8),
        "MCQ topic features coefficients",
        figure_path.replace(".", "_topics."),
        single_plot=True)

    # Selected features
    plot_regression_coefs(
        selected_feat_df["coef"],
        (12, 8),
        "MCQ selected features coefficients",
        figure_path.replace(".", "_sel."),
        single_plot=True)

    # Cross-validation Histogram of coefficients and std. deviation
    def hist_plot(
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

    hist_plot(
        cv_coefs_df, True, False,
        "MCQ cross-validation coefficients",
        "Coefficient", (12, 8),
        figure_path.replace(".", "_hist."))
    hist_plot(
        cv_pvalues_df, True, True,
        "MCQ cross-validation p-values",
        "P-value", (12, 8),
        figure_path.replace("coefs.", "pvalues_hist."))

    hist_plot(
        cv_coefs_df, False, False,
        "MCQ cross-validation all coefficients",
        "Coefficient", (50, 8),
        figure_path.replace(".", "_hist_all."))
    hist_plot(
        cv_pvalues_df, False, True,
        "MCQ cross-validation all p-values",
        "P-value", (50, 8),
        figure_path.replace("coefs.", "pvalues_hist_all."))

    # Prediction scores during cross-validation
    if cv_scores:
        fig, ax = plt.subplots()
        fig.suptitle("MCQ cross-validation prediction scores")
        ax.set_xlabel("Cross-validation runs")
        ax.set_ylabel("Score")
        ax.axhline(y=np.average(cv_scores), color="r", linestyle="-")
        ax.plot(
            [i for i in range(1, cv_splits + 1)], cv_scores,
            c=LABEL_COLOURS["hard"])
        fig.savefig(figure_path.replace("coefs.", "scores."), bbox_inches="tight")


def main_linear_regression(
        force_reload: bool = False,
):
    print("\nLinear Regression")
    from datetime import datetime
    # _date = datetime.strftime(datetime.now(), "%Y%m%d_%H%M")
    _date = datetime.strftime(datetime.now(), "%Y%m%d")
    base_path = f"output/analysis/lin_regression/{_date}/lin_regression_MY_RIDGE_{_date}"
    os.makedirs("/".join(base_path.split("/")[:-1]), exist_ok=True)
    linear_regression(
        corpus_path="data/test-medshake-score.json",
        tags_path="data/tags-test-medshake-score.json",
        use_ngrams=False,
        ngrams_path="data/ngrams-test-medshake-score.json",
        data_output_path="output/analysis/regression-data.json",
        # data_output_path=None,

        coefs_output_path=f"{base_path}_coefs.json",
        # coefs_output_path="output/analysis/lin_regression/20250122/lin_regression_MY_RIDGE_20250122_coefs",

        figure_path=f"{base_path}_coefs.png",
        force_reload=force_reload,
        include_qa_lengths=True,
        include_first_last_words=True,
        # normalise=False,
        cv_splits=5,
        # cv_balance_classes=False,
    )


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
    from util import regression
    method = getattr(regression, method_name)
    if not method:
        raise f"Method '{method_name}' not found"
    return method(*args, **kwargs)


if __name__ == "__main__":
    import fire
    # fire.Fire(main_logistic_regression)
    # fire.Fire(main_linear_regression)
    fire.Fire(main)
