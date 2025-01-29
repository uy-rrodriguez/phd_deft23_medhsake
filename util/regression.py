"""
Utility functions to analyse the difficulty of questions, calculated from human
answers or model responses, in terms of question length, lexical structure, etc.
"""

import json
import os
import sys

from multiprocessing import Pool

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
from tqdm import tqdm

# Trick to import local packages when this script is run from the terminal
sys.path.append(os.path.abspath("."))

from analyse_questions import merge_with_metadata
from classify_questions import load_corpus, LABEL_COLOURS


################################################################################
#   RANDOM FORESTS                                                             #
################################################################################

def async_one_forest(
        df_train_x: pd.DataFrame, df_train_y: pd.DataFrame,
        df_test_x: pd.DataFrame, df_test_y: pd.DataFrame,
        feature_names: list[str],
) -> dict:
    classes = list(LABEL_COLOURS.keys())
    all_results = {}
    progress = tqdm(feature_names)
    for feature in progress:
        if not feature:
            progress.set_description(f"All features")
            df_train_x_ = df_train_x
            df_test_x_ = df_test_x
        else:
            progress.set_description(f"Removed feature '{feature}'")
            df_train_x_ = df_train_x.drop(feature, axis=1)
            df_test_x_ = df_test_x.drop(feature, axis=1)

        clf = RandomForestClassifier()
        clf.fit(df_train_x_.drop("id", axis=1), df_train_y)
        results = {}

        preds = clf.predict(df_test_x_.drop("id", axis=1))
        for i in range(len(df_test_x_)):
            pred = preds[i]
            exp = df_test_y.iloc[i]
            results[df_test_x_.iloc[i]["id"]] = (
                pred, exp,
                classes.index(pred), classes.index(exp),
            )
        all_results[feature or "all"] = results
    return all_results


def async_random_forest(
        df_train_x: pd.DataFrame, df_train_y: pd.DataFrame,
        df_test_x: pd.DataFrame, df_test_y: pd.DataFrame,
        feature_names: list,
        num_cpus: int,
) -> dict:
    subsets = np.array_split(feature_names, num_cpus)

    # Wait on all subsets
    with Pool(num_cpus) as p:
        results = p.starmap(
            async_one_forest,
            [
                (df_train_x, df_train_y, df_test_x, df_test_y, subset)
                for subset in subsets
            ],
        )
    return {
        k: v
        for res in results
        for k, v in res.items()
    }


def random_forest(
        df: pd.DataFrame,
        tags_path: str,
        ngrams_path: str,
        data_output_path: str,
        preds_output_path: str,
        rates_output_path: str,
        train_len: float,
        num_cpus: int,
        force_reload: bool,
        force_reload_forests: bool,
):
    """
    Executes the random forest algorithm over the source data, enriched with
    the given tags and n-grams.
    """
    # classes = list(LABEL_COLOURS.keys())
    all_results = {}
    all_rates = {}
    best_feat = None
    worst_acc = np.inf
    worst_dist = -1 * np.inf
    base_acc = 0
    base_dist = 0

    preds_exist = os.path.exists(preds_output_path)
    if force_reload_forests or not preds_exist:
        df = merge_with_metadata(
            df,
            tags_path=tags_path,
            ngrams_path=ngrams_path,
            data_output_path=data_output_path,
            force_reload=force_reload,
        )
        # df = df[:10]

        # Split train/test randomly
        col_y = "medshake_class"
        df_train_x, df_test_x, df_train_y, df_test_y = train_test_split(
            df.drop(col_y, axis=1), df[col_y],
            train_size=train_len, random_state=None)

        print("Running Random Forest algorithm")

        # Remove features one by one to see when the accuracy degrades
        feature_names = [None] + df_train_x.columns.to_list()

        all_results = async_random_forest(
            df_train_x, df_train_y, df_test_x, df_test_y,
            feature_names, num_cpus,
        )

        # Save results
        if preds_output_path:
            with open(preds_output_path, "w") as fp:
                for feature, results in all_results.items():
                    if feature != "all":
                        print(f"{feature}\n{'-' * 80}", file=fp)
                    for k, v in results.items():
                        print(f"{k};{v[0]}|{v[1]}", file=fp)
                    print("=" * 80, file=fp)

    # Load saved predictions
    else:
        classes = list(LABEL_COLOURS.keys())
        with open(preds_output_path) as fp:
            current_feat = None
            line = fp.readline()
            current_feat = "all"
            results = {}
            while line:
                line = line[:-1]
                data = line.split(";")
                id_ = data[0]
                pred, exp = data[1].split("|")
                # print(current_feat, id_, pred, exp)
                results[id_] = (
                    pred, exp,
                    classes.index(pred), classes.index(exp),
                )
                # Check change of feature
                line = fp.readline()
                if line.startswith("="):
                    all_results[current_feat] = results
                    results = {}

                    # Get next feature
                    line = fp.readline()
                    if line:
                        current_feat = line[:-1]
                        fp.readline()  # Skip line with "---"
                        line = fp.readline()

    # Calculate resulting rates
    for feature, results in all_results.items():
        # print(feature, results)
        num_correct = sum([r[0] == r[1] for r in results.values()])
        distances = [abs(r[2] - r[3]) for r in results.values()]
        acc = num_correct / len(results)
        dist = np.average(distances)
        all_rates[feature] = (acc, dist)
        if feature == "all":
            base_acc = acc
            base_dist = dist
        if acc < worst_acc:
            best_feat = feature
            worst_acc = acc
            worst_dist = dist

    # Save rates
    if rates_output_path:
        with open(rates_output_path, "w") as fp:
            print("{", file=fp)
            for i, k in enumerate(all_rates):
                v = all_rates[k]
                comma = "," if i < len(all_rates) - 1 else ""
                print(f'  "{k}": [{v[0]}, {v[1]}]{comma}', file=fp)
            print("}", file=fp)

    print(f"Accuracy all features", base_acc)
    print("Distance all features (avg)", base_dist)

    print(f"Worst accuracy (feature '{best_feat}')", worst_acc)
    print("Distance of worst accuracy", worst_dist)


def main_random_forests(
        num_cpus: int = 4,
        force_reload: bool = False,
        force_reload_forests: bool = False,
):
    print("\nRandom Forests (with n-grams)")
    corpus_path = "data/test-medshake-score.json"
    df = load_corpus(corpus_path)
    from datetime import date
    today = date.strftime(date.today(), "%Y%m%d")
    base_path = f"output/analysis/forests/random_forests_{today}"
    random_forest(
        df,
        "data/tags-test-medshake-score.json",
        "data/ngrams-test-medshake-score.json",
        "output/analysis/random-forests-data.json",
        f"{base_path}_preds.txt",
        f"{base_path}_rates.txt",
        train_len=0.66,
        num_cpus=num_cpus,
        force_reload=force_reload,
        force_reload_forests=force_reload_forests,
    )
    # print("\nRandom Forests (without n-grams)")
    # random_forest(
    #     df,
    #     "data/tags-test-medshake-score.json",
    #     None,
    #     None,
    #     f"{base_path}_no_ngrams_preds.txt",
    #     f"{base_path}_no_ngrams_rates.txt",
    #     train_len=2/3,
    #     num_cpus=num_cpus,
    #     force_reload=force_reload,
    #     force_reload_forests=force_reload_forests,
    # )


################################################################################
#   LOGISTIC REGRESSION                                                        #
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


def logistic_regression(
        df: pd.DataFrame,
        tags_path: str,
        use_ngrams: bool,
        ngrams_path: str | None,
        data_output_path: str | None,
        coefs_output_path: str,
        result_output_path: str,
        figure_path: str,
        force_reload: bool = False,
):
    """
    Executes a Logistic Regression to determine the most important features that
    predict question difficulty. Utilises the source data enriched with the
    given tags and, optionally, n-grams.
    """
    classes = list(LABEL_COLOURS.keys())
    do_reload = (
        force_reload
        or not coefs_output_path or not os.path.exists(coefs_output_path)
        or not result_output_path or not os.path.exists(result_output_path)
    )
    if not do_reload:
        coefs_df = pd.read_json(coefs_output_path, orient="index")
        with open(result_output_path) as fp:
            results = {}
            for line in fp.readlines():
                line = line[:-1]
                _id, values = line.split(";")
                results[_id] = values.split("|")
    else:
        if use_ngrams:
            assert ngrams_path is not None
        else:
            ngrams_path = None
        df = merge_with_metadata(
            df,
            tags_path=tags_path,
            ngrams_path=ngrams_path,
            data_output_path=data_output_path,
            force_reload=force_reload,
        )

        # Split train/test randomly
        col_y = "medshake_class"
        df_train_x, df_test_x, df_train_y, df_test_y = train_test_split(
            df.drop(col_y, axis=1), df[col_y],
            train_size = 0.66, random_state=None)

        # Logistic Regression
        print("Running Logistic Regression algorithm")
        from sklearn.linear_model import LogisticRegression

        log_reg = LogisticRegression(
            max_iter=10000,
            # solver="newton-cg",  # Default: lbfgs
        )
        log_reg.fit(df_train_x.drop("id", axis=1), df_train_y)
        print("Score:", log_reg.score(df_test_x.drop("id", axis=1), df_test_y))

        coefs = log_reg.coef_.copy()
        coefs_df = pd.DataFrame(
            coefs,
            index=log_reg.classes_,
            columns=log_reg.feature_names_in_,
        )

        if coefs_output_path:
            data = coefs_df.to_dict(orient="index")
            with open(coefs_output_path, "w") as fp:
                json.dump(data, indent=2, fp=fp, ensure_ascii=False)

        # Predict classes
        preds = log_reg.predict(df_test_x.drop("id", axis=1))
        preds_prob = log_reg.predict_proba(df_test_x.drop("id", axis=1))
        # Sort the array of probabilities in the logic order of the classes:
        # v.easy < easy < medium < hard < v.hard
        preds_prob = [
            [
                probs[log_reg.classes_.tolist().index(cls_)]
                for cls_ in classes
            ]
            for probs in preds_prob
        ]
        # print(preds.shape)
        # print(preds[:5])
        # print("Probabilities (extract):")
        # print(preds_prob.shape)
        # print(preds_prob[:5])
        results = {
            _id: [exp, pred] + probs
            for _id, exp, pred, probs in zip(
                df_test_x["id"],
                df_test_y,
                preds,
                preds_prob,
            )
        }
        # print("Predictions:")
        # print(json.dumps({
        #     k: results[k]
        #     for k in list(results.keys())[:5]
        # }, indent=2))

        if result_output_path:
            with open(result_output_path, "w") as fp:
                for k, v in results.items():
                    print(f"{k};{'|'.join(str(x) for x in v)}", file=fp)

    # Resulting rates
    num_correct = sum([r[0] == r[1] for r in results.values()])
    distances = [
        abs(classes.index(r[0]) - classes.index(r[1]))
        for r in results.values()
    ]
    print(f"Accuracy {num_correct}/{len(results)}", num_correct / len(results))
    print("Distance (avg)", np.average(distances))

    ############################################################################

    # Plot features
    coefs_df.sort_index(
        inplace=True,
        key=lambda idx: [list(LABEL_COLOURS.keys()).index(k) for k in idx],
    )

    # Plot main features per class
    print("Generating plots")
    plot_regression_coefs(
        coefs_df,
        (25, 8),
        f"MCQ main features class '<cls>'",
        figure_path,
        one_per_class=True)

    # Plot all features and all classes
    plot_regression_coefs(
        coefs_df,
        (50, 8),
        "MCQ feature coefficients by class",
        figure_path)


def main_logistic_regression(
        force_reload: bool = False,
):
    print("\nLogistic Regression")
    corpus_path = "data/test-medshake-score.json"
    df = load_corpus(corpus_path)
    from datetime import date
    today = date.strftime(date.today(), "%Y%m%d")
    base_path = f"output/analysis/log_regression/log_regression_{today}"
    logistic_regression(
        df,
        "data/tags-test-medshake-score.json",
        use_ngrams=False,
        ngrams_path=None,  # "data/ngrams-test-medshake-score.json",
        data_output_path="output/analysis/regression-data.json",
        coefs_output_path=f"{base_path}_coefs.json",
        result_output_path=f"{base_path}_results.txt",
        figure_path=f"{base_path}_coefs.png",
        force_reload=force_reload,
    )


################################################################################
#   LINEAR REGRESSION                                                          #
################################################################################

def linear_regression(
        df: pd.DataFrame,
        tags_path: str,
        use_ngrams: bool,
        ngrams_path: str | None,
        data_output_path: str | None,
        coefs_output_path: str,
        figure_path: str,
        significance_level: float = 0.05,
        force_reload: bool = False,
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
        df = merge_with_metadata(
            df,
            tags_path=tags_path,
            ngrams_path=ngrams_path,
            data_output_path=data_output_path,
            force_reload=force_reload,
            result_ignored_cols = ["question"],
            # include_qa_lengths=True,
            # include_first_last_words=True,
        )

        col_class = "medshake_class"
        col_y = "medshake_difficulty"
        df_class = df[col_class]
        df = df.drop(col_class, axis=1)

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
            from util.linear_model import Ridge, FitPvalues
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
    corpus_path = "data/test-medshake-score.json"
    df = load_corpus(corpus_path)
    from datetime import datetime
    # _date = datetime.strftime(datetime.now(), "%Y%m%d_%H%M")
    _date = datetime.strftime(datetime.now(), "%Y%m%d")
    base_path = f"output/analysis/lin_regression/{_date}/lin_regression_MY_RIDGE_{_date}"
    os.makedirs("/".join(base_path.split("/")[:-1]), exist_ok=True)
    linear_regression(
        df,
        tags_path="data/tags-test-medshake-score.json",
        use_ngrams=False,
        ngrams_path="data/ngrams-test-medshake-score.json",
        data_output_path="output/analysis/regression-data.json",
        # data_output_path=None,

        coefs_output_path=f"{base_path}_coefs.json",
        # coefs_output_path="output/analysis/lin_regression/20250122/lin_regression_MY_RIDGE_20250122_coefs",

        figure_path=f"{base_path}_coefs.png",
        force_reload=force_reload,
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
        df = merge_with_metadata(
            data_output_path=data_path,
            result_ignored_cols=["question", "medshake_class"],
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
