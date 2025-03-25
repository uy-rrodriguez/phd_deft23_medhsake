"""
Utilities to run Random Forests and Logistic regressions algorithms.

(Both discarded during zork for Humans vs LLMs, article submitted to
BioNLP 2025.)
"""

import json
import os
import sys

from multiprocessing import Pool

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
# from util.linear_model import Ridge, FitPvalues
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# Trick to import local packages when this script is run from the terminal
sys.path.append(os.path.abspath("."))

from util.classify_questions import CLASS_COL, LABEL_COLOURS
from util.regression import corpus_with_metadata


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
        corpus_path: str,
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
        df = corpus_with_metadata(
            corpus_path=corpus_path,
            tags_path=tags_path,
            ngrams_path=ngrams_path,
            data_output_path=data_output_path,
            force_reload=force_reload,
            result_ignored_cols=["medshake_difficulty", "shannon_difficulty"],
        )
        # df = df[:10]

        # Split train/test randomly
        col_y = CLASS_COL
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
    from datetime import date
    today = date.strftime(date.today(), "%Y%m%d")
    base_path = f"output/regression/forests/random_forests_{today}"
    random_forest(
        "data/test-medshake-score.json",
        "data/tags-test-medshake-score.json",
        "data/ngrams-test-medshake-score.json",
        "output/regression/random-forests-data.json",
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
        # ax.plot(coefs_df, marker="o", c=LABEL_COLOURS["hard"])
        ax.bar(coefs_df.index, coefs_df.values, color=LABEL_COLOURS["hard"])
        fig.savefig(figure_path, bbox_inches="tight")
        plt.close(fig)


def logistic_regression(
        corpus_path: str,
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
        df = corpus_with_metadata(
            corpus_path=corpus_path,
            tags_path=tags_path,
            ngrams_path=ngrams_path,
            data_output_path=data_output_path,
            force_reload=force_reload,
            result_ignored_cols=["medshake_difficulty", "shannon_difficulty"],
        )

        # Split train/test randomly
        col_y = CLASS_COL
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
    from datetime import date
    today = date.strftime(date.today(), "%Y%m%d")
    base_path = f"output/regression/logistic/log_regression_{today}"
    logistic_regression(
        "data/test-medshake-score.json",
        "data/tags-test-medshake-score.json",
        use_ngrams=False,
        ngrams_path=None,  # "data/ngrams-test-medshake-score.json",
        data_output_path="output/regression/regression-data.json",
        coefs_output_path=f"{base_path}_coefs.json",
        result_output_path=f"{base_path}_results.txt",
        figure_path=f"{base_path}_coefs.png",
        force_reload=force_reload,
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
    fire.Fire(main)
