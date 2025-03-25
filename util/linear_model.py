"""
Custom implementations of Linear Models, extending models available in
scykit-learn.
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from sklearn import linear_model
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.utils.validation import check_is_fitted


def calc_pvalues(X, y, y_pred, coefs, df_uses_rank = False):
    """
    Calculation of t-statistics and p-values based on:
     - https://stackoverflow.com/a/69095315
     - https://gist.github.com/brentp/5355925
     - https://tidystat.com/calculate-p-value-in-linear-regression/
    """
    n = X.shape[0]  # Number of samples
    p = X.shape[1]  # Number of independent variables /
    # Calculate degrees of freedom like statsmodels, using rank(X) instead of
    # the strict number of independent variables
    if df_uses_rank:
        p = np.linalg.matrix_rank(X)

    from scipy.stats import t

    # NOTE: ignore this as we don't include a constant in X and don't include
    # the intercept in `coefs`, so *I believe* we don' need to add a columns of
    # 1s to X.
    #
    # add ones column
    # X = np.append(np.ones(n), X)

    # standard deviation of the error
    #   https://statisticsbyjim.com/regression/root-mean-square-error-rmse/
    sigma_hat = np.sqrt(np.sum(np.square(y - y_pred)) / (n - p))
    # estimate the covariance matrix for the beta (X)
    beta_cov = np.linalg.inv(X.T@X)
    # the t-test statistic for each variable
    #   Ignore warning due to calculating sqrt of negative values in the
    #   inversed matrix
    import warnings
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        t_statistics = coefs / (sigma_hat * np.sqrt(np.diagonal(beta_cov)))
    # compute 2-sided p-values.
    #   Survival function:
    #   https://docs.scipy.org/doc/scipy-1.15.0/reference/generated/scipy.stats.t.html
    p_vals = t.sf(np.abs(t_statistics), n - p) * 2
    return t_statistics, p_vals


class FitPvalues:
    """
    Partial class to extend sklearn's linear models (LinearRegression, Ridge,
    ElasticNet), that calculates t-statistics and p-values for model
    coefficients.

    Adds the attributes `t_stats` and `p_values` after execution of the method
    `fit()`.

    Adds the method `summary()` that prints a table with coefficients,
    t-statistics and p-values, for each observed independent variable.
    """
    def fit(self, X, y, *args, **kwargs):
        self = super().fit(X, y, *args, **kwargs)
        y_pred = self.predict(X)
        self.t_stats, self.p_values = \
            calc_pvalues(X, y, y_pred, self.coef_)
        return self

    def summary(self, do_print: bool = True) -> pd.DataFrame:
        df = pd.DataFrame({
            "coefficient": self.coef_,
            # "coefficient": np.append(self.intercept_, self.coef_),
            "t_stats": self.t_stats,
            "p_values": self.p_values,
        }, index=self.feature_names_in_)
        # }, index=np.append("intercept", self.feature_names_in_))
        if do_print:
            print(df.round(3))
        return df


class LinearRegression(FitPvalues, linear_model.LinearRegression):
    """
    Extension of sklearn's LinearRegression, that calculates t-statistics and
    p-values for model coefficients.

    Adds the attributes `t_stats` and `p_values` after execution of the method
    `fit()`.
    """


class Ridge(FitPvalues, linear_model.Ridge):
    """
    Extension of sklearn's Ridge, that calculates t-statistics and p-values for
    model coefficients.

    Adds the attributes `t_stats` and `p_values` after execution of the method
    `fit()`.
    """


class ElasticNet(FitPvalues, linear_model.ElasticNet):
    """
    Extension of sklearn's ElasticNet, that calculates t-statistics and p-values
    for model coefficients.

    Adds the attributes `t_stats` and `p_values` after execution of the method
    `fit()`.
    """


class CVRunResult:
    _model: linear_model.Ridge = None
    coefs: list[float] = None
    intercept: float = None
    pvalues: pd.Series = None
    r2: float = None  # Default sklearn Ridge score
    mse: float = None
    rmse: float = None
    mae: float = None
    selected_feat: list[int] = None

    def __init__(
            self, model: linear_model.Ridge,
            coefs: list[float], intercept: float,
    ):
        self._model = model
        self.coefs = coefs
        self.intercept = intercept

    def scores(self, X: pd.DataFrame = None, y: pd.Series = None):
        if X is not None:
            y_pred = self._model.predict(X)
            self.r2 = r2_score(y, y_pred)
            self.mse = mean_squared_error(y, y_pred)
            self.rmse = np.sqrt(self.mse)
            self.mae = mean_absolute_error(y, y_pred)
        return (self.r2, self.mse, self.rmse, self.mae)


class DataCVRidge(RegressorMixin, BaseEstimator):
    """
    Extension of sklearn's Ridge that uses cross-validation by splitting the
    data.
    """

    def __init__(
            self,
            splits: int = 5,
            balance_classes: bool = True,
            use_pvalues: bool = False,
            feature_threshold: float | str = "mean",
            log_steps: bool = False,
    ):
        # Coefficients and p-values after cross-validation executions
        self.cv_results: list[CVRunResult] = []

        # Overall coefficients and intercept will be calculated as the average after
        # cross-validation runs
        self.coef_: list[float] = None
        self.intercept_: float = None
        self.feature_names_in_: list[str] = None
        self.n_features_in_: int = None
        self.pvalues_: list[float] = None

        self.use_pvalues = use_pvalues
        self.feature_threshold = feature_threshold
        self.log_steps = log_steps

        # Use cross-validation with leave-one-out, with the split left out being
        # used as test to calculate score
        if balance_classes:
            self._kf = StratifiedKFold(n_splits=splits)
        else:
            self._kf = KFold(n_splits=splits)

    def fit(self, X: pd.DataFrame, y: pd.Series, classes: pd.Series = None) \
            -> "DataCVRidge":
        self.is_fitted_ = True
        self.feature_names_in_ = X.columns
        self.n_features_in_ = len(X.columns)

        for train_index, test_index in self._kf.split(X, classes):
            # Split X and y for the linear regression algorithm
            df_train_x = X.iloc[train_index]
            df_train_y = y.iloc[train_index]
            df_test_x = X.iloc[test_index]
            df_test_y = y.iloc[test_index]

            # Linear Regression
            if self.log_steps:
                print("\nRunning Linear Regression algorithm")

            # Scikit-learn lib
            # self._model = ElasticNet(alpha=1.0, l1_ratio=0)
            # self._model = LinearRegression()
            model = linear_model.Ridge()
            model.fit(df_train_x, df_train_y)

            result = CVRunResult(model, model.coef_.copy(), model.intercept_)
            self.cv_results.append(result)

            result.scores(df_test_x, df_test_y)

            # Statsmodels lib
            if self.use_pvalues:
                # if isinstance(reg, FitPvalues):
                #     reg_sum = reg.summary()
                #     print(
                #         "\nRidge features retained, significance"
                #         f" {significance_level}:",
                #         reg_sum[reg_sum["p_values"] <= significance_level])

                pvalues_model = sm.OLS(df_train_y, df_train_x, missing="raise")
                # mod = sm.RLM(df_train_y, df_train_x, missing="raise")
                pvalues_res = pvalues_model.fit()
                print("")
                print(pvalues_res.summary())
                # print(
                #     "\n\nFeatures to retain based on significance level"
                #     f" {significance_level}")
                # print(res.pvalues[res.pvalues <= significance_level].round(5))
                result.pvalues = pvalues_res.pvalues

            # When p-values are not used, create a selector to select features
            else:
                # Select features above the mean
                selector = SelectFromModel(
                    model, prefit=True, threshold=self.feature_threshold)
                result.selected_feat = selector.get_support(indices=True)

            if self.log_steps:
                print("R2:", result.r2)
                print("MSE:", result.mse)
                print("Root MSE:", result.rmse)
                print("MAE:", result.mae)
                print(
                    "Selected:",
                    [self.feature_names_in_[i] for i in result.selected_feat])
                print("\n" + "*" * 80 + "\n")

        # Calculate averages after all runs
        self.coef_ = np.average([r.coefs for r in self.cv_results], axis=0)
        self.intercept_ = np.average([r.intercept for r in self.cv_results])
        self.coef_std_ = np.std([r.coefs for r in self.cv_results], axis=0)
        self.intercept_std_ = np.std([r.intercept for r in self.cv_results])
        if self.use_pvalues:
            self.pvalues_ = np.average([r.pvalues for r in self.cv_results], axis=0)
            self.pvalues_std_ = np.std([r.pvalues for r in self.cv_results], axis=0)

        return self

    def predict(self, X: pd.DataFrame) -> np.array:
        check_is_fitted(self)
        if self.coef_.ndim == 1:
            return X @ self.coef_ + self.intercept_
        else:
            return X @ self.coef_.T + self.intercept_
