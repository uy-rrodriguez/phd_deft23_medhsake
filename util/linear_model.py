"""
Custom implementations of Linear Models, extending models available in
scykit-learn.
"""

import numpy as np
import pandas as pd
from scipy import stats
from sklearn import linear_model
from sklearn.feature_selection import f_regression


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
