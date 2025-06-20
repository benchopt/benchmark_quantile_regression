# solvers/solver_skglm_quantile_huber.py

from benchopt import BaseSolver, safe_import_context
import numpy as np
import warnings
with safe_import_context() as import_ctx:
    from skglm.experimental.quantile_huber import SmoothQuantileRegressor


class Solver(BaseSolver):
    """Smooth quantile regression solver using skglm."""
    name = 'skglm-SmoothQuantileRegressor'

    install_cmd = 'conda'
    requirements = ["numpy", "scikit-learn", "numba", "skglm"]
    parameters = {}
    stop_strategy = 'tolerance'

    def set_objective(self, X, y, lmbd, quantile, fit_intercept):
        self.X, self.y = X, y
        self.lmbd = lmbd
        self.quantile = quantile
        self.fit_intercept = fit_intercept

    def run(self, tol):
        est = SmoothQuantileRegressor(
            quantile=self.quantile,
            alpha=self.lmbd,
            delta_init=0.5,
            delta_final=0.001,
            n_deltas=5,
            max_iter=200,
            tol=tol,
            verbose=True,
            fit_intercept=self.fit_intercept,
        )
        warnings.filterwarnings('ignore')
        est.fit(self.X, self.y)

        self.coef_ = est.coef_
        if self.fit_intercept:
            self.intercept_ = est.intercept_

    def get_result(self):
        if self.fit_intercept:
            params = np.concatenate((self.coef_, [self.intercept_]))
        else:
            params = self.coef_
        return dict(params=params)
