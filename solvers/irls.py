from benchopt import BaseSolver, safe_import_context


with safe_import_context() as import_ctx:
    import numpy as np
    from scipy.linalg import pinv


class Solver(BaseSolver):
    """IRLS solver."""

    name = "IRLS"

    install_cmd = "conda"
    requirements = ["numpy", "scipy"]
    stopping_strategy = "callback"

    def set_objective(self, X, y, reg, quantile):
        self.X, self.y, self.reg, self.quantile = X, y, reg, quantile

    def run(self, callback):
        X, y, quantile = self.X, self.y, self.quantile
        n_samples, n_features = X.shape

        # to fit_intercept:
        X = np.concatenate([np.ones((n_samples, 1)), X], axis=1)
        n_features += 1

        xstar = X
        beta = np.ones(n_features)
        reg = self.reg / n_samples  # loss is a mean not a sum

        while callback(beta):
            xtx = np.dot(xstar.T, X)
            xty = np.dot(xstar.T, y)
            this_reg = reg / np.abs(beta)
            this_reg[0] = 0  # do not regularize the intercept
            xtx.flat[:: n_features + 1] += this_reg
            beta = np.dot(pinv(xtx), xty)
            resid = y - np.dot(X, beta)

            resid = np.where(
                resid < 0, quantile * resid, (1 - quantile) * resid
            )
            resid = np.abs(resid)
            xstar = X / resid[:, np.newaxis]

        self.coef_ = beta[1:]
        self.intercept_ = beta[0]

    def get_result(self):
        params = np.concatenate([[self.intercept_], self.coef_], axis=0)
        return params
