from benchopt import BaseSolver
from benchopt import safe_import_context

with safe_import_context() as import_ctx:
    import numpy as np

    from skglm.penalties import L1
    from skglm.experimental.pdcd_ws import PDCD_WS
    from skglm.experimental.quantile_regression import Pinball


class Solver(BaseSolver):
    name = "PDCD-WS"

    requirements = [
        'pip:git+https://github.com/Badr-MOUFAD/skglm.git@pinball-df'
    ]

    references = [
        'Q. Bertrand and Q. Klopfenstein and P.-A. Bannier and G. Gidel'
        'and M. Massias'
        '"Beyond L1: Faster and Better Sparse Models with skglm", '
        'https://arxiv.org/abs/2204.07826'
    ]

    stopping_strategy = "iteration"

    def set_objective(self, X, y, lmbd, quantile, fit_intercept):
        self.X, self.y, self.lmbd = X, y, lmbd
        self.quantile = quantile
        self.fit_intercept = fit_intercept

        self.penalty = L1(len(y) * lmbd)
        self.datafit = Pinball(self.quantile)

        self.solver = PDCD_WS(
            tol=1e-9,
            dual_init=np.sign(y)/2 + (quantile - 0.5)
        )

        # Cache Numba compilation
        self.run(5)

    def run(self, n_iter):
        if n_iter == 0:
            self.coef = np.zeros(self.X.shape[1])
        else:
            self.solver.max_iter = n_iter
            coef = self.solver.solve(self.X, self.y,
                                     self.datafit, self.penalty)[0]

            self.coef = coef.flatten()

    def get_result(self):
        return self.coef
