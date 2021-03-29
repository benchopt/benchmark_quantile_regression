from benchopt import BaseSolver, safe_import_context


with safe_import_context() as import_ctx:
    import numpy as np
    from scipy.optimize import linprog


class Solver(BaseSolver):
    """Simplex solver using scipy linprog function."""
    name = 'scikit-linprog'

    install_cmd = 'conda'
    requirements = [
        'numpy',
        'scipy'
    ]
    parameters = {
        # 'solver': ['simplex', 'interior-point', 'highs'],
        'solver': ['simplex', 'interior-point'],
    }

    def set_objective(self, X, y, reg, quantile):
        self.X, self.y, self.reg, self.quantile = X, y, reg, quantile

    def run(self, n_iter):

        X, y = self.X, self.y
        n_samples, n_features = self.X.shape
        sample_weights = np.ones(n_samples) / n_samples

        # the linear programming formulation of quantile regression
        # follows https://stats.stackexchange.com/questions/384909/
        #
        # The objective is defined as 1/n * sum(pinball loss) + alpha * L1.
        # So we rescale the penalty term, which is equivalent.
        c = np.concatenate([
            np.ones(n_features * 2) * self.reg,
            sample_weights * self.quantile,
            sample_weights * (1 - self.quantile),
        ])

        a_eq_matrix = np.concatenate([
            X,
            -X,
            np.eye(n_samples),
            -np.eye(n_samples),
        ], axis=1)
        b_eq_vector = y

        method = self.solver

        result = linprog(
            c=c,
            A_eq=a_eq_matrix,
            b_eq=b_eq_vector,
            method=method,
            options={'maxiter': n_iter},
        )
        solution = result.x

        params_pos = solution[:n_features]
        params_neg = solution[n_features:2 * n_features]
        self.w = params_pos - params_neg

    def get_result(self):
        return self.w
