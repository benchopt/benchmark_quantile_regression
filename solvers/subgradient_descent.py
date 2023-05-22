from benchopt import BaseSolver, safe_import_context

with safe_import_context() as import_ctx:
    import numpy as np


def quantile_gradient(X, y, reg=0, q=0.5, base_lr=0.1, momentum=0.9,
                      momentum_decay=0.2, lr_decay=0.8, max_steps=1000,
                      rel_tol=1e-2, abs_tol=1e-4, patience=10,
                      intercept=True, verbose=False):
    lr = base_lr * np.std(y)
    prev_loss = np.infty
    min_loss = np.infty
    patience_steps = 0
    n, m = X.shape
    beta = np.zeros(m + 1)
    cum_grad = beta * 0

    for i in range(max_steps):
        resid = y - np.dot(X, beta[:m]) - beta[-1] * intercept
        coef = beta[:m]
        loss = np.mean(
            (resid > 0) * resid * q - (resid < 0) * resid * (1 - q)
        ) + reg * sum((coef > 0) * coef - (coef < 0) * coef)
        if verbose:
            print(loss)
        if loss > prev_loss:
            cum_grad *= momentum_decay
            lr *= lr_decay
        if loss < min_loss * (1-rel_tol) or loss < min_loss - abs_tol:
            min_loss = loss
            patience_steps = 0
        else:
            patience_steps += 1
        if patience_steps > patience:
            if verbose:
                print(f'early stopping after {i} steps')
            break
        prev_loss = loss
        dldpred = (resid > 0) * q - (resid < 0) * (1 - q)
        grad = np.concatenate([
            np.dot(dldpred, X) /
            len(resid) - reg * ((coef > 0) * 1 - (coef < 0) * 1),
            [np.mean(dldpred)]
        ])
        cum_grad = cum_grad * momentum + grad
        delta = cum_grad * lr
        if reg:
            # small coefficients with small gradient stay small
            delta[:m][(coef == 0) & (np.abs(delta[:m]) < reg)] = 0
        beta += delta
        if reg:
            # coefficient that would change sign just stay zero
            beta[:m][beta[:m] * coef < 0] = 0
    if intercept:
        return beta[-1], beta[:m]
    return beta


class Solver(BaseSolver):
    name = 'Python-subgradient'

    stop_strategy = 'iteration'
    support_sparse = False

    def set_objective(self, X, y, reg, quantile):
        self.X, self.y, self.reg, self.quantile = X, y, reg, quantile

    def run(self, n_iter):
        self.intercept_, self.coef_ = quantile_gradient(
            self.X, self.y, self.reg, self.quantile, max_steps=n_iter
        )

    def get_result(self):
        params = np.concatenate([[self.intercept_], self.coef_], axis=0)
        return params
