from benchopt import BaseObjective
import numpy as np


def pin_ball_loss(y_true, y_pred, quantile=0.5):
    diff = y_true - y_pred
    sign = (diff >= 0).astype(diff.dtype)
    loss = quantile * sign * diff - (1 - quantile) * (1 - sign) * diff
    return np.mean(loss)


class Objective(BaseObjective):
    name = "L1-regularized Quantile Regression"

    parameters = {
        'reg': [0.05, .1, .5],
        'quantile': [0.2, 0.5, 0.8]
    }

    def __init__(self, reg, quantile):
        self.reg = reg
        self.quantile = quantile

    def set_data(self, X, y):
        self.X, self.y = X, y

    def compute(self, params):
        intercept = params[0]
        beta = params[1:]
        y_pred = self.X.dot(beta) + intercept
        l1 = np.sum(np.abs(beta))
        return pin_ball_loss(self.y, y_pred, self.quantile) + self.reg * l1

    def to_dict(self):
        return dict(X=self.X, y=self.y, reg=self.reg, quantile=self.quantile)
