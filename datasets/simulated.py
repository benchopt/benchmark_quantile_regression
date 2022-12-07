import numpy as np

from benchopt import BaseDataset


class Dataset(BaseDataset):

    name = "Simulated"

    # List of parameters to generate the datasets. The benchmark will consider
    # the cross product for each key in the dictionary.
    parameters = {
        'n_samples, n_features': [
            (100, 50),
            (1000, 200)
        ]
    }

    def __init__(self, n_samples=10, n_features=50, random_state=27):
        # Store the parameters of the dataset
        self.n_samples = n_samples
        self.n_features = n_features
        self.random_state = random_state

    def get_data(self):
        rng = np.random.RandomState(self.random_state)
        coef = rng.randn(self.n_features)
        X = rng.randn(self.n_samples, self.n_features)
        y = X @ coef + 0.1 * rng.randn(self.n_samples)
        y += 100  # add intercept

        data = dict(X=X, y=y)

        return self.n_features + 1, data
