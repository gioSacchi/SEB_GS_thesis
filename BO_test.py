from BO import BayesianOptimization
import matplotlib.pyplot as plt

import numpy as np

def f(X, noise=0):
    return -np.sin(5*X) - X**2 + 0.7*X + noise * np.random.randn(*X.shape)

def g(X, noise=0):
    X = np.sum(X, axis=1).reshape(-1, 1)
    return -np.sin(5*X) - X**2 + 0.7*X + noise * np.random.randn(*X.shape)

def bo_test():
    # bounds = np.array([[-1.0, 2.0], [-2.0, 1.0]])
    bounds = np.array([[-1.0, 2.0]])
    # bo = BayesianOptimization(g, 2, bounds=bounds, n_iter=42, n_init=25, noise_std=0)
    bo = BayesianOptimization(f, 1, bounds=bounds, n_iter=10, n_init=2, noise_std=0)
    bo.run_BO()
    bo.make_plots()

    plt.show()

if __name__ == "__main__":
    bo_test()