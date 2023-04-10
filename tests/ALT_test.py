from ALT_opt import ComparisonOptimizers
import matplotlib.pyplot as plt
import numpy as np


def f(X, noise=0):
    return -np.sin(5*X) - X**2 + 0.7*X + noise * np.random.randn(*X.shape)

def g(X, noise=0):
    X = np.sum(X, axis=1).reshape(-1, 1)
    return -np.sin(5*X) - X**2 + 0.7*X + noise * np.random.randn(*X.shape)


def comp():
    # bounds = np.array([[-1.0, 2.0], [-2.0, 1.0]])
    bounds = np.array([[-1.0, 2.0]])
    alt = ComparisonOptimizers(f, 1, method="random", bounds=bounds, n_iter=10, n_init=2, noise_std=0, n_stop_iter=4)
    # alt = ComparisonOptimizers(g, 2, method="l-bfgs-b", bounds=bounds, n_iter=60, n_init=6, noise_std=0, n_stop_iter=4)
    alt.run()
    alt.make_plots()

    plt.show()


if __name__ == "__main__":
    comp()