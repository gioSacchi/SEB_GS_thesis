from BO import BayesianOptimization
import matplotlib.pyplot as plt
from scipy.optimize import minimize, show_options
import sys

import numpy as np

def f(X, noise=0):
    return -(-np.sin(5*X) - X**2 + 0.7*X + noise * np.random.randn(*X.shape))

def g(X, noise=0):
    X = np.sum(X, axis=1).reshape(-1, 1)
    return -np.sin(5*X) - X**2 + 0.7*X + noise * np.random.randn(*X.shape)

def bo_test():
    # bounds = np.array([[-1.0, 2.0], [-2.0, 1.0]])
    bounds = np.array([[-1.0, 2.0]])
    # bo = BayesianOptimization(g, 2, bounds=bounds, n_iter=42, n_init=4, noise_std=0, normalize_Y=True, n_stop_iter=40)
    bo = BayesianOptimization(f, 1, bounds=bounds, n_iter=5, n_init=3, noise_std=0, n_stop_iter=None, normalize_Y=True, n_opt=2000, n_restarts=100, random_state=32)
    bo.run_BO()
    bo.make_plots(save=True)

    plt.show()

def lbfgs_test():
    # sys.stdout = open(r"goat.txt", "w")
    # filename  = open(r"outputfile.txt",'w')
    # sys.stdout = filename
    # print ("test sys.stdout")
    np.random.seed(1235)
    # bounds = np.array([[-1.0, 2.0], [-2.0, 1.0]])
    bounds = np.array([[-1.0, 2.0]])
    # bo = BayesianOptimization(g, 2, bounds=bounds, n_iter=42, n_init=4, noise_std=0)
    # bo = BayesianOptimization(f, 1, bounds=bounds, n_iter=10, n_init=2, noise_std=0)
    x0 = np.random.uniform(bounds[:, 0], bounds[:, 1], size=(1, bounds.shape[0]))
    print(x0, f(x0))
    
    # peform the minimization and save the evaluated points of each iteration in a list using callback
    x_list = []
    def callback(x):
        x_list.append(x)

    res = minimize(f, x0, method='L-BFGS-B', bounds=bounds, callback=callback, options={'maxiter': -1, 'disp': False})
    print(res)
    print(x_list)
    # print(show_options('L-BFGS-B'))


if __name__ == "__main__":
    bo_test()
    # lbfgs_test()