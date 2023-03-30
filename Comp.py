import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.optimize import minimize
from BO_util import plot_surrogate_approx2D, plot_obj_approx2D, plot_acquisition, plot_convergence, plot_obj_approx3D, plot_surrogate_approx3D
from acquisition import pre_acquisition
from Model import GPmodel

from sklearn.gaussian_process import GaussianProcessRegressor as GPR
from sklearn.gaussian_process.kernels import Matern, ConstantKernel

class ComparisonOptimizers:
    def __init__(self, opt_func, dim, method="random", bounds=None, noise_std=1e-5, n_init=5, n_iter=50, n_restarts=20, random_state=1234, n_stop_iter=5):
        #set seed
        np.random.seed(random_state)
        # function to optimize
        self.opt_func = opt_func
        # optimization method
        methods = ["random", "grid", "quasi-newton", "L-BFGS-B"]
        if method not in methods:
            # get standard methods
            pass
        else:
            # custome method, need to be ...
            pass
        self.method = method

        # dimension of input
        self.dim = dim
        self.bounds = bounds
        # check bounds
        if self.bounds is not None:
            if bounds.shape[0] != self.dim:
                raise ValueError("Dimension of bounds is not equal to dimension of input")

        self.n_init = n_init
        self.n_iter = n_iter

        self.noise_std = noise_std

        self.n_restarts = n_restarts
        self.random_state = random_state
        self.n_stop_iter = n_stop_iter

        self.opt_val = None
        self.opt_x = None

        self.X_samples = None
        self.Y_samples = None
    
    def update_samples(self, X_next, Y_next):
        # update samples incl. objective function values if available
        self.X_samples = np.vstack((self.X_samples, X_next))
        self.Y_samples = np.vstack((self.Y_samples, Y_next))

    def compute_opt(self, X_samples=None, Y_samples=None):
        # if not passed, use samples from self
        if X_samples is None:
            X_samples = self.X_samples
        if Y_samples is None:
            Y_samples = self.Y_samples

        # if we have objective function use that for opt_val, otherwise use Y
        # if noisy then use mean of GP
        # if not self.noisy_evaluations:
        #     opt_val_ind = np.argmin(Y_samples)
        #     opt_val = Y_samples[opt_val_ind]
        # else:
        pred_mu = self.model.predict(X_samples)[0]
        opt_val_ind = np.argmin(pred_mu)
        opt_val = pred_mu[opt_val_ind]
    
        opt_x = X_samples[opt_val_ind, :].reshape(-1, self.dim)
        
        return opt_val, opt_x

    def update_opt(self):
        # update opt_val
        new_opt, new_x = self.compute_opt()
        if new_opt < self.opt_val:
            # opt_val init as inf, so this will always be true for first iteration
            self.opt_val = new_opt
            self.opt_x = new_x.reshape(-1, self.dim)
            return True
        elif new_opt == self.opt_val and self.opt_x is not None:
            # if point has same value and is not in opt_x, add it
            abs_row_diff = np.abs(self.opt_x - new_x)
            tot_row_diff = np.sum(abs_row_diff, axis=1)
            if np.abs(tot_row_diff) > 1e-3:
                self.opt_x = np.vstack((self.opt_x, new_x))
                return True
        
        return False
    
    def next_sample_random(self):
        # sample from uniform distribution
        X_next = np.random.uniform(self.bounds[:, 0], self.bounds[:, 1], size=(1, self.dim))
        return X_next
    
    def next_sample_grid(self):
        # sample from grid
        pass

    def next_sample_quasi_newton(self):
        # sample from quasi newton
        
        min_val = np.inf
        min_x = None

        def min_func(x):
            x = x.reshape(-1, self.dim)
            y = -self.opt_func(x)
            return y
        
        for x0 in np.random.uniform(self.bounds[:, 0], self.bounds[:, 1], size=(self.n_restarts, self.dim)):
            res = minimize(min_func, x0, method=self.method, bounds=self.bounds, options={'maxiter': 1000, 'disp': True})
            if res.fun < min_val:
                min_val = res.fun
                min_x = res.x
        
        return min_x.reshape(-1, self.dim), -min_val