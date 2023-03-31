import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.optimize import minimize
from BO_util import plot_sampled_points2D, plot_sampled_points3D
from acquisition import pre_acquisition
from Model import GPmodel

from sklearn.gaussian_process import GaussianProcessRegressor as GPR
from sklearn.gaussian_process.kernels import Matern, ConstantKernel

class ComparisonOptimizers:
    def __init__(self, opt_func, dim, method="random", bounds=None, noise_std=1e-5, init_points = None, n_init=5, n_iter=50, n_restarts=20, random_state=1234, n_stop_iter=5):
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

        # TODO: add init_points logic, do so for BO as well
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
    
    def random_sampling_search(self):
        
        # sample one point randomly
        def next_sample_random():
            X_next = np.random.uniform(self.bounds[:, 0], self.bounds[:, 1], size=(1, self.dim))
            return X_next

        for i in range(self.n_iter):
            X_next = next_sample_random()
            Y_next = self.opt_func(X_next)
            self.update_samples(X_next, Y_next)
            _ = self.update_opt()

            # TODO: stopping criteria ???

    def quasi_newton_search(self):
        # sample from quasi newton
        
        n_func_eval = 0
        def min_func(x):
            x = x.reshape(-1, self.dim)
            y = -self.opt_func(x)
            return y

        def callback(x):
            self.update_samples(x.reshape(-1, self.dim), -min_func(x))
            _ = self.update_opt()
        
        # TODO: should I really restart?
        for x0 in np.random.uniform(self.bounds[:, 0], self.bounds[:, 1], size=(self.n_init, self.dim)):
            self.update_samples(x0.reshape(-1, self.dim), -min_func(x0))
            res = minimize(min_func, x0, method=self.method, bounds=self.bounds, callback=callback, options={'maxiter': self.n_iter})
            n_func_eval += res.nfev
            # if res.fun < min_val:
            #     min_val = res.fun
            #     min_x = res.x
    
    def run(self):
        # run optimization
        if self.method == "random":
            self.random_sampling_search()
        elif self.method == "l-bfgs-b":
            self.quasi_newton_search()
        else:
            raise ValueError("Method not implemented")

        # print optimal value and point
        print("Optimal value found: ", self.opt_val)
        print("Optimal point found: ", self.opt_x)

    def make_plots(self, save=False, save_path=None):
        # make plots
        n_plots = len(self.X_samples)-self.n_init
        if self.dim == 1:
            plt.figure(figsize=(12, n_plots * 3))
            plt.subplots_adjust(hspace=0.4)
            X = np.arange(self.bounds[:, 0], self.bounds[:, 1], 0.01).reshape(-1, 1)
            Y = self.f(X).reshape(-1,1)
            opts = np.array([])
            for i in range(n_plots):
                X_samples = self.X_samples[:i, :]
                Y_samples = self.Y_samples[:i, :]
                X_next = self.X_samples[i, :]

                plt.subplot(math.ceil(n_plots/2), 2, i+1)
                plot_sampled_points2D(X, Y, X_samples, Y_samples, X_next=X_next)

                opt, _ = self.compute_opt(X_samples, Y_samples)
                opts = np.append(opts, opt)
        elif self.dim == 2:
            X1, X2 = np.meshgrid(*[np.arange(bound[0], bound[1], 0.01) for bound in self.bounds])
            Y = self.f(X).reshape(X1.shape)
            
            n_plots = (len(self.X_samples)-self.n_init)//10
            opts = np.array([])
            for i in range(n_plots):
                X_samples = self.X_samples[:i*10, :]
                Y_samples = self.Y_samples[:i*10, :]
                X_next = self.X_samples[i*10, :]

                opt, _ = self.compute_opt(X_samples, Y_samples)
                opts = np.append(opts, opt)

                fig = plt.figure()
                ax = fig.add_subplot(111, projection='3d')
                plot_sampled_points3D(X1, X2, Y, X_samples, Y_samples, X_next=X_next, ax=ax)

        else:
            raise ValueError("Dimension not implemented")
        
        if save:
            if save_path is None:
                acq_path = "comp_sampling_plots.png"
            else:
                acq_path = save_path + "_sampling.png"
            plt.savefig(acq_path)

        plt.figure()
        plt.plot(opts)
        plt.title('Opt over iterations')
        plt.xlabel('Iteration')
        plt.ylabel('Opt')

        if save:
            if save_path is None:
                opt_path = "comp_opt_plots.png"
            else:
                opt_path = save_path + "_opt.png"
            plt.savefig(opt_path)
