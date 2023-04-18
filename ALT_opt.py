import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from src.visualization.BO_util import plot_sampled_points2D, plot_sampled_points3D

class ComparisonOptimizers:
    def __init__(self, opt_func, dim, method="random", bounds=None, noise_std=1e-5, init_points = None, n_init=5, n_iter=50, plotting_freq=None, random_state=1234, n_stop_iter=5):
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

        if plotting_freq is not None:
            self.plotting_freq = plotting_freq
        else:
            if self.dim == 1:
                self.plotting_freq = 1
            elif self.dim == 2:
                self.plotting_freq = 5

        self.n_init = n_init
        self.n_iter = n_iter

        self.noise_std = noise_std

        self.random_state = random_state
        self.n_stop_iter = n_stop_iter

        self.opt_val = np.inf
        self.opt_x = None

        self.init_points = init_points
        if self.init_points is not None:
            if self.init_points.shape[1] != self.dim:
                raise ValueError("Dimension of init_points is not equal to dimension of input")
            # if init_points is passed, then n_init is set to the number of init_points
            self.n_init = self.init_points.shape[0]
            
        self.X_samples = None
        self.Y_samples = None

        self.n_evals = None
    
    def init_samples(self, compute_y=True):
        # initialize samples
        if self.init_points is not None:
            self.X_samples = self.init_points
            if compute_y:
                self.Y_samples = self.opt_func(self.X_samples)
        else:
            self.X_samples = np.random.uniform(self.bounds[:, 0], self.bounds[:, 1], size=(self.n_init, self.dim))
            if compute_y:
                self.Y_samples = self.opt_func(self.X_samples)
        # compute opts
        _ = self.update_opt()

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

        opt_val_ind = np.argmin(Y_samples)
        opt_val = Y_samples[opt_val_ind]    
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

        self.init_samples(compute_y=True)

        for i in range(self.n_iter):
            X_next = next_sample_random()
            Y_next = self.opt_func(X_next)
            self.update_samples(X_next, Y_next)
            _ = self.update_opt()

        return len(self.X_samples)

    def quasi_newton_search(self):
        # sample from quasi newton
        
        self.init_samples(compute_y=True)
        # Will not count these as function evaluations as they will be done in the optimization loop
        # these here are for us.
        n_func_eval = 0
        def min_func(x):
            nonlocal n_func_eval
            x = x.reshape(-1, self.dim)
            y = self.opt_func(x).reshape(-1)
            n_func_eval += 1
            return y

        def callback(x):
            self.update_samples(x.reshape(-1, self.dim), min_func(x))
            _ = self.update_opt()
        
        X_inits = np.copy(self.X_samples)
        allowed_iters = self.n_iter // len(X_inits)
        for x0 in X_inits:
            res = minimize(min_func, x0, method=self.method, bounds=self.bounds, callback=callback, options={'maxiter': allowed_iters})
            # n_func_eval += res.nfev
        
        # print("Number of function evaluations: ", n_func_eval)
        return n_func_eval
    
    def run(self):
        # run optimization
        if self.method.lower() == "random":
            func_evals = self.random_sampling_search()
        elif self.method.lower() == "l-bfgs-b":
            func_evals = self.quasi_newton_search()
        else:
            raise ValueError("Method not implemented")

        # # print optimal value and point
        # print("Method used: ", self.method)
        # print("Optimal value found: ", self.opt_val)
        # print("Optimal point found: ", self.opt_x)
        # print("Number of function evaluations: ", func_evals)
        self.n_evals = func_evals

    def make_plots(self, save=False, save_path=None):
        # make plots
        n_plots = (len(self.X_samples)-self.n_init)//self.plotting_freq
        if self.dim == 1:
            fig = plt.figure(figsize=(12, n_plots * 3))
            fig.subplots_adjust(hspace=0.4)
            fig.suptitle(f'Optimization using {self.method.lower()}', fontsize=16)
            X = np.arange(self.bounds[:, 0], self.bounds[:, 1], 0.01).reshape(-1, 1)
            Y = self.opt_func(X).reshape(-1, 1)
            opts = np.array([])
            for i in range(n_plots):
                elem_i = self.n_init*self.plotting_freq + i
                X_samples = self.X_samples[:elem_i, :]
                Y_samples = self.Y_samples[:elem_i, :]
                X_next = self.X_samples[elem_i, :]

                plt.subplot(math.ceil(n_plots/2), 2, i+1)
                plot_sampled_points2D(X, Y, X_samples, Y_samples, X_next=X_next)
                plt.title(f'Iteration {i+1}')

                opt, _ = self.compute_opt(X_samples, Y_samples)
                opts = np.append(opts, opt)
        elif self.dim == 2:
            X1, X2 = np.meshgrid(*[np.arange(bound[0], bound[1], 0.01) for bound in self.bounds])
            X = np.hstack((X1.reshape(-1, 1), X2.reshape(-1, 1)))
            Y = self.opt_func(X).reshape(X1.shape)
            
            n_plots = (len(self.X_samples)-self.n_init)//self.plotting_freq
            opts = np.array([])
            for i in range(n_plots):
                print(f"Plotting iteration {i+1}")
                elem_i = i*self.plotting_freq + self.n_init
                X_samples = self.X_samples[:elem_i, :]
                Y_samples = self.Y_samples[:elem_i, :]
                X_next = self.X_samples[elem_i, :]

                opt, _ = self.compute_opt(X_samples, Y_samples)
                opts = np.append(opts, opt)

                fig = plt.figure()
                ax = fig.add_subplot(111, projection='3d')
                plot_sampled_points3D(X1, X2, Y, X_samples, Y_samples, X_next=X_next, ax=ax)
                ax.set_title(f'Iteration {i+1}')

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
        plt.title('Change of optimal value over iterations')
        plt.xlabel('Iterations')
        plt.ylabel('Optimal value')

        if save:
            if save_path is None:
                opt_path = "comp_opt_plots.png"
            else:
                opt_path = save_path + "_opt.png"
            plt.savefig(opt_path)
