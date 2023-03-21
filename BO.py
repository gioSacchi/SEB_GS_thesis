import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.optimize import minimize
from BO_util import plot_approximation2D, plot_acquisition, plot_convergence, plot_approximation3D
from acquisition import pre_acquisition
from Model import GPmodel

from sklearn.gaussian_process import GaussianProcessRegressor as GPR
from sklearn.gaussian_process.kernels import Matern, ConstantKernel

class BayesianOptimization:
    def __init__(self, f, dim, obj_func = None, acquisition='EI', kernel=None, noise_std=1e-5, bounds=None, n_init=2, n_iter=10, n_opt = 50, random_state=1234, n_stop_iter=4):
        # set seed
        np.random.seed(random_state)
        self.f = f
        # if f is not the objective function, obj_func is the objective function
        # obj_func should be callable function that takes in X and f(X) (or surrogate of f) as input and returns the objective function value
        self.obj_func = obj_func

        acqiuisitions = ['EI', 'PI', 'UCB', 'LCB']
        if acquisition in acqiuisitions:
            # Acqiuisition function is one of the predefined functions, will return callable function
            self.acquisition = self.get_acquisition(acquisition)
        else:
            # Custome acqiuisition function is passed and should be callable function
            # takes as input X, model, opt, xi (optional)
            self.acquisition = acquisition

        self.dim = dim
        # check bounds dims
        if bounds is not None:
            if bounds.shape[0] != self.dim:
                raise ValueError('Bounds dimension does not match the dimension of the data')

        self.bounds = bounds
        self.n_init = n_init
        self.n_iter = n_iter
        self.n_opt = n_opt

        self.noisy_evaluations = True if noise_std > 0 else False
        self.opt_val = None

        # for stopping BO loop
        self.stop = False 
        self.n_stop_iter = n_stop_iter

        # initialize samples
        self.X_samples = None
        self.Y_samples = None
        self.obj_samples = None
        self.init_samples()
        # initialize GP
        self.kernel = kernel
        self.noise_std = noise_std
        self.model = GPmodel(kernel=kernel, noise=noise_std)

    def init_samples(self):
        # initialize samples by sampling Xs uniformly from the bounds and computing Ys
        X_samples = np.random.uniform(self.bounds[:, 0], self.bounds[:, 1], size=(self.n_init, self.bounds.shape[0]))
        Y_samples = self.f(X_samples)
        # check output dimension, if (n_init,) then reshape to (n_init, 1)
        if Y_samples.shape == (self.n_init,):
            Y_samples = Y_samples.reshape(-1,1)
        self.X_samples = X_samples
        self.Y_samples = Y_samples
        # if we have objective function, compute objective function values
        if self.obj_func is not None:
            self.obj_samples = self.obj_func(X_samples, Y_samples)
        # compute opt_val
        _ = self.update_opt()

    def get_acquisition(self, acquisition: str):
        # return callable function for acquisition function
        return pre_acquisition().get_standard_acquisition(acquisition)

    def update_samples(self, X_next, Y_next, obj_next=None):
        # update samples incl. objective function values if available
        self.X_samples = np.vstack((self.X_samples, X_next))
        self.Y_samples = np.vstack((self.Y_samples, Y_next))
            
        if obj_next is not None:
            self.obj_samples = np.vstack((self.obj_samples, obj_next))
        
    def compute_opt(self, X_samples=None, Y_samples=None, obj_samples=None):
        # if not passed, use samples from self
        if X_samples is None:
            X_samples = self.X_samples
        if Y_samples is None:
            Y_samples = self.Y_samples
        if obj_samples is None:
            obj_samples = self.obj_samples

        # if we have objective function use that for opt_val, otherwise use Y
        # if noisy then use mean of GP
        if not self.noisy_evaluations:
            if self.obj_func is not None:
                opt_val = np.min(obj_samples)
            else:
                opt_val = np.min(Y_samples)
        else:
            pred_mu = self.model.predict(X_samples)[0]
            if self.obj_func is not None:
                opt_val = np.min(self.obj_func(X_samples, pred_mu)) # TODO Check this
            else:
                opt_val = np.min(pred_mu)
        
        return opt_val

    def update_opt(self):
        # update opt_val
        new_opt = self.compute_opt()
        if new_opt != self.opt_val:
            self.opt_val = new_opt
            return True
        return False

    def next_sample(self):
        # init min value and min x for optimization of acquisition function
        min_val = 1e100
        min_x = None

        # define objective function for optimization, minus of acquisition function 
        # since we want to maximize the acquisition function but use minimizer
        def min_obj(x):
            return -self.acquisition(x.reshape(-1, self.dim), self.model, self.opt_val).flatten()
        
        # optimization loop, initialize x0 randomly n_opt times and do optimization for each x0
        for x0 in np.random.uniform(self.bounds[:, 0], self.bounds[:, 1], size=(self.n_opt, self.bounds.shape[0])):
            res = minimize(min_obj, x0=x0, bounds=self.bounds, method='L-BFGS-B')
            # update min value and min x if the current x0 gives better result
            if res.fun < min_val:
                min_val = res.fun[0]
                min_x = res.x
        
        print('min_val: ', min_val)
        print('min_x: ', min_x)

        # TODO: add logic for objective function
        # store next sample
        X_next = min_x.reshape(-1, self.dim)
        Y_next = self.f(X_next)
        # check dimension of Y_next
        if Y_next.shape == (1,):
            Y_next = Y_next.reshape(-1,1) 
        # if we have objective function compute
        if self.obj_func is not None:
            obj_next = self.obj_func(X_next, Y_next)
            # check dimension of obj_next
            if obj_next.shape == (1,):
                obj_next = obj_next.reshape(-1,1)
        else:
            obj_next = None
        return X_next, Y_next, obj_next
    
    def check_and_fix_dimenstion(self, X):
        # check if X is 2D array and if not reshape it
        if X.ndim == 1:
            X = X.reshape(1, -1)
            if X.shape[1] != self.dim:
                raise ValueError('X should have same number of coloms as dim, shape of X is {} but should be (n_samples, dim)'.format(X.shape))
        elif X.ndim == 2:
            # check if X has correct shape, ie (n_samples, dim). Try to reshape if not
            if X.shape[1] != self.dim:
                try:
                    X = X.reshape(-1, self.dim)
                except ValueError:
                    raise ValueError('X should have same number of coloms as dim, shape of X is {} but should be (n_samples, dim)'.format(X.shape))
        else:
            raise ValueError('X should be 1D or 2D array with shape (n_samples, dim), but shape is {}'.format(X.shape))
        
        return X
    
    def stop_BO(self, opt_iter):
        # if opt_val has not improved for n_stop_iter iterations, stop BO
        current_iter = self.X_samples.shape[0]-self.n_init
        if current_iter - opt_iter >= self.n_stop_iter:
            self.stop = True

    def run_BO(self):
        opt_iter = 0
        # run BO loop
        for i in range(self.n_iter):
            # fit GP to samples
            self.model.fit(self.X_samples, self.Y_samples)
            # get next sample
            X_next, Y_next, obj_next = self.next_sample()
            # update samples
            self.update_samples(X_next, Y_next, obj_next)
            # update opt_val
            updated = self.update_opt()
            if updated:
                opt_iter = i
            # check if we should stop BO
            self.stop_BO(opt_iter)
            if self.stop:
                break
    
    def make_3D_plots(self):
        # make grid of point between bounds with step 0.01 and format it to 2D array with shape (n_samples, dim)
        X1, X2 = np.meshgrid(*[np.arange(bound[0], bound[1], 0.01) for bound in self.bounds])
        X = np.vstack(map(np.ravel, [X1, X2])).T
        # alt code
        # X = np.vstack([X1.ravel(), X2.ravel()]).T
        # X = np.c_[x_prim.ravel(), y_prim.ravel()]
        # Y = self.f(X,0).reshape(X1.shape)
        Y = self.f(X).reshape(X1.shape)
        
        plt.figure(figsize=(12, self.n_iter * 3))
        plt.subplots_adjust(hspace=0.4)
        model = GPmodel(kernel=self.kernel, noise=self.noise_std)
        for i in range(self.n_iter):
            elem_i = i + self.n_init
            X_samples = self.X_samples[:elem_i, :]
            Y_samples = self.Y_samples[:elem_i, :]
            model.fit(X_samples, Y_samples)
            X_next = self.X_samples[elem_i, :]
            Y_next = self.Y_samples[elem_i, :]
            # Plot samples
            # ax = plt.subplot(self.n_iter, 2, 2 * i + 1, projection='3d')
            if i%10 == 0:
                ax = plt.figure().add_subplot(111, projection='3d')
                plot_approximation3D(model, X, X1, X2, Y, X_samples, Y_samples, ax, X_next, Y_next)
                plt.title(f"Iteration {i + 1}")
        
        plt.show()

    def make_plots(self):
        # Dense grid of points within bounds
        X = np.arange(self.bounds[:, 0], self.bounds[:, 1], 0.01).reshape(-1, 1)

        # Y = self.f(X,0)
        Y = self.f(X)
        plt.figure(figsize=(12, self.n_iter * 3))
        plt.subplots_adjust(hspace=0.4)
        model = GPmodel(kernel=self.kernel, noise=self.noise_std)
        for i in range(self.n_iter):
            elem_i = i + self.n_init
            X_samples = self.X_samples[:elem_i, :]
            Y_samples = self.Y_samples[:elem_i, :]
            model.fit(X_samples, Y_samples)
            X_next = self.X_samples[elem_i, :]
            Y_next = self.Y_samples[elem_i, :]
            # Plot samples, surrogate function, noise-free objective and next sampling location
            plt.subplot(self.n_iter, 2, 2 * i + 1)
            plot_approximation2D(model, X, Y, X_samples, Y_samples, X_next, show_legend=i==0)
            plt.title(f'Iteration {i+1}')

            plt.subplot(self.n_iter, 2, 2 * i + 2)
            opt = self.compute_opt(X_samples, Y_samples)
            plot_acquisition(X, self.acquisition(X, model, opt), X_next, show_legend=i==0)
        plt.show()