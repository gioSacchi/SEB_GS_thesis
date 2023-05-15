import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from src.visualization.BO_util import plot_surrogate_approx2D, plot_obj_approx2D, plot_acquisition, plot_convergence, plot_obj_approx3D, plot_surrogate_approx3D
from src.features.acquisition import pre_acquisition
from src.features.Model import GPmodel
import time

class BayesianOptimization:
    def __init__(self, f, dim, bounds, obj_func = None, acquisition='EI', kernel=None, noise_std=1e-5, 
                 n_init=2, n_iter=10, n_opt = None, normalize_Y=True, random_state=1234, n_stop_iter=2,
                 acq_threshold=0.01, init_points=None, plotting_freq=None, n_restarts=20):
        # set seed
        np.random.seed(random_state)
        self.normalize_Y = normalize_Y

        self.f = f
        # if f is not the function to minimize, obj_func is the function to minimize
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
        if bounds.shape[0] != self.dim:
            raise ValueError('Bounds dimension does not match the dimension of the data')
        self.bounds = bounds

        # check init_points dims
        if init_points is not None:
            # check if np array or touple
            if isinstance(init_points, np.ndarray):
                # these are the X points to initialize with
                if init_points.shape[1] != self.dim:
                    raise ValueError('Init_points dimension does not match the dimension of the data')
                self.n_init = init_points.shape[0]
            elif isinstance(init_points, tuple):
                # these are X and Y points to initialize with
                if init_points[0].shape[1] != self.dim:
                    raise ValueError('Init_points dimension does not match the dimension of the data')
                if init_points[1].shape[1] != 1:
                    raise ValueError('Init_points dimension does not match the dimension of the data')
                if init_points[0].shape[0] != init_points[1].shape[0]:
                    raise ValueError('Init_points dimension does not match the dimension of the data')
                self.n_init = init_points[0].shape[0]
        else:
            self.n_init = n_init

        if plotting_freq is not None:
            self.plotting_freq = plotting_freq
        else:
            if self.dim == 1:
                self.plotting_freq = 1
            elif self.dim == 2:
                self.plotting_freq = 5

        self.n_iter = n_iter
        if n_opt is None:
            # make n_opt i proportional to area of bounds
            area = np.prod(np.diff(bounds, axis=1))
            self.n_opt = int(np.ceil(20 * area))
        else:
            self.n_opt = n_opt

        self.opt_val = np.inf
        self.opt_x = None
        self.acq_threshold = acq_threshold

        # for stopping BO loop
        self.stop = False 
        if n_stop_iter is None:
            self.n_stop_iter = np.inf
        else:
            self.n_stop_iter = n_stop_iter

        # initialize GP
        self.kernel = kernel
        self.noisy_evaluations = True if noise_std > 1e-5 else False
        self.noise_std = noise_std
        self.model = GPmodel(kernel=kernel, noise=noise_std, normalize_Y=normalize_Y, n_restarts=n_restarts)

        # initialize samples
        self.X_samples = None
        self.Y_samples = None
        self.obj_samples = None
        self.number_of_evaluations_f = 0
        self.init_samples(init_points=init_points)

    def init_samples(self, init_points=None):
        # check init_points dims and initialize samples
        if init_points is not None:
            # check if np array or touple
            if isinstance(init_points, np.ndarray):
                X_samples = init_points
                Y_samples = self.f(X_samples)
                self.number_of_evaluations_f += self.n_init
            elif isinstance(init_points, tuple):
                X_samples = init_points[0]
                Y_samples = init_points[1]
                # do not update number of evaluations because these are not new evaluations of f which is expensive!!
        else:
            # initialize samples by sampling Xs uniformly from the bounds and computing Ys
            X_samples = np.random.uniform(self.bounds[:, 0], self.bounds[:, 1], size=(self.n_init, self.bounds.shape[0]))
            Y_samples = self.f(X_samples)
            self.number_of_evaluations_f += self.n_init
        # check output dimension, if (n_init,) then reshape to (n_init, 1)
        if Y_samples.shape == (self.n_init,):
            Y_samples = Y_samples.reshape(-1,1)
        # update samples
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
                opt_val_ind = np.argmin(obj_samples)
                opt_val = obj_samples[opt_val_ind]
            else:
                opt_val_ind = np.argmin(Y_samples)
                opt_val = Y_samples[opt_val_ind]
        else:
            pred_mu = self.model.predict(X_samples)[0]
            if self.obj_func is not None:
                values = self.obj_func(X_samples, pred_mu)
                opt_val_ind = np.argmin(values)
                opt_val = values[opt_val_ind]
            else:
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

    def next_sample(self):
        # avg_acq_time = []
        # init min value and min x for optimization of acquisition function
        min_val = np.inf
        min_x = None

        # define objective function for optimization, minus of acquisition function 
        # since we want to maximize the acquisition function but use minimizer
        def min_obj(x):
            # t1 = time.time()
            acq = -self.acquisition(x.reshape(-1, self.dim), self.model, self.opt_val).flatten()
            # t2 = time.time()
            # avg_acq_time.append(t2-t1)
            return acq
        
        # optimization loop, initialize x0 randomly n_opt times and do optimization for each x0
        for x0 in np.random.uniform(self.bounds[:, 0], self.bounds[:, 1], size=(self.n_opt, self.bounds.shape[0])):
            res = minimize(min_obj, x0=x0, bounds=self.bounds, method='L-BFGS-B')
            # update min value and min x if the current x0 gives better result
            if res.fun < min_val:
                min_val = res.fun[0]
                min_x = res.x
        
        # print('min_val: ', min_val)
        # print('min_x: ', min_x)

        # print('avg acq time: ', np.sum(avg_acq_time))
        # print(len(avg_acq_time))

        # store next sample
        X_next = min_x.reshape(-1, self.dim)
        Y_next = self.f(X_next)
        # update number of evaluations
        self.number_of_evaluations_f += 1

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
        return X_next, Y_next, obj_next, -min_val
    
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
    
    def stop_BO(self, acq_val, opt_iter, updated):
        # # if opt_val has not improved for n_stop_iter iterations, stop BO
        # current_iter = self.X_samples.shape[0]-self.n_init
        # if current_iter - opt_iter >= self.n_stop_iter:
        #     self.stop = True

        # stop if the ratio of acquisition function to optimal value is below threshold
        # stop is false if ratio is above threshold, number has been below threshold for at most n_stop_iter iterations
        # and is true if ratio has been below threshold for n_stop_iter iterations, i.e. stop
        ratio = acq_val/self.opt_val
        # print('ratio: ', ratio)
        if ratio < self.acq_threshold:
            if self.stop is False:
                self.stop = 1
            elif 1<=self.stop<self.n_stop_iter-1:
                self.stop += 1
            else:
                self.stop = True
        else:
            self.stop = False

    def run_BO(self):
        opt_iter = 0
        # run BO loop
        for i in range(self.n_iter):
            print('Loop it: ', i)
            # fit GP to samples
            # t1 = time.time()
            self.model.fit(self.X_samples, self.Y_samples)
            # print('Time to fit GP: ', time.time()-t1)
            # get next sample
            # t1 = time.time()
            X_next, Y_next, obj_next, acq_val = self.next_sample()
            # print('Time to get next sample: ', time.time()-t1)
            # update samples
            self.update_samples(X_next, Y_next, obj_next)
            # update opt_val
            updated = self.update_opt()
            if updated:
                opt_iter = i
            # check if we should stop BO
            self.stop_BO(acq_val, opt_iter, updated)
            if self.stop is True:
                # only stop when self.stop is True, not 1 or 2 etc
                print('BO stopped after {} iterations'.format(i))
                break
        
        # # print final results
        # print('Final opt_val: ', self.opt_val)
        # print('Final opt_x: ', self.opt_x)
    
    def make_plots(self, y_prim=None, save=False, save_path=None):
        # call correct function based on dim

        # y_prim necessary for objective function plots
        if self.obj_func is not None and y_prim is None:
            raise ValueError('y_prim is necessary for objective function plots')
        
        if self.dim == 1:
            self._make_plots(y_prim, save=save, save_path=save_path)
        elif self.dim == 2:
            self._make_3D_plots(y_prim, save=save, save_path=save_path)
        else:
            print('Cannot make plots for dim > 2')

    def _make_3D_plots(self, y_prim, save=False, save_path=None):
        # make grid of point between bounds with step 0.01 and format it to 2D array with shape (n_samples, dim)
        X1, X2 = np.meshgrid(*[np.arange(bound[0], bound[1], 0.01) for bound in self.bounds])
        X = np.c_[X1.ravel(), X2.ravel()]
        Y = self.f(X).reshape(X1.shape)
        if self.obj_func is not None:
            obj = self.obj_func(X, Y.reshape(-1,1)).reshape(X1.shape)
            plot_ind = 1
        else:
            obj = None
            plot_ind = 0

        n_plots = (len(self.X_samples)-self.n_init)//self.plotting_freq
        # fig = plt.figure(figsize=(12, n_plots * 3))
        # plt.subplots_adjust(hspace=0.4)

        model = GPmodel(kernel=self.kernel, noise=self.noise_std, normalize_Y=self.normalize_Y)
        opts = np.array([])
        for i in range(n_plots):
            print('Plotting iteration {}'.format(i))
            fig = plt.figure(figsize=(12, 24))
            plt.subplots_adjust(hspace=0.4)
            elem_i = i*self.plotting_freq + self.n_init
            X_samples = self.X_samples[:elem_i, :]
            Y_samples = self.Y_samples[:elem_i, :]
            if self.obj_func is not None:
                obj_samples = self.obj_samples[:elem_i, :]
            else:
                obj_samples = None
            model.fit(X_samples, Y_samples)
            X_next = self.X_samples[elem_i, :]

            opt_val, opt_x = self.compute_opt(X_samples, Y_samples, obj_samples)
            opts = np.append(opts, opt_val)
            opt_tup = (opt_x, opt_val) # tuple of opt_x and opt_val for plotting

            # if (i+1)%10 == 0:
            if self.obj_func is not None:
                # ax = fig.add_subplot(n_plots, 3, 3*i + 1, projection='3d')
                ax = fig.add_subplot(1, 3, 1, projection='3d')
                plot_obj_approx3D(model, X, X1, X2, Y, X_samples, Y_samples, self.obj_func, ax, y_prim, X_next=X_next, 
                                    obj=obj, obj_sample=obj_samples, opt=opt_tup)
                opt_tup = None # So that opt is not plotted twice
            
            # ax = fig.add_subplot(n_plots, (2+plot_ind), (plot_ind + 2)*i + 1 + plot_ind, projection='3d')
            ax = fig.add_subplot(1, 2 + plot_ind, 1 + plot_ind, projection='3d')
            plot_surrogate_approx3D(model, X, X1, X2, Y, X_samples, Y_samples, ax, X_next=X_next, opt=opt_tup)
            ax.set_title('Iteration {}'.format(elem_i+1))

            # ax = fig.add_subplot(n_plots, (2+plot_ind), (plot_ind + 2)*i + 2 + plot_ind, projection='3d')
            ax = fig.add_subplot(1, 2 + plot_ind, 2 + plot_ind, projection='3d')
            plot_acquisition(np.array([X1, X2]), self.acquisition(X, model, opt_val).reshape(X1.shape), X_next=X_next, 
                                ax=ax)
        
        if save:
            if save_path is None:
                acq_path = "BO_acq_plots.png"
            else:
                acq_path = save_path + "_acq.png"
            plt.savefig(acq_path)

        # plot opt over iterations in new figure
        plt.figure()
        plt.plot(np.arange(opts.shape[0]),opts)
        plt.title('Change of optimal value over iterations')
        plt.xlabel('Iterations')
        plt.ylabel('Optimal value')

        if save:
            if save_path is None:
                opt_path = "BO_opt_plots.png"
            else:
                opt_path = save_path + "_opt.png"
            plt.savefig(opt_path)

    def _make_plots(self, y_prim, save=False, save_path=None, plot_final=True):
        # Dense grid of points within bounds
        X = np.arange(self.bounds[:, 0], self.bounds[:, 1], 0.01).reshape(-1, 1)
        Y = self.f(X).reshape(-1,1)

        if self.obj_func is not None:
            obj = self.obj_func(X, Y).reshape(-1,1)
            plot_ind = 1
        else:
            obj = None
            plot_ind = 0
        if plot_final:
            add_rows = 1
        else:
            add_rows = 0

        n_plots = (len(self.X_samples)-self.n_init)//self.plotting_freq
        fig = plt.figure(figsize=(12, n_plots * 3))
        fig.subplots_adjust(hspace=0.4)
        fig.suptitle('Bayesian Optimization', fontsize=16)
        model = GPmodel(kernel=self.kernel, noise=self.noise_std, normalize_Y=self.normalize_Y)
        opts = np.array([])
        for i in range(n_plots):
            elem_i = i*self.plotting_freq + self.n_init
            X_samples = self.X_samples[:elem_i, :]
            Y_samples = self.Y_samples[:elem_i, :]
            if self.obj_func is not None:
                obj_samples = self.obj_samples[:elem_i, :]
            else:
                obj_samples = None
            model.fit(X_samples, Y_samples)
            # print paramters of model
            # print(f'Iteration {i+1}')
            # print(f'Kernel parameters: {model.gpr.kernel_.get_params()}')
            X_next = self.X_samples[elem_i, :]
            
            opt_val, opt_x = self.compute_opt(X_samples, Y_samples, obj_samples)
            opts = np.append(opts, opt_val)
            opt_tup = (opt_x, opt_val) # tuple of opt_x and opt_val for plotting

            if self.obj_func is not None:
                plt.subplot(n_plots + add_rows, 3, 3 * i + 1)
                plot_obj_approx2D(model, X, Y, X_samples, Y_samples, X_next=X_next, obj_func=self.obj_func, y_prim=y_prim,
                                    obj=obj, obj_sample=obj_samples, opt=opt_tup, show_legend=i==0)
                opt_tup = None # So that opt is not plotted twice

            plt.subplot(n_plots + add_rows, 2+plot_ind, (2+plot_ind) * i + 1 + plot_ind)
            plot_surrogate_approx2D(model, X, Y, X_samples, Y_samples, X_next=X_next, opt=opt_tup, show_legend=i==0)
            plt.title(f'Iteration {i+1}')

            plt.subplot(n_plots + add_rows, 2+plot_ind, (2+plot_ind) * i + 2 + plot_ind)
            plot_acquisition(X, self.acquisition(X, model, opt_val), X_next, show_legend=i==0)
        
        if plot_final:
            # plot final model using all samples
            X_samples = self.X_samples
            Y_samples = self.Y_samples
            if self.obj_func is not None:
                obj_samples = self.obj_samples
            else:
                obj_samples = None
            model.fit(X_samples, Y_samples)
            # print paramters of model
            # print(f'Iteration {i+1}')
            # print(f'Kernel parameters: {model.gpr.kernel_.get_params()}')
            
            opt_val, opt_x = self.compute_opt(X_samples, Y_samples, obj_samples)
            opts = np.append(opts, opt_val)
            opt_tup = (opt_x, opt_val) # tuple of opt_x and opt_val for plotting

            if self.obj_func is not None:
                plt.subplot(n_plots + add_rows, 3, 3 * n_plots + 1)
                plot_obj_approx2D(model, X, Y, X_samples, Y_samples, obj_func=self.obj_func, y_prim=y_prim,
                                    obj=obj, obj_sample=obj_samples, opt=opt_tup, show_legend=False)
                opt_tup = None # So that opt is not plotted twice

            plt.subplot(n_plots + add_rows, 2+plot_ind, (2+plot_ind) * n_plots + 1 + plot_ind)
            plot_surrogate_approx2D(model, X, Y, X_samples, Y_samples, opt=opt_tup, show_legend=False)
            plt.title(f'Final')
        
        if save:
            if save_path is None:
                acq_path = "BO_acq_plots.png"
            else:
                acq_path = save_path + "_acq.png"
            plt.savefig(acq_path)
            
        # plot opt over iterations in new figure
        plt.figure()
        plt.plot(opts)
        plt.title('Change of optimal value over iterations')
        plt.xlabel('Iterations')
        plt.ylabel('Optimal value')

        if save:
            if save_path is None:
                opt_path = "BO_opt_plots.png"
            else:
                opt_path = save_path + "_opt.png"
            plt.savefig(opt_path)
