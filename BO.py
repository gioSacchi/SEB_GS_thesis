import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.optimize import minimize
from sklearn.gaussian_process import GaussianProcessRegressor as GPR
from sklearn.gaussian_process.kernels import Matern, ConstantKernel
from BO_util import plot_approximation, plot_acquisition, plot_convergence
from acquisition import pre_acquisition

class BayesianOptimization:
    def __init__(self, f, obj_func = None, acquisition='EI', kernel=None, noise_std=1e-5, bounds=None, n_init=2, n_iter=10, random_state=1234):
        self.f = f
        # if f is not the objective function, obj_func is the objective function
        # obj_func should be callable function that takes in X and f(X) (or surrogate of f) as input and returns the objective function value
        self.obj_func = obj_func if obj_func != None else f

        acqiuisitions = ['EI', 'PI', 'UCB', 'LCB']
        if acquisition in acqiuisitions:
            # Acqiuisition function is one of the predefined functions, will return callable function
            self.acquisition = self.get_acquisition(acquisition)
        else:
            # Custome acqiuisition function is passed and should be callable function
            self.acquisition = acquisition

        self.bounds = bounds
        self.n_init = n_init
        self.n_iter = n_iter

        # set seed
        np.random.seed(random_state)

        # initialize samples
        self.X_sample = None
        self.Y_sample = None
        self.obj_sample = None
        self.init_sample()

        self.X_next = None
        self.Y_next = None

        # initialize GP
        if kernel == None:
            self.kernel = ConstantKernel(1.0, (1e-3, 1e3)) * Matern(length_scale=1.0, length_scale_bounds=(1e-3, 1e3), nu=2.5)
        else:
            self.kernel = kernel
        self.gpr = GPR(kernel=self.kernel, alpha=noise_std**2, n_restarts_optimizer=10)

    def init_sample(self):
        X_sample = np.random.uniform(self.bounds[:, 0], self.bounds[:, 1], size=(self.n_init, self.bounds.shape[0]))
        Y_sample = self.f(X_sample)
        self.X_sample = X_sample
        self.Y_sample = Y_sample
        self.obj_sample = self.obj_func(X_sample, Y_sample) if self.obj_func != self.f else Y_sample


    def get_acquisition(self, acquisition: str):
        return pre_acquisition().get_standard_acquisition(acquisition)


    def next_sample(self):
        self.gpr.fit(self.X_sample, self.Y_sample)

        min_val = 1e100
        min_x = None

        for x0 in np.random.uniform(self.bounds[:, 0], self.bounds[:, 1], size=(self.n_iter, self.bounds.shape[0])):
            res = minimize(lambda X: -self.acquisition(X.reshape(-1, 1), self.X_sample, self.Y_sample, self.gpr).flatten(), x0=x0, bounds=self.bounds, method='L-BFGS-B')
            if res.fun < min_val:
                min_val = res.fun[0]
                min_x = res.x

        X_next = self.acquisition(self.gpr, self.bounds, self.X_sample, self.Y_sample)
        Y_next = self.f(X_next)
        self.X_next = X_next
        self.Y_next = Y_next
    
    def run(self):
        for i in range(self.n_iter):
            self.next_sample()
            self.X_sample = np.vstack((self.X_sample, self.X_next))
            self.Y_sample = np.vstack((self.Y_sample, self.Y_next))
        return self.X_sample, self.Y_sample

    

def f(X, noise=0):
    return -np.sin(3*X) - X**2 + 0.7*X + noise * np.random.randn(*X.shape)

def expected_improvement(X, X_sample, Y_sample, gpr, xi=0.01):
    '''
    Computes the EI at points X based on existing samples X_sample
    and Y_sample using a Gaussian process surrogate model.
    
    Args:
        X: Points at which EI shall be computed (m x d).
        X_sample: Sample locations (n x d).
        Y_sample: Sample values (n x 1).
        gpr: A GaussianProcessRegressor fitted to samples.
        xi: Exploitation-exploration trade-off parameter.
    
    Returns:
        Expected improvements at points X.
    '''
    mu, sigma = gpr.predict(X, return_std=True)
    mu_sample = gpr.predict(X_sample)
    sigma = sigma.reshape(-1, 1)
    # Needed for noise-based model,
    # otherwise use np.max(Y_sample).
    # See also section 2.4 in [1]
    mu_sample_opt = np.max(mu_sample)

    with np.errstate(divide='warn'):
        imp = mu - mu_sample_opt - xi
        Z = imp / sigma
        ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
        ei[sigma == 0.0] = 0.0

    return ei

def propose_location(acquisition, X_sample, Y_sample, gpr, bounds, n_restarts=25):
    '''
    Proposes the next sampling point by optimizing the acquisition function.
    
    Args:
        acquisition: Acquisition function.
        X_sample: Sample locations (n x d).
        Y_sample: Sample values (n x 1).
        gpr: A GaussianProcessRegressor fitted to samples.

    Returns:
        Location of the acquisition function maximum.
    '''
    dim = X_sample.shape[1]
    min_val = 1
    min_x = None
    
    def min_obj(X):
        # Minimization objective is the negative acquisition function
        return -acquisition(X.reshape(-1, dim), X_sample, Y_sample, gpr).flatten()
    
    # Find the best optimum by starting from n_restart different random points.
    for x0 in np.random.uniform(bounds[:, 0], bounds[:, 1], size=(n_restarts, dim)):
        res = minimize(min_obj, x0=x0, bounds=bounds, method='L-BFGS-B')        
        if res.fun < min_val:
            min_val = res.fun[0]
            min_x = res.x           
            
    return min_x.reshape(-1, 1)


def bay_opt():
    bounds = np.array([[-1.0, 2.0]])
    noise = 0.2

    X_init = np.array([[-0.9], [1.1]])
    Y_init = f(X_init)

    # Dense grid of points within bounds
    X = np.arange(bounds[:, 0], bounds[:, 1], 0.01).reshape(-1, 1)
    Y = f(X,0)

    # Gaussian process with Matérn kernel as surrogate model
    m52 = ConstantKernel(1.0) * Matern(length_scale=1.0, nu=2.5)
    gpr = GPR(kernel=m52, alpha=noise**2, n_restarts_optimizer=10)

    # Initialize samples
    X_sample = X_init
    Y_sample = Y_init

    # Number of iterations
    n_iter = 10

    plt.figure(figsize=(12, n_iter * 3))
    plt.subplots_adjust(hspace=0.4)

    for i in range(n_iter):
        # Update Gaussian process with existing samples
        gpr.fit(X_sample, Y_sample)

        # Obtain next sampling point from the acquisition function (expected_improvement)
        X_next = propose_location(expected_improvement, X_sample, Y_sample, gpr, bounds)
        
        # Obtain next noisy sample from the objective function
        Y_next = f(X_next, noise)
        
        # Plot samples, surrogate function, noise-free objective and next sampling location
        plt.subplot(n_iter, 2, 2 * i + 1)
        plot_approximation(gpr, X, Y, X_sample, Y_sample, X_next, show_legend=i==0)
        plt.title(f'Iteration {i+1}')

        plt.subplot(n_iter, 2, 2 * i + 2)
        plot_acquisition(X, expected_improvement(X, X_sample, Y_sample, gpr), X_next, show_legend=i==0)
            
        # Add sample to previous samples
        X_sample = np.vstack((X_sample, X_next))
        Y_sample = np.vstack((Y_sample, Y_next))
    
    plt.show()



def first_show():
    bounds = np.array([[-1.0, 2.0]])
    noise = 0.2

    X_init = np.array([[-0.9], [1.1]])
    Y_init = f(X_init)

    # Dense grid of points within bounds
    X = np.arange(bounds[:, 0], bounds[:, 1], 0.01).reshape(-1, 1)

    # Noise-free objective function values at X 
    Y = f(X,0)

    # Plot optimization objective with noise level 
    plt.plot(X, Y, 'y--', lw=2, label='Noise-free objective')
    plt.plot(X, f(X, noise), 'bx', lw=1, alpha=0.1, label='Noisy samples')
    plt.plot(X_init, Y_init, 'kx', mew=3, label='Initial samples')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    bay_opt()