import numpy as np
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor as GPR
from sklearn.gaussian_process.kernels import Matern, ConstantKernel

class CF_acquisition():

    def __init__(self, distance_function, desired_output, current_point, lam, g_func):
        self.desired_output = desired_output
        self.current_point = current_point
        self.distance_function = distance_function
        self.lam = lam
        self.g_func = g_func

    def get_CF_EI(self):
        return self.CF_expected_improvement
    
    # TODO: where does xi come in?
    def CF_expected_improvement(self, X, model, opt, xi=0.01):
        # Expected improvement acquisition function, xi does NOT mean no exploration
        dist = self.distance_function(X, self.current_point)

        if opt < dist:
            raise ValueError('opt must be greater than dist')
        
        quat = (opt - dist) / self.lam
        UB = self.desired_output + np.sqrt(quat)
        LB = self.desired_output - np.sqrt(quat)
        mu, std = model.clip_predict(X) # clip std to avoid numerical error

        f1 = (opt - dist + self.lam*(2*self.desired_output*mu - mu**2 - std**2 - self.desired_output**2))
        f2 = self.lam*std*(mu + UB - 2*self.desired_output)
        f3 = self.lam*std*(2*self.desired_output - mu - LB)

        arg_UB = (UB - mu) / std
        arg_LB = (LB - mu) / std

        ei = f1*(norm.cdf(arg_UB)-norm.cdf(arg_LB)) + f2*norm.cdf(arg_UB) + f3*norm.cdf(arg_LB)
        return ei