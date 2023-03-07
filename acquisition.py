import numpy as np
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor as GPR
from sklearn.gaussian_process.kernels import Matern, ConstantKernel

class pre_acquisition():
    # xi is exploitation-exploration trade-off parameter, higher value means more exploration

    def get_standard_acquisition(self, acquisition: str):
        if acquisition == 'EI':
            return self.expected_improvement
        elif acquisition == 'PI':
            return self.probability_improvement
        elif acquisition == 'UCB':
            return self.upper_confidence_bound
        elif acquisition == 'LCB':
            return self.lower_confidence_bound
        else:
            raise ValueError('Acquisition function not supported')

    # TODO: look into alternative implementation of acq funcs, particularly part about sigma
    # TODO: gradient of acq funcs ?

    def expected_improvement(self, X, gpr, opt, xi=0.01):
        # Expected improvement acquisition function, xi does NOT mean no exploration
        mu, std = gpr.predict(X, return_std=True)
        # clip std to avoid numerical error
        # std = np.clip(std, 1e-5, np.inf) from GpyOpt
        std = np.maximum(std, 1e-9)
        imp = opt + xi - mu
        Z = imp/std

        # # Blog version 
        # mu, sigma = gpr.predict(X, return_std=True)
        # mu_sample = gpr.predict(X_samples)
        # sigma = sigma.reshape(-1, 1)
        # print("sigma", sigma.shape)
        # # Needed for noise-based model,
        # # otherwise use np.max(Y_samples).
        # # See also section 2.4 in [1]
        # mu_sample_opt = np.max(mu_sample)
        # with np.errstate(divide='warn'):
        #     imp = mu - mu_sample_opt - xi
        #     Z = imp / sigma
        #     ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
        #     ei[sigma == 0.0] = 0.0

        # Giorgio's version
        # with np.errstate(divide='warn'):
        #     small_sigma_idxs = np.where(sigma < 10**(-8))[0]


        #     if len(small_sigma_idxs) == 0:
        #         imp = mu - mu_sample_opt - xi
        #         Z = imp / sigma
        #         ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
            
        #     elif 0 < len(small_sigma_idxs) < len(sigma):
        #         # sets all values to 0
        #         ei = np.zeros_like(sigma)

        #         # updates only non zero indices according to formula
        #         non_zero_idxs = np.where(sigma >= 10**(-8))[0]
        #         imp = mu[non_zero_idxs] - mu_sample_opt - xi
        #         Z = imp / sigma[non_zero_idxs]
        #         ei[non_zero_idxs] = imp * norm.cdf(Z) + sigma[non_zero_idxs] * norm.pdf(Z)
        #     else:
        #         # sets all values to 0 becasue all sigma are 0
        #         ei = np.zeros_like(sigma)     
        #         print('sigma is 0')

        return imp * norm.cdf(Z) + std * norm.pdf(Z)
    
    def probability_improvement(self, X, gpr, opt, xi=0):
        # xi 0 means no exploration at all
        mu, std = gpr.predict(X, return_std=True)
        std = np.maximum(std, 1e-9)
        Z = (opt + xi- mu) / std
        return norm.cdf(Z)
    
    def upper_confidence_bound(self, X, gpr, opt, xi=0):
        # For maximization problem
        # xi 0 means no exploration at all
        mu, std = gpr.predict(X, return_std=True)
        std = np.maximum(std, 1e-9)
        return mu + xi * std
    
    def lower_confidence_bound(self, X, gpr, opt, xi=0):
        # For minimization problem
        # xi 0 means no exploration at all
        mu, std = gpr.predict(X, return_std=True)
        std = np.maximum(std, 1e-9)
        return mu - xi * std
    


def test():
    pa = pre_acquisition()
    EI = pa.get_standard_acquisition('EI')
    print(EI)

    X = np.array([[1, 2], [3,4]])
    X_samples = np.array([[1.5, 2], [2.9, 3.3]])
    kernel = ConstantKernel(1.0, (1e-3, 1e3)) * Matern(length_scale=1.0, length_scale_bounds=(1e-3, 1e3), nu=2.5)
    gpr = GPR(kernel=kernel, alpha=1e-5**2, n_restarts_optimizer=10)
    gpr.fit(X_samples, np.array([1, 2]))
    print(EI(X, X_samples, gpr, noisy_obs=True))


if __name__ == '__main__':
    test()
    pass