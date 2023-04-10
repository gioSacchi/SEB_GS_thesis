import numpy as np
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor as GPR
from sklearn.gaussian_process.kernels import Matern, ConstantKernel
import scipy 

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

    # def expected_improvement(self, X, model, opt, xi=0.01):
    #     # Blog version but with opt given instead of X_samples and Y_samples

    #     mu, sigma = model.predict(X, return_std=True)
    #     # mu_sample = model.predict(X_sample)

    #     sigma = sigma.reshape(-1, 1)
        
    #     # Needed for noise-based model,
    #     # otherwise use np.max(Y_sample).
    #     # See also section 2.4 in [1]
    #     # mu_sample_opt = np.max(mu_sample)


    #     with np.errstate(divide='warn'):
    #         # imp = mu - mu_sample_opt - xi
    #         imp = opt - mu + xi # check this + xi or - xi
    #         Z = imp / sigma
    #         ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
    #         ei[sigma == 0.0] = 0.0

    #     return ei

    def expected_improvement(self, x, model, opt, xi=10):
        # GpyOpt version but with quantiles inside instead of import, they have a clip in predict

        def get_quantiles(acquisition_par, fmin, m, s):
            # if isinstance(s, np.ndarray):
            #     s[s<1e-10] = 1e-10
            # elif s< 1e-10:
            #     s = 1e-10
            s = np.clip(s, 1e-10, np.inf) # clip substitutes the original check above
            u = (fmin - m - acquisition_par)/s
            phi = np.exp(-0.5 * u**2) / np.sqrt(2*np.pi)
            Phi = 0.5 * scipy.special.erfc(-u / np.sqrt(2))
            return (phi, Phi, u)
        
        m, s = model.predict(x, return_std=True) # look at shapes of m and s to fit our code
        m = m.reshape(-1, 1)
        s = s.reshape(-1, 1)

        # fmin = model.get_fmin() # fmin = opt
        phi, Phi, u = get_quantiles(xi, opt, m, s)
        f_acqu = s * (u * Phi + phi)
        return f_acqu

    # def expected_improvement(self, X, model, opt, xi=0.01):
    #     # First early Giorgio version

    #     mu, std = model.predict(X, return_std=True)
    #     # check shapes
    #     with np.errstate(divide='warn'):
    #         small_sigma_idxs = np.where(std < 10**(-8))[0]


    #         if len(small_sigma_idxs) == 0:
    #             imp = mu - opt - xi
    #             Z = imp / std
    #             ei = imp * norm.cdf(Z) + std * norm.pdf(Z)
            
    #         elif 0 < len(small_sigma_idxs) < len(std):
    #             # sets all values to 0
    #             ei = np.zeros_like(std)

    #             # updates only non zero indices according to formula
    #             non_zero_idxs = np.where(std >= 10**(-8))[0]
    #             imp = mu[non_zero_idxs] - opt - xi
    #             Z = imp / std[non_zero_idxs]
    #             ei[non_zero_idxs] = imp * norm.cdf(Z) + std[non_zero_idxs] * norm.pdf(Z)
    #         else:
    #             # sets all values to 0 becasue all sigma are 0
    #             ei = np.zeros_like(std)     
    #             print('sigma is 0')
    #     return ei

    # # TODO: update to use GPmodel class
    # # First used version
    # def expected_improvement(self, X, model, opt, xi=0.01, eps=1e-8, threshold=1e-7):
    #     # Expected improvement acquisition function, xi does NOT mean no exploration
    #     mu, std = model.predict(X, return_std=True)
    #     mu = mu.reshape(-1, 1)
    #     std = std.reshape(-1, 1)
    #     # clip std to avoid numerical error
    #     # std = np.clip(std, threshold, np.inf) from GpyOpt
    #     std = np.maximum(std, threshold)
    #     with np.errstate(divide='warn'):
    #         imp = opt + xi - mu # check this + xi or - xi
    #         Z = imp/std
    #         ei = imp * norm.cdf(Z) + std * norm.pdf(Z)
    #         ei[std < eps] = 0.0

    #     return ei

    # def expected_improvement(self, X, model, opt, xi=0.01, eps=1e-8, threshold=1e-7):
    #     mu, sigma = model.predict(X, return_std=True)
    #     sigma = sigma.reshape(-1, 1)
    #     mu = mu.reshape(-1, 1) 

    #     # In case sigma  equals zero
    #     with np.errstate(divide='ignore'):
    #         Z = (opt-mu + xi) / sigma
    #         expected_improvement = (opt-mu + xi) * norm.cdf(Z) + sigma * norm.pdf(Z)
    #         expected_improvement[sigma == 0.0] = 0.0
        
    #     return expected_improvement
    
    def probability_improvement(self, X, model, opt, xi=0):
        # xi 0 means no exploration at all
        mu, std = model.predict(X, return_std=True)
        std = np.maximum(std, 1e-9)
        Z = (opt + xi- mu) / std
        return norm.cdf(Z)
    
    def upper_confidence_bound(self, X, model, opt, xi=0):
        # For maximization problem
        # xi 0 means no exploration at all
        mu, std = model.predict(X, return_std=True)
        std = np.maximum(std, 1e-9)
        return mu + xi * std
    
    def lower_confidence_bound(self, X, model, opt, xi=0):
        # For minimization problem
        # xi 0 means no exploration at all
        mu, std = model.predict(X, return_std=True)
        std = np.maximum(std, 1e-9)
        return mu - xi * std
    


def test():
    pa = pre_acquisition()
    EI = pa.get_standard_acquisition('EI')
    print(EI)

    X = np.array([[1, 2], [3,4]])
    X_samples = np.array([[1.5, 2], [2.9, 3.3]])
    kernel = ConstantKernel(1.0, (1e-3, 1e3)) * Matern(length_scale=1.0, length_scale_bounds=(1e-3, 1e3), nu=2.5)
    model = GPR(kernel=kernel, alpha=1e-5**2, n_restarts_optimizer=10)
    model.fit(X_samples, np.array([1, 2]))
    print(EI(X, X_samples, model, noisy_obs=True))


if __name__ == '__main__':
    test()
    pass