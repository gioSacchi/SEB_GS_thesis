import numpy as np
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor as GPR
from sklearn.gaussian_process.kernels import Matern, ConstantKernel
from matplotlib import pyplot as plt

class CF_acquisition():

    def __init__(self, distance_function, desired_output, current_point, lam, g_inv = None, xi = 0.01, eps=1e-8, threshold=1e-7):
        self.desired_output = desired_output
        self.current_point = current_point
        self.distance_function = distance_function
        self.lam = lam
        self.g_inv = g_inv
        self.eps = eps
        self.threshold = threshold
        self.xi = xi
        self.neg_tol = 1e-10

    def get_CF_EI(self):
        return self.CF_expected_improvement
    
    # TODO: where does xi come in?
    # TODO: possible to use some general inverse?
    def CF_expected_improvement(self, X_prim, model, opt):
        # Expected improvement acquisition function, xi does NOT mean no exploration
        dist = self.distance_function(X_prim, self.current_point).reshape(-1, 1)

        quat = (opt - dist) / self.lam

        if self.g_inv is None: # standard Watcher et al. CF formulation
            # ingore warning about invalid value encountered in sqrt
            with np.errstate(invalid='ignore'):
                UB = self.desired_output + np.sqrt(quat)
                LB = self.desired_output - np.sqrt(quat)
        else:
            pass # TODO: implement this using g_inv
            
        mu, std = model.clip_predict(X_prim) # clip std to avoid numerical error
        mu = mu.reshape(-1, 1)
        std = std.reshape(-1, 1)

        f1 = opt - dist + self.lam*(2*self.desired_output*mu - mu**2 - std**2 - self.desired_output**2)
        f2 = self.lam*std*(mu + UB - 2*self.desired_output)
        f3 = self.lam*std*(2*self.desired_output - mu - LB)

        arg_UB = (UB - mu) / std
        arg_LB = (LB - mu) / std

        ei = f1*(norm.cdf(arg_UB)-norm.cdf(arg_LB)) + f2*norm.pdf(arg_UB) + f3*norm.pdf(arg_LB)

        # set ei to 0 if dist < opt as this is not a valid point to evaluate
        if np.any(opt < dist):
            problem_ind = np.where(opt < dist)[0]
            ei[problem_ind] = 0
            # raise ValueError('opt must be greater than dist') # TODO: should I raise error or set to -1?
            print('opt must be greater than dist')

        if np.any(ei < 0):
            # self.compare_CF_EI(X_prim, model, opt, ei)
            # check if the problem is due to numerical error, if so set to 0 and continue otherwise raise error
            if np.all(np.abs(ei[ei<0]) < self.neg_tol):
                ei[ei<0] = 0
                print('Numerical error in EI, setting to 0')
            else:
                raise ValueError('ei must be non-negative')
                            
        return ei
    
    def CF_expected_improvement_MC(self, X_prim, model, opt):
        mu, std = model.clip_predict(X_prim) # clip std to avoid numerical error
        mu = mu.reshape(-1, 1)
        std = std.reshape(-1, 1)
        # for each poimt in X sample 1000 points from normal distribution
        rv = np.random.normal(mu, std, size=(X_prim.shape[0], 1000))
        dist = self.distance_function(X_prim, self.current_point).reshape(-1, 1)
        c_hats = dist + self.lam*(rv-self.desired_output)**2
        c_diff = opt - c_hats
        c_diff[c_diff < 0] = 0
        ei = np.mean(c_diff, axis=1)
        return ei

    def compare_CF_EI(self, X_prim, model, opt, ei=None):
        ei_ex = self.CF_expected_improvement(X_prim, model, opt) if ei is None else ei
        ei_mc = self.CF_expected_improvement_MC(X_prim, model, opt)
        old_ei = self.old_ei_imp(X_prim, model, opt)
        # if more than one element, plot otherwise print
        # plot the expected improvements as functions of X in one figure and then also make a differences plot
        if X_prim.shape[0] > 1:
            plt.figure()
            plt.plot(X_prim, ei_ex, label='CF EI')
            plt.plot(X_prim, ei_mc, label='CF EI MC')
            plt.plot(X_prim, old_ei, label='old EI')
            plt.legend()

            plt.figure()
            plt.plot(X_prim, ei_ex - ei_mc)
            plt.title('Difference between CF EI and CF EI MC')
            plt.show()
        else:
            print('ei_ex = ', ei_ex)
            print('ei_mc = ', ei_mc) 
            print('old_ei = ', old_ei)

    def old_ei_imp(self, X_prim, model, opt):
        # Calculate improtant data
        dist = np.linalg.norm(self.current_point-X_prim, axis=1).reshape(-1, 1)

        UB = np.sqrt((opt-dist)/self.lam) + self.desired_output
        LB = -np.sqrt((opt-dist)/self.lam) + self.desired_output

        mu, std = model.clip_predict(X_prim) # clip std to avoid numerical error
        mu = mu.reshape(-1, 1)
        std = std.reshape(-1, 1)
        UB_arg = (UB - mu)/std
        LB_arg = (LB - mu)/std
        CDF_diff = norm.cdf(UB_arg) - norm.cdf(LB_arg)
        PDF_UB = norm.pdf(UB_arg)
        PDF_LB = norm.pdf(LB_arg)

        first_term = opt-dist+ self.lam*(2*self.desired_output*mu-self.desired_output**2-mu**2-std**2)
        second_term = self.lam*std*(mu+UB-2*self.desired_output)
        third_term = self.lam*std*(2*self.desired_output-mu-LB)
        EI_exact = first_term*CDF_diff + second_term*PDF_UB + third_term*PDF_LB
        # set to 0 if dist < opt or std = 0
        if np.any(opt < dist):
            problem_ind = np.where(opt < dist)[0]
            EI_exact[problem_ind] = 0
            print('opt must be greater than dist')
        if np.any(std == 0):
            problem_ind = np.where(std == 0)[0]
            EI_exact[problem_ind] = 0
            print('std = 0')

        return EI_exact

