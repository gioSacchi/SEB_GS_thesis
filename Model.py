import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor as GPR
from sklearn.gaussian_process.kernels import Matern, ConstantKernel


class GPmodel(GPR):
    def __init__(self, X=None, Y=None, kernel=None, noise=0.1, normalize_Y=True, n_restarts=20):
        self.X = X
        self.Y = Y
        self.noise = noise
        self.normalize_Y = normalize_Y
        self.Y_mean = 0
        self.Y_std = 1
        self.n_restarts = n_restarts
        self.kernel = kernel
        self.gpr = self.create()
        if X is not None and Y is not None:
            self.fit(X, Y)

    def create(self):
        if self.kernel is None:
            # self.kernel = ConstantKernel(1.0, constant_value_bounds=(1e-3, 1e3)) * Matern(length_scale=1.0, length_scale_bounds=(1e-2, 1e2), nu=2.5)
            # self.kernel = ConstantKernel(1.0) * Matern(length_scale=1.0, nu=2.5)
            self.kernel = Matern(length_scale=1.0, nu=2.5)
            # self.kernel = ConstantKernel(1.0) * Matern(length_scale=1.0, nu=2.5)
        gpr = GPR(kernel=self.kernel, alpha=self.noise**2, n_restarts_optimizer=self.n_restarts)
        return gpr

    def fit(self, X, Y):
        self.X = X
        self.Y = Y
        if self.normalize_Y:
            self.Y_mean = np.mean(self.Y)
            self.Y_std = np.std(self.Y)
            self.Y = (self.Y - self.Y_mean) / self.Y_std

        self.gpr.fit(self.X, self.Y)

    def update(self, X_new, Y_new):
        self.X = np.vstack((self.X, X_new))
        self.Y = np.vstack((self.Y, Y_new))
        self.fit(self.X, self.Y)

    def predict(self, X, return_std=True):
        mu, std = self.gpr.predict(X, return_std=return_std)
        if self.normalize_Y:
            # print("mu: ", mu)
            # print("std: ", std)
            # print("Y_mean: ", self.Y_mean)
            # print("Y_std: ", self.Y_std)
            return  mu*self.Y_std + self.Y_mean, std*self.Y_std
        else:
            return mu, std

    def clip_predict(self, X, lower_clip = 1e-10, return_std=True):
        m, std = self.gpr.predict(X, return_std=return_std)
        if self.normalize_Y:
            m = m*self.Y_std + self.Y_mean
            std = std*self.Y_std
        std = np.clip(std, lower_clip, np.inf)
        return m, std