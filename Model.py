import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor as GPR
from sklearn.gaussian_process.kernels import Matern, ConstantKernel


class GPmodel(GPR):
    def __init__(self, X=None, Y=None, kernel=None, noise=0.1, normalize_y=True, n_restarts=10):
        self.X = X
        self.Y = Y
        self.noise = noise
        self.normalize_y = normalize_y
        self.n_restarts = n_restarts
        self.kernel = kernel
        self.gpr = self.create()
        if X is not None and Y is not None:
            self.fit(X, Y)

    def create(self):
        if self.kernel is None:
            self.kernel = ConstantKernel(1.0, (1e-3, 1e3)) * Matern(length_scale=1.0, length_scale_bounds=(1e-2, 1e2), nu=2.5)
        gpr = GPR(kernel=self.kernel, alpha=self.noise**2, n_restarts_optimizer=self.n_restarts)
        # normalize_y=self.normalize_y ??
        return gpr

    def fit(self, X, Y):
        self.X = X
        self.Y = Y
        self.gpr.fit(X, Y)

    def update(self, X_new, Y_new):
        self.X = np.vstack((self.X, X_new))
        self.Y = np.vstack((self.Y, Y_new))
        self.fit(self.X, self.Y)

    def pure_predict(self, X, return_std=True):
        return self.gpr.predict(X, return_std=return_std)

    def clip_predict(self, X, return_std=True):
        m, std = self.gpr.predict(X, return_std=return_std)
        std = np.clip(std, 1e-5, np.inf)
        return m, std