import numpy as np
import matplotlib.pyplot as plt

def plot_approximation2D(gpr, X, Y, X_sample, Y_sample, X_next=None, show_legend=False):
    mu, std = gpr.predict(X, return_std=True)
    plt.fill_between(X.ravel(), 
                     mu.ravel() + 1.96 * std, 
                     mu.ravel() - 1.96 * std, 
                     alpha=0.1) 
    plt.plot(X, Y, 'y--', lw=1, label='Noise-free objective')
    plt.plot(X, mu, 'b-', lw=1, label='Surrogate function')
    plt.plot(X_sample, Y_sample, 'kx', mew=3, label='Noisy samples')
    if X_next:
        plt.axvline(x=X_next, ls='--', c='k', lw=1)
    if show_legend:
        plt.legend()

def plot_acquisition(X, Y, X_next, show_legend=False):
    plt.plot(X, Y, 'r-', lw=1, label='Acquisition function')
    plt.axvline(x=X_next, ls='--', c='k', lw=1, label='Next sampling location')
    if show_legend:
        plt.legend()    
        
def plot_convergence(X_sample, Y_sample, n_init=2):
    plt.figure(figsize=(12, 3))

    x = X_sample[n_init:].ravel()
    y = Y_sample[n_init:].ravel()
    r = range(1, len(x)+1)
    
    x_neighbor_dist = [np.abs(a-b) for a, b in zip(x, x[1:])]
    y_max_watermark = np.maximum.accumulate(y)
    
    plt.subplot(1, 2, 1)
    plt.plot(r[1:], x_neighbor_dist, 'bo-')
    plt.xlabel('Iteration')
    plt.ylabel('Distance')
    plt.title('Distance between consecutive x\'s')

    plt.subplot(1, 2, 2)
    plt.plot(r, y_max_watermark, 'ro-')
    plt.xlabel('Iteration')
    plt.ylabel('Best Y')
    plt.title('Value of best selected sample')

def plot_approximation3D(gpr, X, X1, X2, Y, X_sample, Y_sample, ax, X_next=None, show_legend=False):
    # where X, Y are 2D meshgrid of the domain 
    # X_sample, Y_sample are 2D arrays of the sample points
    # X_next is a 1D array of the next sample point

    mu, std = gpr.predict(X, return_std=True)
    # std = std.reshape(-1,1)
    mu = mu.reshape(X1.shape)
    std = std.reshape(X1.shape)
    ax.plot_surface(X1, X2, Y, alpha=0.5)
    ax.plot_surface(X1, X2, mu, alpha=0.5)
    ax.plot_surface(X1, X2, mu + 1.96 * std, alpha=0.1)
    ax.plot_surface(X1, X2, mu - 1.96 * std, alpha=0.1)
    ax.plot3D(X_sample[:, 0], X_sample[:, 1], Y_sample.ravel(), 'kx', mew=3, label='Noisy samples')
    if X_next is None:
        ax.plot(X_next[0], X_next[1], 'kx', mew=3)
    if show_legend:
        ax.legend()