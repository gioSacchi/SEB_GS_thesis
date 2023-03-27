import numpy as np
import matplotlib.pyplot as plt

def plot_acquisition(X, Y, X_next, ax=None, show_legend=False):
    # plot the acquisition function
    # plot in 3D if X is 2D, X should be 2 block matrices
    if X.shape[0] == 2 and ax is not None:
        ax.plot_surface(X[0], X[1], Y, cmap='viridis', linewidth=0.2, label='Acquisition function')
        ax.scatter(X_next[0], X_next[1], Y.max(), marker='x', s=100, label='Next sampling location')
        if show_legend:
            ax.legend()
    else:
        # plt.figure()
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

def plot_obj_approx2D(gpr, X, Y, X_sample, Y_sample, X_next=None, obj_func=None, obj=None, obj_sample=None, show_legend=False):
    mu, std = gpr.predict(X, return_std=True)
    mu = mu.reshape(-1, 1)
    std = std.reshape(-1, 1)
    # plot the objective function 
    if obj is None:
        obj = obj_func(X, Y).reshape(X.shape) 
    plt.plot(X, obj, 'r--', lw=1, label='Obj function')

    # plot the surrogate function approximation the objective function
    obj_approx = obj_func(X, mu).reshape(X.shape)
    obj_approx_upper = obj_func(X, mu + 1.96 * std).reshape(X.shape)
    obj_approx_lower = obj_func(X, mu - 1.96 * std).reshape(X.shape)
    plt.plot(X, obj_approx, 'b-', lw=1, label='approx obj function')
    plt.fill_between(X.ravel(), 
                    obj_approx_upper.ravel(), 
                    obj_approx_lower.ravel(), 
                    alpha=0.1)

    # plot the sampled points
    if obj_sample is None:
        obj_sample = obj_func(X_sample, Y_sample).reshape(X_sample.shape)
    plt.plot(X_sample, obj_sample, 'kx', mew=3, label='Samples')
    if X_next is not None:
        plt.axvline(x=X_next, ls='--', c='k', lw=1)
    if show_legend:
        plt.legend()

def plot_surrogate_approx2D(gpr, X, Y, X_sample, Y_sample, X_next=None, show_legend=False):
    mu, std = gpr.predict(X, return_std=True)
    plt.fill_between(X.ravel(), 
                    mu.ravel() + 1.96 * std, 
                    mu.ravel() - 1.96 * std, 
                    alpha=0.1) 
    plt.plot(X, Y, 'y--', lw=1, label='True model')
    plt.plot(X, mu, 'b-', lw=1, label='Surrogate function')
    plt.plot(X_sample, Y_sample, 'kx', mew=3, label='Samples')
    if X_next:
        plt.axvline(x=X_next, ls='--', c='k', lw=1)
    if show_legend:
        plt.legend()

def plot_obj_approx3D(gpr, X, X1, X2, Y, X_sample, Y_sample,  obj_func, ax, X_next=None, obj=None, obj_sample=None, obj_next=None, show_legend=False):
    # where X, Y are 2D meshgrid of the domain 
    # X_sample, Y_sample are 2D arrays of the sample points
    # X_next is a 1D array of the next sample point

    mu, std = gpr.predict(X, return_std=True)
    mu = mu.reshape(X1.shape)
    std = std.reshape(X1.shape)

    # plot the objective function
    if obj is None:
        obj = obj_func(X, Y.reshape(-1,1)).reshape(X1.shape)
    ax.plot_surface(X1, X2, obj, alpha=0.5, label='Obj function') 

    # plot the surrogate function approximation the objective function
    obj_approx = obj_func(X, mu.reshape(-1,1)).reshape(X1.shape)
    obj_approx_upper = obj_func(X, (mu + 1.96 * std).reshape(-1,1)).reshape(X1.shape)
    obj_approx_lower = obj_func(X, (mu - 1.96 * std).reshape(-1,1)).reshape(X1.shape)
    ax.plot_surface(X1, X2, obj_approx, alpha=0.5, label='approx obj function')
    ax.plot_surface(X1, X2, obj_approx_upper, alpha=0.1)
    ax.plot_surface(X1, X2, obj_approx_lower, alpha=0.1)

    # plot the sampled points
    if obj_sample is None:
        obj_sample = obj_func(X_sample, Y_sample).reshape(X_sample.shape)
    ax.scatter(X_sample[:, 0], X_sample[:, 1], obj_sample, marker='x', s=100, label='Samples')
    if X_next is not None:
        ax.scatter(X_next[0], X_next[1], obj_next, marker='x', s=100, label='Next sampling location')
    if show_legend:
        ax.legend()


def plot_surrogate_approx3D(gpr, X, X1, X2, Y, X_sample, Y_sample, ax, X_next=None, show_legend=False):
    # where X, Y are 2D meshgrid of the domain 
    # X_sample, Y_sample are 2D arrays of the sample points
    # X_next is a 1D array of the next sample point

    mu, std = gpr.predict(X, return_std=True)
    # std = std.reshape(-1,1)
    mu = mu.reshape(X1.shape)
    std = std.reshape(X1.shape)
    # if obj_func is not None:
    #     ax = fig.add_subplot(121, projection='3d').set_title('Objective function (CF) to be optimized')

    #     # plot the objective function
    #     if obj is None:
    #         obj = obj_func(X, Y.reshape(-1,1)).reshape(X1.shape)
    #     ax.plot_surface(X1, X2, obj, alpha=0.5) 

    #     # plot the surrogate function approximation the objective function
    #     obj_approx = obj_func(X, mu.reshape(-1,1)).reshape(X1.shape)
    #     obj_approx_upper = obj_func(X, (mu + 1.96 * std).reshape(-1,1)).reshape(X1.shape)
    #     obj_approx_lower = obj_func(X, (mu - 1.96 * std).reshape(-1,1)).reshape(X1.shape)
    #     ax.plot_surface(X1, X2, obj_approx, alpha=0.5)
    #     ax.plot_surface(X1, X2, obj_approx_upper, alpha=0.1)
    #     ax.plot_surface(X1, X2, obj_approx_lower, alpha=0.1)

    #     # plot the sampled points
    #     if obj_sample is None:
    #         obj_sample = obj_func(X_sample, Y_sample).reshape(X_sample.shape)
    #     ax.plot3D(X_sample[:, 0], X_sample[:, 1], obj_sample.ravel(), 'kx', mew=3, label='Samples')
    #     if obj_next is not None:
    #         ax.plot([obj_next[0]], [obj_next[1]], 'kx', mew=3)
    #     if show_legend:
    #         ax.legend()

    #     ax = fig.add_subplot(122, projection='3d').set_title('Surrogate function approximating model')
    #     ax.plot_surface(X1, X2, Y, alpha=0.5) # plot the true model
    #     ax.plot_surface(X1, X2, mu, alpha=0.5) # plot the surrogate function
    #     ax.plot_surface(X1, X2, mu + 1.96 * std, alpha=0.1) # plot the confidence interval
    #     ax.plot_surface(X1, X2, mu - 1.96 * std, alpha=0.1)
    #     ax.plot3D(X_sample[:, 0], X_sample[:, 1], Y_sample.ravel(), 'kx', mew=3, label='Samples') # plot the sampled points
    #     if X_next is None:
    #         ax.plot(X_next[0], X_next[1], 'kx', mew=3)
    #     if show_legend:
    #         ax.legend()
    # else:
    # ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X1, X2, Y, alpha=0.5, label='True model')
    ax.plot_surface(X1, X2, mu, alpha=0.5, label='Surrogate model')
    ax.plot_surface(X1, X2, mu + 1.96 * std, alpha=0.1)
    ax.plot_surface(X1, X2, mu - 1.96 * std, alpha=0.1)
    ax.plot3D(X_sample[:, 0], X_sample[:, 1], Y_sample.ravel(), 'x', mew=3, label='Samples')
    if X_next is not None:
        ax.plot(X_next[0], X_next[1], 'x', mew=3)
    if show_legend:
        ax.legend()