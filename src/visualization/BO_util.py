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

def plot_obj_approx2D(model, X, Y, X_sample, Y_sample, obj_func, y_prim, X_next=None, obj=None, obj_sample=None, opt=None, show_legend=False):
    mu, std = model.predict(X, return_std=True)
    mu = mu.reshape(-1, 1)
    std = std.reshape(-1, 1)
    # plot the objective function 
    if obj is None:
        obj = obj_func(X, Y)
    plt.plot(X, obj, 'r--', lw=1, label='Objective function')

    # compute the upper and lower bound of the surrogate function
    model_upper = mu + 1.96 * std
    model_lower = mu - 1.96 * std
    # compute the upper and lower bound of the objective function
    obj_val_y_prim = obj_func(X, y_prim*np.ones_like(mu))
    obj_approx_upper = obj_func(X, mu + 1.96 * std)
    obj_approx_lower = obj_func(X, mu - 1.96 * std)

    # since obj_func is not monotone we need to adapt. We know that the min value is for y_prim.
    plot_lower = np.where(np.logical_and(model_lower < y_prim, y_prim < model_upper), obj_val_y_prim, np.minimum(obj_approx_lower, obj_approx_upper))
    plot_upper = np.maximum(obj_approx_lower, obj_approx_upper)
    
    # plot the surrogate function approximation the objective function
    obj_approx = obj_func(X, mu)
    plt.plot(X, obj_approx, 'b-', lw=1, label='Approximate objective function')

    # plot the upper and lower bound of the objective function
    plt.fill_between(X.ravel(), 
                    plot_lower.ravel(), 
                    plot_upper.ravel(), 
                    alpha=0.2)

    # plot the sampled points
    if obj_sample is None:
        obj_sample = obj_func(X_sample, Y_sample)
    plt.plot(X_sample, obj_sample, 'kx', mew=3, label='Samples')

    # plot the optimal point
    if opt is not None:
        # then opt should be a tuple (opt_x, opt_obj)
        plt.plot(opt[0], opt[1], 'gx', mew=3, label='Optimal point')
    if X_next is not None:
        plt.axvline(x=X_next, ls='--', c='k', lw=1)
    if show_legend:
        plt.legend()

def plot_surrogate_approx2D(model, X, Y, X_sample, Y_sample, X_next=None, opt=None, show_legend=False):
    mu, std = model.predict(X, return_std=True)
    plt.fill_between(X.ravel(), 
                    mu.ravel() + 1.96 * std, 
                    mu.ravel() - 1.96 * std, 
                    alpha=0.2) 
    plt.plot(X, Y, 'y--', lw=1, label='Real model')
    plt.plot(X, mu, 'b-', lw=1, label='Surrogate model')
    plt.plot(X_sample, Y_sample, 'kx', mew=3, label='Samples')
    if X_next:
        plt.axvline(x=X_next, ls='--', c='k', lw=1)
    if opt is not None:
        # then opt should be a tuple (opt_x, opt_y)
        plt.plot(opt[0], opt[1], 'gx', mew=3, label='Optimal point')
    if show_legend:
        plt.legend()

def plot_obj_approx3D(model, X, X1, X2, Y, X_sample, Y_sample,  obj_func, ax, y_prim, X_next=None, obj=None, obj_sample=None, obj_next=None, opt=None, show_legend=False):
    # where X, Y are 2D meshgrid of the domain 
    # X_sample, Y_sample are 2D arrays of the sample points
    # X_next is a 1D array of the next sample point

    mu, std = model.predict(X, return_std=True)
    mu = mu.reshape(X1.shape)
    std = std.reshape(X1.shape)

    # plot the objective function
    if obj is None:
        obj = obj_func(X, Y.reshape(-1,1)).reshape(X1.shape)
    ax.plot_surface(X1, X2, obj, alpha=0.5, label='Real objective function') 

    # compute the upper and lower bound of the surrogate function
    model_upper = (mu + 1.96 * std).reshape(-1,1)
    model_lower = (mu - 1.96 * std).reshape(-1,1)
    # compute the upper and lower bound of the objective function
    obj_val_y_prim = obj_func(X, y_prim*np.ones_like(mu))
    obj_approx_upper = obj_func(X, model_upper)
    obj_approx_lower = obj_func(X, model_lower)

    # since obj_func is not monotone we need to adapt. We know that the min value is for y_prim.
    plot_lower = np.where(np.logical_and(model_lower < y_prim, y_prim < model_upper), obj_val_y_prim, np.minimum(obj_approx_lower, obj_approx_upper)).reshape(X1.shape)
    plot_upper = np.maximum(obj_approx_lower, obj_approx_upper).reshape(X1.shape)
    
    # plot the surrogate function approximation the objective function
    obj_approx = obj_func(X, mu.reshape(-1,1)).reshape(X1.shape)
    ax.plot_surface(X1, X2, obj_approx, alpha=0.5, label='Approximated objective function')
    # plot the upper and lower bound of the objective function
    ax.plot_surface(X1, X2, plot_upper, alpha=0.2)
    ax.plot_surface(X1, X2, plot_lower, alpha=0.2)

    # plot the sampled points
    if obj_sample is None:
        obj_sample = obj_func(X_sample, Y_sample).reshape(X_sample.shape)
    ax.scatter(X_sample[:, 0], X_sample[:, 1], obj_sample, marker='x', s=100, label='Samples')
    # plot the optimal point
    if opt is not None:
        # then opt should be a tuple (opt_x, opt_obj)
        ax.scatter(opt[0][0], opt[0][1], opt[1], marker='x', s=100, label='Optimal point')
    if X_next is not None:
        ax.scatter(X_next[0], X_next[1], obj_next, marker='x', s=100, label='Next sampling location')
    if show_legend:
        ax.legend()


def plot_surrogate_approx3D(model, X, X1, X2, Y, X_sample, Y_sample, ax, X_next=None, opt=None, show_legend=False):
    # where X, Y are 2D meshgrid of the domain 
    # X_sample, Y_sample are 2D arrays of the sample points
    # X_next is a 1D array of the next sample point

    mu, std = model.predict(X, return_std=True)
    mu = mu.reshape(X1.shape)
    std = std.reshape(X1.shape)
    ax.plot_surface(X1, X2, Y, alpha=0.5, label='Real model')
    ax.plot_surface(X1, X2, mu, alpha=0.5, label='Surrogate model')
    ax.plot_surface(X1, X2, mu + 1.96 * std, alpha=0.2)
    ax.plot_surface(X1, X2, mu - 1.96 * std, alpha=0.2)
    ax.plot3D(X_sample[:, 0], X_sample[:, 1], Y_sample.ravel(), 'x', mew=3, label='Samples')
    if X_next is not None:
        ax.scatter(X_next[0], X_next[1], model.predict(X_next), marker='x', s=100, label='Next sampling location')
    if opt is not None:
        # then opt should be a tuple (opt_x, opt_y)
        ax.scatter(opt[0][0], opt[0][1], opt[1], marker='x', s=100, label='Optimal point')
    if show_legend:
        ax.legend()

def plot_sampled_points2D(X, Y, X_sample, Y_sample, X_next=None, show_legend=False):
    plt.plot(X, Y, 'y--', lw=1, label='Real model')
    plt.plot(X_sample, Y_sample, 'kx', mew=3, label='Samples')
    if X_next:
        plt.axvline(x=X_next, ls='--', c='k', lw=1)
    if show_legend:
        plt.legend()

def plot_sampled_points3D(X1, X2, Y, X_sample, Y_sample, ax, X_next=None, show_legend=False):
    ax.plot_surface(X1, X2, Y, alpha=0.5, label='Real model')
    ax.plot3D(X_sample[:, 0], X_sample[:, 1], Y_sample.ravel(), 'x', mew=3, label='Samples')
    if X_next is not None:
        ax.plot(X_next[0], X_next[1], 'x', mew=3)
    if show_legend:
        ax.legend()