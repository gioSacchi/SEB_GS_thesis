import numpy as np
import pandas as pd
import itertools
import os
from sklearn.gaussian_process.kernels import Matern
from BO import BayesianOptimization
from src.features.CF_acq import CF_acquisition
import matplotlib.pyplot as plt

def sample_selection_warm(X_samples, Y_samples, max_samples_tot, method, max_samples_per_it):
    # code to select samples from the new samples and the old samples
    # will always select max_samples_per_it samples from the new samples and add them to the old samples
    # if necessary, it will remove samples from the old samples to make room for the new samples

    # append X_samples to samples to use them as starting points for next iteration
    if X_samples is None and Y_samples is None:
        n_new_samples = method.X_samples.shape[0]
        n_to_swap = min(max_samples_per_it, n_new_samples)
        # chose indices of samples to add
        add_indices = np.random.choice(n_new_samples, n_to_swap, replace=False)
        # add new samples
        X_samples = method.X_samples[add_indices]
        Y_samples = method.Y_samples[add_indices]
    else:
        nr_init_points = X_samples.shape[0]
        new_Xs = method.X_samples[nr_init_points:]
        new_Ys = method.Y_samples[nr_init_points:]
        n_new_samples = new_Xs.shape[0]
        n_to_swap = min(max_samples_per_it, n_new_samples)
        # allow a maximum max_samples of samples
        if nr_init_points == max_samples_tot:
            # get indices of samples to swap
            swap_indices_old = np.random.choice(nr_init_points, n_to_swap, replace=False)
            swap_indices_new = np.random.choice(n_new_samples, n_to_swap, replace=False)
            # swap samples
            X_samples[swap_indices_old] = new_Xs[swap_indices_new]
            Y_samples[swap_indices_old] = new_Ys[swap_indices_new]
        elif nr_init_points + n_to_swap <= max_samples_tot:
            # get indices of samples to add
            add_indices = np.random.choice(n_new_samples, n_to_swap, replace=False)
            # add new samples
            X_samples = np.vstack((X_samples, new_Xs[add_indices]))
            Y_samples = np.vstack((Y_samples, new_Ys[add_indices]))
        elif nr_init_points + n_to_swap > max_samples_tot:
            # choose randomly which samples to add to get to 50
            n_to_take = nr_init_points + n_to_swap - max_samples_tot
            add_indices = np.random.choice(n_new_samples, n_to_swap, replace=False)
            take_indices = np.random.choice(nr_init_points, n_to_take, replace=False)
            # remove samples 
            X_samples = np.delete(X_samples, take_indices, axis=0)
            Y_samples = np.delete(Y_samples, take_indices, axis=0)
            # add new samples
            X_samples = np.vstack((X_samples, new_Xs[add_indices]))
            Y_samples = np.vstack((Y_samples, new_Ys[add_indices]))
        else:
            raise ValueError("Something went wrong with the samples")
        
        # if X_samples is None and Y_samples is None:
        #     X_samples = method["warm_starting"].X_samples
        #     Y_samples = method["warm_starting"].Y_samples
        # else:
        #     # add all new samples
        #     X_samples = np.vstack((X_samples, method["warm_starting"].X_samples))
        #     Y_samples = np.vstack((Y_samples, method["warm_starting"].Y_samples))
        
    return X_samples, Y_samples

def warm_starting_1D(df_train, model, folder):
    # init data
    input_data = df_train.drop("Income", axis=1)
    output_data = df_train["Income"]
    std = input_data.std().to_numpy().reshape(-1)
    min1 = input_data.iloc[:, 0].min()
    max1 = input_data.iloc[:, 0].max()
    margin1 = (max1 - min1) * 0.1
    bounds = np.array([[min1 - margin1, max1 + margin1]])
    np.random.seed(1234)
    grid = 20

    # generated points between min and max of data
    x = np.linspace(input_data.min(), input_data.max(), grid)
    current_points = x.reshape(-1, 1)
    np.random.shuffle(current_points)
    # generate grid of desired output values between min and max of output data
    desired = np.linspace(output_data.min(), output_data.max(), grid)
    np.random.shuffle(desired)
    # generate all combinations of current points and desired output values
    combinations = list(itertools.product(*[current_points, desired]))
    combinations = [list(x) for x in combinations]
    # shuffle combinations 
    shuffled_combinations = np.random.permutation(combinations)
    # get indices of shuffled combinations in original combinations
    # unshuffled_ind = []
    # for elem in combinations:
    #     # get index of row in shuffled combinations that is equal to elem
    #     index = np.where((shuffled_combinations == elem).all(axis=1))[0][0]
    #     unshuffled_ind.append(index)
        
    base_lambda = 10
    n_starting_points = 2
    n_max_iterations = 50
    n_stop = 3

    warm_keys = ["warm_starting_40", "warm_starting_20", "warm_starting_10"]
    max_samples = {"warm_starting_40": 40, "warm_starting_20": 20, "warm_starting_10": 10}
    # warm_keys = ["warm_starting_2"]
    # max_samples = {"warm_starting_2": 2}
    keys = warm_keys #+ ["reference_SBO"]
    evals = {}
    error_x = {}
    error_val = {}
    X_samples = {}
    Y_samples = {}
    method = {}
    for key in keys:
        evals[key] = []
        error_x[key] = []
        error_val[key] = []
        method[key] = None
        X_samples[key] = None
        Y_samples[key] = None
    
    # kernel = Matern(length_scale=1.0, length_scale_bounds=(1e-1, 1e5), nu=0.5)
    kernel = Matern(length_scale=1.0, nu=1.5)
    dist_func = lambda point1, point2: np.linalg.norm(np.divide(point1-point2, std), axis=1)
    offset = 0
    for point, y_prim in shuffled_combinations:
        print("current point: ", point, " desired output: ", y_prim)
        print("iteration: ", len(evals[keys[0]]))
        point = point.reshape(-1, 1)
        # define objective function
        lam = base_lambda/y_prim**2
        objective_func = lambda x_prim, func_val: f(point, x_prim, y_prim, func_val, lam, std=std)
        opt_func = lambda x_prim: f(point, x_prim, y_prim, model(x_prim), lam, std=std)
        acq_func = CF_acquisition(dist_func, y_prim, point, lam).get_CF_EI()
        
        for key in warm_keys:
            # initialize method with correct initial points
            method[key] = BayesianOptimization(f = model, obj_func=objective_func, acquisition=acq_func, dim = 1,
                                                bounds = bounds, n_iter=n_max_iterations+offset, n_init=n_starting_points,  
                                                n_stop_iter=n_stop, normalize_Y=True, kernel=kernel,
                                                init_points=None if X_samples[key] is None else (X_samples[key], Y_samples[key]), 
                                                plotting_freq=10)
            method[key].run_BO()
            evals[key].append(method[key].number_of_evaluations_f)
            # select and store new samples
            np.random.seed(len(evals[keys[0]])+1) # because of some weird sampling bug
            X_samples[key], Y_samples[key] = sample_selection_warm(X_samples[key], Y_samples[key], max_samples[key],
                                                                    method[key], max_samples_per_it=1)
        if offset == 0:
            offset = 2
        # method["reference_SBO"] = BayesianOptimization(f = model, obj_func=objective_func, acquisition=acq_func, dim = 1,
        #                                     bounds = bounds, n_iter=n_max_iterations, n_init=n_starting_points,
        #                                     n_stop_iter=n_stop, normalize_Y=True, kernel=kernel,
        #                                     init_points=None, plotting_freq=10)
        # method["reference_SBO"].run_BO()
        # evals["reference_SBO"].append(method["reference_SBO"].number_of_evaluations_f)
        
        # find optimal point, grid search
        opt_x, opt_val = grid_search(opt_func, 100, bounds)
        # calculate error, distance between optimal point and best found point
        for key in keys:
            error_x[key].append(dist_func(opt_x, method[key].opt_x)[0])
            error_val[key].append(np.abs(opt_val - method[key].opt_val)[0])

        # save it in folder, create folder if it does not exist
        if not os.path.exists(folder):
            os.makedirs(folder)
        # save data in csv file
        df = pd.DataFrame({key: evals[key] for key in keys})
        df.to_csv(folder+"/evals.csv", index=False)
        df = pd.DataFrame({key: error_x[key] for key in keys})
        df.to_csv(folder+"/error_x.csv", index=False)
        df = pd.DataFrame({key: error_val[key] for key in keys})
        df.to_csv(folder+"/error_val.csv", index=False)

        # save samples in csv file, but X_samples[key] is numpy array, so convert it to list
        for key in keys:
            df = pd.DataFrame({"X_samples": X_samples[key].tolist(), "Y_samples": Y_samples[key].tolist()})
            df.to_csv(folder+"/samples_"+key+".csv", index=False)

        
    # # unshuffle data lists and save them
    # for key in keys:
    #     evals[key] = [evals[key][i] for i in unshuffled_ind]
    #     error_x[key] = [error_x[key][i] for i in unshuffled_ind]
    #     error_val[key] = [error_val[key][i] for i in unshuffled_ind]
    # df = pd.DataFrame({key: evals[key] for key in keys})
    # df.to_csv(folder+"/evals.csv", index=False)
    # df = pd.DataFrame({key: error_x[key] for key in keys})
    # df.to_csv(folder+"/error_x.csv", index=False)
    # df = pd.DataFrame({key: error_val[key] for key in keys})
    # df.to_csv(folder+"/error_val.csv", index=False)

def plot_warm_starting(evals_df, error_x_df, error_val_df):
    keys = ["BO_15", "Sep_BO_15", "BO_25", "Sep_BO_25", "warm_starting_10", "warm_starting_20", "warm_starting_40", 
            "Quasi", "Random_10", "Random_20", "Random_40"]
    labels = [r"BO $\nu=1.5$", r"SBO $\nu=1.5$", r"BO $\nu=2.5$", r"SBO $\nu=2.5$", "WS_10", "WS_20", "WS_40", "Quasi",
                "Rand_10", "Rand_20", "Rand_40"]
    latex_names = ["BO $\\nu=1.5$", "SBO $\\nu=1.5$", "BO $\\nu=2.5$", "SBO $\\nu=2.5$", "WS$\_$10", "WS$\_$20", "WS$\_$40", "Quasi",
                "Rand$\_$10", "Rand$\_$20", "Rand$\_$40"]
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]
    evals = {key: evals_df[key].values for key in keys}
    error_x = {key: error_x_df[key].values for key in keys}
    error_val = {key: error_val_df[key].values for key in keys}

    # initalize empty dataframes
    # the columns are: mean, median, Q1, Q3, whisker_low, whisker_high
    # the rows are: keys
    evals_tot_df = pd.DataFrame(columns=["mean", "median", "Q1", "Q3", "whisker low", "whisker high"], index=latex_names)
    error_x_tot_df = pd.DataFrame(columns=["mean", "median", "Q1", "Q3", "whisker low", "whisker high"], index=latex_names)
    error_val_tot_df = pd.DataFrame(columns=["mean", "median", "Q1", "Q3", "whisker low", "whisker high"], index=latex_names)
    # generate data for table, mean, median, IQR and whiskers for each method. Whiskers are Q1/Q3 -/+ 1.5*IQR
    for key, ln in zip(keys, latex_names):
        # compute mean, median, IQR and whiskers and store them in dataframe
        evals_tot_df.loc[ln] = [np.mean(evals[key]), np.median(evals[key]), np.quantile(evals[key], 0.25), np.quantile(evals[key], 0.75),
                                    np.quantile(evals[key], 0.25) - 1.5*(np.quantile(evals[key], 0.75) - np.quantile(evals[key], 0.25)),
                                    np.quantile(evals[key], 0.75) + 1.5*(np.quantile(evals[key], 0.75) - np.quantile(evals[key], 0.25))]
        error_x_tot_df.loc[ln] = [np.mean(error_x[key]), np.median(error_x[key]), np.quantile(error_x[key], 0.25), np.quantile(error_x[key], 0.75),
                                    np.quantile(error_x[key], 0.25) - 1.5*(np.quantile(error_x[key], 0.75) - np.quantile(error_x[key], 0.25)),
                                    np.quantile(error_x[key], 0.75) + 1.5*(np.quantile(error_x[key], 0.75) - np.quantile(error_x[key], 0.25))]
        error_val_tot_df.loc[ln] = [np.mean(error_val[key]), np.median(error_val[key]), np.quantile(error_val[key], 0.25), np.quantile(error_val[key], 0.75),
                                    np.quantile(error_val[key], 0.25) - 1.5*(np.quantile(error_val[key], 0.75) - np.quantile(error_val[key], 0.25)),
                                    np.quantile(error_val[key], 0.75) + 1.5*(np.quantile(error_val[key], 0.75) - np.quantile(error_val[key], 0.25))]
    # save dataframes, in WS_tables folder in Data folder, create folder if it does not exist
    if not os.path.exists("Data/WS_tables"):
        os.makedirs("Data/WS_tables")
    evals_tot_df.to_csv("Data/WS_tables/evals.csv")
    error_x_tot_df.to_csv("Data/WS_tables/error_x.csv")
    error_val_tot_df.to_csv("Data/WS_tables/error_val.csv")

    # make plots 
    # accumulated number of evaluations
    fig1, ax1 = plt.subplots()
    for label, key, color in zip(labels, keys, colors):
        ax1.plot(np.cumsum(evals[key]), label=label, color=color)
    ax1.legend()
    ax1.set_xlabel("# of counterfactuals computed")
    ax1.set_ylabel("Accumulated Number of Function Evaluations")
    fig1.tight_layout()

    # accumulated error in x
    fig2, ax2 = plt.subplots()
    for label, key, color in zip(labels, keys, colors):
        ax2.plot(np.cumsum(error_x[key]), label=label, color=color)
    ax2.legend()
    ax2.set_xlabel("# of counterfactuals computed")
    ax2.set_ylabel("Accumulated Error - Distance to optimal point")
    fig2.tight_layout()

    # accumulated error in val
    fig3, ax3 = plt.subplots()
    for label, key, color in zip(labels, keys, colors):
        ax3.plot(np.cumsum(error_val[key]), label=label, color=color)
    ax3.legend()
    ax3.set_xlabel("# of counterfactuals computed")
    ax3.set_ylabel("Accumulated Error - Distance to optimal value")
    fig3.tight_layout()
    plt.show()

    fig4, ax4 = plt.subplots()
    ax4.boxplot([error_val[key] for key in keys], showmeans=True)
    ax4.set_xticklabels(labels, rotation=30)
    ax4.set_ylabel("Error - Distance to optimal value")
    ax4.yaxis.set_label_position("right")
    fig4.subplots_adjust(top=0.97, bottom=0.13, right=0.95)

    fig5, ax5 = plt.subplots()
    ax5.boxplot([error_x[key] for key in keys], showmeans=True)
    ax5.set_xticklabels(labels, rotation=30)
    ax5.set_ylabel("Error - Distance to optimal point")
    ax5.yaxis.set_label_position("right")
    fig5.subplots_adjust(top=0.97, bottom=0.13, right=0.95)

    fig6, ax6 = plt.subplots()
    ax6.boxplot([evals[key] for key in keys], showmeans=True)
    ax6.set_xticklabels(labels, rotation=30)
    ax6.set_ylabel("# of function evaluations")
    ax6.yaxis.set_label_position("right")
    fig6.subplots_adjust(top=0.97, bottom=0.13, right=0.95)

    legend_elements = [
        plt.Line2D([0], [0], color='k', lw=1, label='Whiskers'),
        plt.Line2D([0], [0], marker='o', markeredgecolor='k', markerfacecolor='w', label='Outliers', lw=0),
        plt.Rectangle((0, 0), 1, 1, ec='k', fc="w", alpha=0.5, label='IQR Box'),
        plt.Line2D([0], [0], color='orange', lw=1, label='Median'),
        plt.Line2D([0], [0], marker='^', color='g', label='Mean', lw=0),
    ]
    ax4.legend(handles=legend_elements, loc='best')
    ax5.legend(handles=legend_elements, loc='best')
    ax6.legend(handles=legend_elements, loc='best')

def grid_search(function, grid_level, bounds):
    points_in_each_dim = grid_level*np.diff(bounds, axis=1).reshape(-1) + 1
    points_in_each_dim = points_in_each_dim.astype(int)
    grid = np.meshgrid(*[np.linspace(bounds[i,0], bounds[i,1], points_in_each_dim[i]) for i in range(bounds.shape[0])])
    grid = np.array(grid).reshape(bounds.shape[0], -1).T
    y = function(grid)
    opt_index = np.argmin(y)
    opt_point = grid[opt_index]
    return opt_point, y[opt_index]

def f(x, x_prim, y_prim, computed_val, lam, std=None):
    if std is not None:
        # check std right dimension
        if std.shape[0] != x.shape[1]:
            raise Exception("std has wrong dimension")
        x = np.divide(x, std)
        x_prim = np.divide(x_prim, std)
    val = np.linalg.norm(x - x_prim, axis=1).reshape(-1,1) + (lam * (computed_val - y_prim)**2).reshape(-1,1)
    return val