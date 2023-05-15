import numpy as np
import pandas as pd
import itertools
import os
from sklearn.gaussian_process.kernels import Matern
from BO import BayesianOptimization
from src.features.CF_acq import CF_acquisition
import matplotlib.pyplot as plt
from ALT_opt import ComparisonOptimizers

def f(x, x_prim, y_prim, computed_val, lam, std=None):
    if std is not None:
        # check std right dimension
        if std.shape[0] != x.shape[1]:
            raise Exception("std has wrong dimension")
        x = np.divide(x, std)
        x_prim = np.divide(x_prim, std)
    val = np.linalg.norm(x - x_prim, axis=1).reshape(-1,1) + (lam * (computed_val - y_prim)**2).reshape(-1,1)
    return val

def grid_search(function, grid_level, bounds):
    points_in_each_dim = grid_level*np.diff(bounds, axis=1).reshape(-1) + 1
    points_in_each_dim = points_in_each_dim.astype(int)
    grid = np.meshgrid(*[np.linspace(bounds[i,0], bounds[i,1], points_in_each_dim[i]) for i in range(bounds.shape[0])])
    grid = np.array(grid).reshape(bounds.shape[0], -1).T
    y = function(grid)
    opt_index = np.argmin(y)
    opt_point = grid[opt_index]
    return opt_point, y[opt_index]

def num_evals_and_error_2D(df_train, model, folder):
    # init data
    input_data = df_train.drop("Income", axis=1)
    output_data = df_train["Income"]
    std = input_data.std().to_numpy().reshape(-1)
    min1 = input_data.iloc[:, 0].min()
    max1 = input_data.iloc[:, 0].max()
    margin1 = (max1 - min1) * 0.1
    min2 = input_data.iloc[:, 1].min()
    max2 = input_data.iloc[:, 1].max()
    margin2 = (max2 - min2) * 0.1
    bounds = np.array([[min1 - margin1, max1 + margin1], [min2 - margin2, max2 + margin2]])
    np.random.seed(1234)
    grid = 20

    # generated points between min and max of data and create 2D grid of points
    x1 = np.linspace(min1, max1, grid)
    x2 = np.linspace(min2, max2, grid)
    x = np.array(list(itertools.product(*[x1, x2])))
    current_points = x.reshape(-1, 2)
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
    n_starting_points = 6
    n_max_iterations = 50
    n_stop = 3

    # bayesian_keys = ["BO_15", "BO_25"]
    bayesian_keys = ["BO_25"]
    # sep_keys = ["SBO_15", "SBO_25"]
    sep_keys = ["SBO_25"]
    # kernel_pram = {"BO_15": 1.5, "BO_25": 2.5, "SBO_15": 1.5, "SBO_25": 2.5}
    kernel_pram = {"BO_25": 2.5, "SBO_25": 2.5}
    random_keys = ["Random_10", "Random_20", "Random_40"]
    random_evals = {"Random_10": 10, "Random_20": 20, "Random_40": 40}

    keys = bayesian_keys + sep_keys + random_keys + ["Quasi"]
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

    dist_func = lambda point1, point2: np.linalg.norm(np.divide(point1-point2, std), axis=1)
    for point, y_prim in shuffled_combinations:
        print("current point: ", point, " desired output: ", y_prim)
        print("iteration: ", len(evals[keys[0]]))
        point = point.reshape(-1, 2)
        # define objective function
        lam = base_lambda/y_prim**2
        objective_func = lambda x_prim, func_val: f(point, x_prim, y_prim, func_val, lam, std=std)
        opt_func = lambda x_prim: f(point, x_prim, y_prim, model(x_prim), lam, std=std)
        acq_func = CF_acquisition(dist_func, y_prim, point, lam).get_CF_EI()
        
        for key in bayesian_keys:
            # initialize method with correct initial points
            kernel = Matern(length_scale=1.0, nu=kernel_pram[key])
            method[key] = BayesianOptimization(f = opt_func, acquisition="EI", dim = 2, bounds = bounds, 
                                      n_iter=n_max_iterations, n_init=n_starting_points, n_stop_iter=n_stop, 
                                      normalize_Y=True, kernel=kernel)
            method[key].run_BO()
            evals[key].append(method[key].number_of_evaluations_f)
        
        for key in sep_keys:
            kernel = Matern(length_scale=1.0, nu=kernel_pram[key])
            method[key] = BayesianOptimization(f = model, obj_func=objective_func, acquisition=acq_func, dim = 2, 
                                                bounds = bounds, n_iter=n_max_iterations, n_init=n_starting_points,  
                                                n_stop_iter=n_stop, normalize_Y=True, kernel=kernel)
            method[key].run_BO()
            evals[key].append(method[key].number_of_evaluations_f)

        for key in random_keys:
            method[key] = ComparisonOptimizers(opt_func, dim = 2, bounds=bounds, method='random',
                                                n_iter=random_evals[key]-n_starting_points, n_init=n_starting_points, 
                                                n_stop_iter=n_stop, noise_std=0)
            method[key].run()
            evals[key].append(method[key].n_evals)
        
        method["Quasi"] = ComparisonOptimizers(opt_func, dim = 2, bounds=bounds, method='l-bfgs-b',
                                                n_iter=n_max_iterations, n_init=n_starting_points, n_stop_iter=n_stop,
                                                noise_std=0)
        method["Quasi"].run()
        evals["Quasi"].append(method["Quasi"].n_evals)
        
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

