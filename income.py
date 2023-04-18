import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import re
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR

from BO import BayesianOptimization
from src.features.CF_acq import CF_acquisition
from ALT_opt import ComparisonOptimizers

def visualize(df_train):
    """Visualize data"""
    # plot the data points, in 2D if one input feature, in 3D if two input features
    output_data = df_train["Income"]
    input_data = df_train.drop("Income", axis=1)
    if input_data.shape[1] == 1:
        plt.scatter(input_data, output_data)
        plt.xlabel(df_train.columns[0])
        plt.ylabel(output_data.name)
    elif input_data.shape[1] == 2:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        ax.scatter(input_data.iloc[:, 0], input_data.iloc[:, 1], output_data)
        ax.set_xlabel(df_train.columns[0])
        ax.set_ylabel(df_train.columns[1])
        ax.set_zlabel(output_data.name)
    else:
        print("Cannot visualize data with more than 2 input features")

def visualize_regressor(df_train, model):
    # plot the data points, in 2D if one input feature, in 3D if two input features
    # generate a grid of points to plot the model's predictions
    output_data = df_train["Income"]
    input_data = df_train.drop("Income", axis=1)
    fig = plt.figure()
    if input_data.shape[1] == 1:
        plt.scatter(input_data, output_data)
        plt.xlabel(df_train.columns[0])
        plt.ylabel(output_data.name)
        x_min, x_max = plt.xlim()
        x = np.linspace(x_min, x_max, 100)
        y = model.predict(x.reshape(-1, 1))
        plt.plot(x, y, color="red")
        plt.axhline(y=60, color='black', linestyle='dotted')
    elif input_data.shape[1] == 2:
        ax = fig.add_subplot(111, projection="3d")
        ax.scatter(input_data.iloc[:, 0], input_data.iloc[:, 1], output_data)
        ax.set_xlabel(df_train.columns[0])
        ax.set_ylabel(df_train.columns[1])
        ax.set_zlabel(output_data.name)
        x_min, x_max = ax.get_xlim()
        y_min, y_max = ax.get_ylim()
        x = np.linspace(x_min, x_max, 100)
        y = np.linspace(y_min, y_max, 100)
        x, y = np.meshgrid(x, y)
        z = model.predict(np.c_[x.ravel(), y.ravel()])
        z = z.reshape(x.shape)
        ax.plot_surface(x, y, z, antialiased=False, cmap="plasma")
    else:
        print("Cannot visualize data with more than 2 input features")

def linear_model_train(df_train):
    # X_train, X_test, y_train, y_test = train_test_split(df_train.drop("Income",axis=1), df_train["Income"], test_size=0.2, random_state=2)
    model = LinearRegression()
    model.fit(df_train.drop("Income",axis=1).to_numpy(), df_train["Income"].to_numpy())
    # model.fit(X_train.to_numpy(), y_train.to_numpy())
    # y_pred = model.predict(X_test.to_numpy())
    
    # # evaluate the model
    # print('R2: %.3f' % r2_score(y_test, y_pred))
    # print("pred", y_pred.round(3))
    # print("true", y_test.values.round(3))

    return model

def dct_model_train(df_train):
    # X_train, X_test, y_train, y_test = train_test_split(df_train.drop("Income",axis=1), df_train["Income"], test_size=0.2, random_state=2)
    model = DecisionTreeRegressor()
    model.fit(df_train.drop("Income",axis=1).to_numpy(), df_train["Income"].to_numpy())
    # model.fit(X_train.to_numpy(), y_train.to_numpy())
    # y_pred = model.predict(X_test.to_numpy())

    # # evaluate the model
    # print('R2: %.3f' % r2_score(y_test, y_pred))
    # print("pred", y_pred.round(3))
    # print("true", y_test.values.round(3))

    return model

def svr_model_train(df_train):
    # X_train, X_test, y_train, y_test = train_test_split(df_train.drop("Income",axis=1), df_train["Income"], test_size=0.2, random_state=2)
    model = SVR(degree=3, C=100, epsilon=0.1)
    X_train = df_train.drop("Income",axis=1).to_numpy()
    y_train = df_train["Income"].to_numpy()

    # sixdubble the training data by copying it, good for 3D data
    X_train = np.concatenate((X_train, X_train, X_train, X_train, X_train, X_train), axis=0)
    y_train = np.concatenate((y_train, y_train, y_train, y_train, y_train, y_train), axis=0)
    
    model.fit(X_train, y_train)
    # y_pred = model.predict(X_test.to_numpy())

    # # evaluate the model
    # print('R2: %.3f' % r2_score(y_test, y_pred))
    # print("pred", y_pred.round(3))
    # print("true", y_test.values.round(3))

    return model

def f(x, x_prim, y_prim, computed_val, lam, std=None):
    if std is not None:
        # check std right dimension
        if std.shape[0] != x.shape[1]:
            raise Exception("std has wrong dimension")
        x = np.divide(x, std)
        x_prim = np.divide(x_prim, std)
    val = np.linalg.norm(x - x_prim, axis=1).reshape(-1,1) + (lam * (computed_val - y_prim)**2).reshape(-1,1)
    return val

def visualize_f(f, model, x, des_out, lam, df_train):
    # generates grid and plots f, in 2D if one input feature, in 3D if two input features
    # generate a grid of points to plot the model's predictions
    fig = plt.figure()
    # calc std for each feature in training data. Remove feature name
    std = df_train.drop("Income", axis=1).std().to_numpy().reshape(-1)
    output_data = df_train["Income"]
    input_data = df_train.drop("Income", axis=1)
    if x.shape[1] == 1:
        x_min, x_max = df_train.iloc[:, 0].min(), df_train.iloc[:, 0].max()
        x_prim = np.linspace(x_min, x_max, 100).reshape(-1, 1)
        y = f(x, x_prim, des_out, model.predict(x_prim).reshape(-1,1), lam, std=std)
        plt.plot(x_prim, y, color="magenta")
        plt.xlabel(input_data.columns[0])
        plt.ylabel("counterfactual value")
    elif x.shape[1] == 2:
        # get min and max of for input feature in training data
        x_min, x_max = df_train.iloc[:, 0].min(), df_train.iloc[:, 0].max()
        y_min, y_max = df_train.iloc[:, 1].min(), df_train.iloc[:, 1].max()
        x_prim = np.linspace(x_min, x_max, 100)
        y_prim = np.linspace(y_min, y_max, 100)
        x_prim, y_prim = np.meshgrid(x_prim, y_prim)
        xy_prim = np.c_[x_prim.ravel(), y_prim.ravel()]
        z = f(x, xy_prim, des_out, model.predict(xy_prim).reshape(-1,1), lam, std=std)
        ax = fig.add_subplot(111, projection="3d")
        ax.set_xlabel(df_train.columns[0])
        ax.set_ylabel(df_train.columns[1])
        ax.set_zlabel("counterfactual value")
        ax.plot_surface(x_prim, y_prim, z.reshape(x_prim.shape), antialiased=False, cmap="plasma")
    else:
        print("Cannot visualize data with more than 2 input features")

def bo_test_own(func, obj_func, acq, df_train, y_prim):
    # handle cases of 1 and 2 input features and
    # define bounds from data and add a margin which is 10% of the range of the data in each dimension
    # output_data = df_train["Income"]
    input_data = df_train.drop("Income", axis=1)
    if input_data.shape[1] == 2:
        min1 = input_data.iloc[:, 0].min()
        max1 = input_data.iloc[:, 0].max()
        min2 = input_data.iloc[:, 1].min()
        max2 = input_data.iloc[:, 1].max()
        margin1 = (max1 - min1) * 0.1
        margin2 = (max2 - min2) * 0.1
        bounds = np.array([[min1 - margin1, max1 + margin1], [min2 - margin2, max2 + margin2]])
    elif input_data.shape[1] == 1:
        min1 = input_data.iloc[:, 0].min()
        max1 = input_data.iloc[:, 0].max()
        margin1 = (max1 - min1) * 0.1
        bounds = np.array([[min1 - margin1, max1 + margin1]])
    else:
        print("Cannot optimize data with more than 2 input features")
        return
    
    bo = BayesianOptimization(f = func, obj_func=obj_func, acquisition=acq, dim = input_data.shape[1], 
                              bounds = bounds, n_iter=10*input_data.shape[1]**2, n_init=2*input_data.shape[1], noise_std=0, normalize_Y=True)
    bo.run_BO()
    print(bo.number_of_evaluations_f)
    bo.make_plots(y_prim=y_prim)

def BO_on_regressor(model, df_train):
    # handle cases of 1 and 2 input features and
    # define bounds from data and add a margin which is 10% of the range of the data in each dimension
    # output_data = df_train["Income"]
    input_data = df_train.drop("Income", axis=1)
    if input_data.shape[1] == 2:
        min1 = input_data.iloc[:, 0].min()
        max1 = input_data.iloc[:, 0].max()
        min2 = input_data.iloc[:, 1].min()
        max2 = input_data.iloc[:, 1].max()
        margin1 = (max1 - min1) * 0.1
        margin2 = (max2 - min2) * 0.1
        bounds = np.array([[min1 - margin1, max1 + margin1], [min2 - margin2, max2 + margin2]])
    elif input_data.shape[1] == 1:
        min1 = input_data.iloc[:, 0].min()
        max1 = input_data.iloc[:, 0].max()
        margin1 = (max1 - min1) * 0.1
        bounds = np.array([[min1 - margin1, max1 + margin1]])
    else:
        print("Cannot optimize data with more than 2 input features")
        return
    bo = BayesianOptimization(f = model, dim = input_data.shape[1],bounds = bounds, n_iter=10*input_data.shape[1]**2, n_init=2*input_data.shape[1], noise_std=0, normalize_Y=True)
    bo.run_BO()
    print(bo.number_of_evaluations_f)
    bo.make_plots()

def comparison(model, df_train, opt_func = None, objective_function=None, acq_func=None, y_prim=None):
    # handle cases of 1 and 2 input features and
    # define bounds from data and add a margin which is 10% of the range of the data in each dimension
    # output_data = df_train["Income"]
    input_data = df_train.drop("Income", axis=1)
    if input_data.shape[1] == 2:
        min1 = input_data.iloc[:, 0].min()
        max1 = input_data.iloc[:, 0].max()
        min2 = input_data.iloc[:, 1].min()
        max2 = input_data.iloc[:, 1].max()
        margin1 = (max1 - min1) * 0.1
        margin2 = (max2 - min2) * 0.1
        bounds = np.array([[min1 - margin1, max1 + margin1], [min2 - margin2, max2 + margin2]])
    elif input_data.shape[1] == 1:
        min1 = input_data.iloc[:, 0].min()
        max1 = input_data.iloc[:, 0].max()
        margin1 = (max1 - min1) * 0.1
        bounds = np.array([[min1 - margin1, max1 + margin1]])
    else:
        print("Cannot optimize data with more than 2 input features")
        return
    if objective_function is None:
        bo = BayesianOptimization(f = model, dim = input_data.shape[1], bounds = bounds, n_iter=10*input_data.shape[1]**2, n_init=2*input_data.shape[1], noise_std=0, normalize_Y=True)
    else:
        bo = BayesianOptimization(f = model, obj_func=objective_function, acquisition=acq_func, dim = input_data.shape[1], bounds = bounds, n_iter=10*input_data.shape[1]**2, n_init=2*input_data.shape[1], noise_std=0, normalize_Y=True)
    bo.run_BO()
    print(bo.number_of_evaluations_f)
    if y_prim is None:
        bo.make_plots()
    else:
        bo.make_plots(y_prim=y_prim)
    if opt_func is None:
        alt_random = ComparisonOptimizers(model, input_data.shape[1], bounds=bounds, method = 'random', n_iter=10*input_data.shape[1]**2, n_init=2*input_data.shape[1], noise_std=0)
    else:
        alt_random = ComparisonOptimizers(opt_func, input_data.shape[1], bounds=bounds, method = 'random',n_iter=10*input_data.shape[1]**2, n_init=2*input_data.shape[1], noise_std=0)    
    alt_random.run()
    alt_random.make_plots()
    if opt_func is None:
        alt_quasi = ComparisonOptimizers(model, input_data.shape[1], bounds=bounds,  method='l-bfgs-b', n_iter=10*input_data.shape[1]**2, n_init=2*input_data.shape[1], noise_std=0)
    else:
        alt_quasi = ComparisonOptimizers(opt_func, input_data.shape[1], bounds=bounds,  method='l-bfgs-b', n_iter=10*input_data.shape[1]**2, n_init=2*input_data.shape[1], noise_std=0)
    alt_quasi.run()
    alt_quasi.make_plots()

def grid_search(function, grid_level, bounds):
    points_in_each_dim = grid_level*np.diff(bounds, axis=1).reshape(-1) + 1
    points_in_each_dim = points_in_each_dim.astype(int)
    grid = np.meshgrid(*[np.linspace(bounds[i,0], bounds[i,1], points_in_each_dim[i]) for i in range(bounds.shape[0])])
    grid = np.array(grid).reshape(bounds.shape[0], -1).T
    y = function(grid)
    opt_index = np.argmin(y)
    opt_point = grid[opt_index]
    return opt_point, y[opt_index]

def experiment_1_1D(df_train, model, folder):
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
    base_lambda = 10

    n_starting_points = 2
    n_max_iterations = 30
    n_stop = 3

    evals = {"BO": [], "Random_10": [], "Random_20": [], "Random_40": [], "Quasi": [], "Sep_BO": []}
    error_x = {"BO": [], "Random_10": [], "Random_20": [], "Random_40": [], "Quasi": [], "Sep_BO": []}
    error_val = {"BO": [], "Random_10": [], "Random_20": [], "Random_40": [], "Quasi": [], "Sep_BO": []}

    dist_func = lambda point1, point2: np.linalg.norm(np.divide(point1-point2, std), axis=1)
    for point in current_points:
        for y_prim in desired:
            print("Point: ", point, " Desired: ", y_prim)
            print("iteration: ", len(evals["BO"]))
            point = point.reshape(-1, 1)
            # define objective function
            lam = base_lambda/y_prim**2
            objective_func = lambda x_prim, func_val: f(point, x_prim, y_prim, func_val, lam, std=std)
            opt_func = lambda x_prim: f(point, x_prim, y_prim, model(x_prim), lam, std=std)
            acq_func = CF_acquisition(dist_func, y_prim, point, lam).get_CF_EI()
            # run experiment
            # Normal BO
            bo_normal = BayesianOptimization(f = opt_func, acquisition="EI", dim = 1, bounds = bounds, 
                                      n_iter=n_max_iterations, n_init=n_starting_points, n_stop_iter=n_stop, 
                                      noise_std=0, normalize_Y=True)
            bo_normal.run_BO()
            # bo_normal.make_plots()
            # plt.show()
            evals["BO"].append(bo_normal.number_of_evaluations_f)

            # Separated BO
            bo_separated = BayesianOptimization(f = model, obj_func=objective_func, acquisition=acq_func, dim = 1, 
                                                bounds = bounds, n_iter=n_max_iterations, n_init=n_starting_points,  
                                                n_stop_iter=n_stop, noise_std=0, normalize_Y=True)
            bo_separated.run_BO()
            # bo_separated.make_plots(y_prim=y_prim)
            # plt.show()
            evals["Sep_BO"].append(bo_separated.number_of_evaluations_f)

            # Random with 10 total points
            alt_random_10 = ComparisonOptimizers(opt_func, 1, bounds=bounds, method='random',
                                                n_iter=10-n_starting_points, n_init=n_starting_points, n_stop_iter=n_stop,
                                                noise_std=0)
            alt_random_10.run()
            # alt_random.make_plots()
            # plt.show()
            evals["Random_10"].append(alt_random_10.n_evals)

            # Random with 20 total points
            alt_random_20 = ComparisonOptimizers(opt_func, 1, bounds=bounds, method='random',
                                                n_iter=20-n_starting_points, n_init=n_starting_points, n_stop_iter=n_stop,
                                                noise_std=0)
            alt_random_20.run()
            evals["Random_20"].append(alt_random_20.n_evals)

            # Random with 40 total points
            alt_random_40 = ComparisonOptimizers(opt_func, 1, bounds=bounds, method='random',
                                                n_iter=40-n_starting_points, n_init=n_starting_points, n_stop_iter=n_stop,
                                                noise_std=0)
            alt_random_40.run()
            evals["Random_40"].append(alt_random_40.n_evals)

            # Quasi
            alt_quasi = ComparisonOptimizers(opt_func, 1, bounds=bounds, method='l-bfgs-b',
                                                n_iter=n_max_iterations, n_init=n_starting_points, n_stop_iter=n_stop,
                                                noise_std=0)
            alt_quasi.run()
            # alt_quasi.make_plots()
            # plt.show()
            evals["Quasi"].append(alt_quasi.n_evals)

            # find optimal point, grid search
            opt_x, opt_val = grid_search(opt_func, 100, bounds)
            # calculate error, distance between optimal point and best found point
            error_x["BO"].append(dist_func(opt_x, bo_normal.opt_x))
            error_x["Sep_BO"].append(dist_func(opt_x, bo_separated.opt_x))
            error_x["Random_10"].append(dist_func(opt_x, alt_random_10.opt_x))
            error_x["Random_20"].append(dist_func(opt_x, alt_random_20.opt_x))
            error_x["Random_40"].append(dist_func(opt_x, alt_random_40.opt_x))
            error_x["Quasi"].append(dist_func(opt_x, alt_quasi.opt_x))
            error_val["BO"].append(np.abs(opt_val - bo_normal.opt_val))
            error_val["Sep_BO"].append(np.abs(opt_val - bo_separated.opt_val))
            error_val["Random_10"].append(np.abs(opt_val - alt_random_10.opt_val))
            error_val["Random_20"].append(np.abs(opt_val - alt_random_20.opt_val))
            error_val["Random_40"].append(np.abs(opt_val - alt_random_40.opt_val))
            error_val["Quasi"].append(np.abs(opt_val - alt_quasi.opt_val))

             # save data in csv file
            df = pd.DataFrame({"BO": evals["BO"], "Sep_BO": evals["Sep_BO"], "Random_10": evals["Random_10"],
                                "Random_20": evals["Random_20"], "Random_40": evals["Random_40"], "Quasi": evals["Quasi"]})
            # save it in folder, create folder if it does not exist
            if not os.path.exists(folder):
                os.makedirs(folder)
            df.to_csv(folder+"/evals.csv", index=False)
            df = pd.DataFrame({"BO": error_x["BO"], "Sep_BO": error_x["Sep_BO"], "Random_10": error_x["Random_10"],
                                "Random_20": error_x["Random_20"], "Random_40": error_x["Random_40"], "Quasi": error_x["Quasi"]})
            df.to_csv(folder+"/error_x.csv", index=False)
            df = pd.DataFrame({"BO": error_val["BO"], "Sep_BO": error_val["Sep_BO"], "Random_10": error_val["Random_10"],
                                "Random_20": error_val["Random_20"], "Random_40": error_val["Random_40"], "Quasi": error_val["Quasi"]})
            df.to_csv(folder+"/error_val.csv", index=False)

    
    # plot results
    # make two boxplots, one for evals, one for error. Two separe figures
    fig1, ax1 = plt.subplots()
    ax1.boxplot([evals["BO"], evals["Random_10"], evals["Random_20"], evals["Random_40"], evals["Quasi"], evals["Sep_BO"]])
    ax1.set_xticklabels(["BO", "Random_10", "Random_20", "Random_40", "Quasi", "SBO"])
    ax1.set_ylabel("Number of evaluations")

    fig2, ax2 = plt.subplots()
    ax2.boxplot([error_x["BO"], error_x["Random_10"], error_x["Random_20"], error_x["Random_40"], error_x["Quasi"], error_x["Sep_BO"]])
    ax2.set_xticklabels(["BO", "Random_10", "Random_20", "Random_40", "Quasi", "SBO"])
    ax2.set_ylabel("Error - Distance to optimal point")

    fig4, ax4 = plt.subplots()
    ax4.boxplot([error_val["BO"], error_val["Random_10"], error_val["Random_20"], error_val["Random_40"], error_val["Quasi"], error_val["Sep_BO"]])
    ax4.set_xticklabels(["BO", "Random_10", "Random_20", "Random_40", "Quasi", "SBO"])
    ax4.set_ylabel("Error - Distance to optimal value")

    # make a plot for accumulated error
    fig3, ax3 = plt.subplots()
    ax3.plot(np.cumsum(error_x["BO"]), label="BO")
    ax3.plot(np.cumsum(error_x["Random_10"]), label="Random_10")
    ax3.plot(np.cumsum(error_x["Random_20"]), label="Random_20")
    ax3.plot(np.cumsum(error_x["Random_40"]), label="Random_40")
    ax3.plot(np.cumsum(error_x["Quasi"]), label="Quasi")
    ax3.plot(np.cumsum(error_x["Sep_BO"]), label="SBO")
    ax3.legend()
    ax3.set_xlabel("# of counterfactuals computed")
    ax3.set_ylabel("Accumulated Error - Distance to optimal point")

    fig5, ax5 = plt.subplots()
    ax5.plot(np.cumsum(error_val["BO"]), label="BO")
    ax5.plot(np.cumsum(error_val["Random_10"]), label="Random_10")
    ax5.plot(np.cumsum(error_val["Random_20"]), label="Random_20")
    ax5.plot(np.cumsum(error_val["Random_40"]), label="Random_40")
    ax5.plot(np.cumsum(error_val["Quasi"]), label="Quasi")
    ax5.plot(np.cumsum(error_val["Sep_BO"]), label="SBO")
    ax5.legend()
    ax5.set_xlabel("# of counterfactuals computed")
    ax5.set_ylabel("Accumulated Error - Distance to optimal value")

def main():
    """Main function"""
    data_path = "src\data\Income1.csv"
    # read data
    df_train = pd.read_csv(data_path)
    df_train = df_train.rename(columns = lambda x:re.sub('[^A-Za-z0-9_]+', '', x))
    df_train = df_train.drop(columns=['Unnamed0'])
    std = df_train.drop("Income", axis=1).std().to_numpy().reshape(-1)

    # print(df_train.head())

    # visualize data
    # visualize(df_train)

    # train model
    # model = linear_model_train(df_train)
    # visualize_regressor(df_train, model)
    # model = dct_model_train(df_train)
    # visualize_regressor(df_train, model)
    # model = svr_model_train(df_train)

    models = [dct_model_train(df_train), svr_model_train(df_train), linear_model_train(df_train)]
    folders = ["dct", "svr", "linear"]
    for model, folder in zip(models, folders):
         # Experiment 1 - 1D
        experiment_1_1D(df_train, model.predict, folder)

    # visualize regressor
    # visualize_regressor(df_train, model)

    # # Try BO on regressor
    # BO_on_regressor(model.predict, df_train)

    # # for seeing objective function
    # x = np.array([[14]])
    # y_prim = 60
    # lam = 10 / y_prim**2
    # print("x, punkt i fråga", x)
    # print("y_prim, önskad output", y_prim)
    # print("model.predict(x), nuvarande output", model.predict(x))
    # visualize_f(f, model, x, y_prim, lam, df_train)

    # # init f for BO
    # func = lambda x: model.predict(x)

    # # objective function for BO
    # x = np.array([[14]])
    # y_prim = 50
    # lam = 5/y_prim**2
    # func = lambda x_prim: f(x, x_prim, y_prim, model.predict(x_prim), lam, std=std)
    # print("x, punkt i fråga", x)
    # print("y_prim, önskad output", y_prim)
    # print("f(x), nuvarande output", func(x))
    # BO_on_regressor(func, df_train)

    # # For testing own acquisition function
    # # define functions
    # x = np.array([[14]])
    # y_prim = 50
    # lam = 10/y_prim**2
    # objective_func = lambda x_prim, func_val: f(x, x_prim, y_prim, func_val, lam, std=std)
    # dist_func = lambda x_prim, current_point: np.linalg.norm(np.divide(x_prim-current_point, std), axis=1)
    # acq_func = CF_acquisition(dist_func, y_prim, x, lam).get_CF_EI()
    # bo_test_own(model.predict, objective_func, acq_func, df_train, y_prim=y_prim)

    # # compare with other optimizers
    # x = np.array([[14]])
    # y_prim = 50
    # lam = 5/y_prim**2
    # objective_func = lambda x_prim, func_val: f(x, x_prim, y_prim, func_val, lam, std=std)
    # opt_func = lambda x_prim: objective_func(x_prim, model.predict(x_prim))
    # dist_func = lambda x_prim, current_point: np.linalg.norm(np.divide(x_prim-current_point, std), axis=1)
    # acq_func = CF_acquisition(dist_func, y_prim, x, lam).get_CF_EI()
    # comparison(model.predict, df_train, opt_func, objective_func, acq_func, y_prim)

    plt.show()



if __name__ == "__main__":
    main()