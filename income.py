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

def experiment_1_1D(df_train, model):
    # init data
    input_data = df_train.drop("Income", axis=1)
    output_data = df_train["Income"]
    std = input_data.std().to_numpy().reshape(-1)
    min1 = input_data.iloc[:, 0].min()
    max1 = input_data.iloc[:, 0].max()
    margin1 = (max1 - min1) * 0.1
    bounds = np.array([[min1 - margin1, max1 + margin1]])

    grid = 100

    # generated points between min and max of data
    x = np.linspace(input_data.min(), input_data.max(), grid)
    current_points = x.reshape(-1, 1)
    # generate grid of desired output values between min and max of output data
    desired = np.linspace(output_data.min(), output_data.max(), grid)
    base_lambda = 10

    n_starting_points = 2
    n_max_iterations = 15
    n_stop = 4

    evals = {"BO": [], "Random": [], "Quasi": [], "Sep_BO": []}
    error = {"BO": [], "Random": [], "Quasi": [], "Sep_BO": []}
    for point in current_points:
        for y_prim in desired:
            # define objective function
            lam = base_lambda/y_prim**2
            objective_func = lambda x_prim, func_val: f(point, x_prim, y_prim, func_val, lam, std=std)
            opt_func = lambda x_prim: f(point, x_prim, y_prim, model.predict(x_prim), lam, std=std)
            dist_func = lambda x_prim, current_point: np.linalg.norm(np.divide(x_prim-current_point, std), axis=1)
            acq_func = CF_acquisition(dist_func, y_prim, x, lam).get_CF_EI()
            # run experiment
            # Normal BO
            bo_normal = BayesianOptimization(f = opt_func, acquisition="EI", dim = 1, bounds = bounds, 
                                      n_iter=n_max_iterations, n_init=n_starting_points, n_stop_iter=n_stop, 
                                      noise_std=0, normalize_Y=True)
            bo_normal.run_BO()
            evals["BO"].append(bo_normal.number_of_evaluations_f)
            # Separated BO
            bo_separated = BayesianOptimization(f = model.predict, obj_func=objective_func, acquisition=acq_func, dim = 1, 
                                                bounds = bounds, n_iter=n_max_iterations, n_init=n_starting_points,  
                                                n_stop_iter=n_stop, noise_std=0, normalize_Y=True)
            bo_separated.run_BO()
            evals["Sep_BO"].append(bo_separated.number_of_evaluations_f)
            # Random
            alt_random = ComparisonOptimizers(opt_func, 1, bounds=bounds, method = 'random',
                                                n_iter=n_max_iterations, n_init=n_starting_points, n_stop_iter=n_stop,
                                                noise_std=0)
            alt_random.run()
            evals["Random"].append(alt_random.n_evals)
            # Quasi
            alt_quasi = ComparisonOptimizers(opt_func, 1, bounds=bounds, method='l-bfgs-b',
                                                n_iter=n_max_iterations, n_init=n_starting_points, n_stop_iter=n_stop,
                                                noise_std=0)
            alt_quasi.run()
            evals["Quasi"].append(alt_quasi.n_evals)









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
    model = dct_model_train(df_train)
    # visualize_regressor(df_train, model)
    # model = svr_model_train(df_train)

    # visualize regressor
    # visualize_regressor(df_train, model)

    # # Try BO on regressor
    # BO_on_regressor(model.predict, df_train)

    # for seeing objective function
    x = np.array([[14]])
    y_prim = 60
    lam = 10 / y_prim**2
    print("x, punkt i fråga", x)
    print("y_prim, önskad output", y_prim)
    print("model.predict(x), nuvarande output", model.predict(x))
    visualize_f(f, model, x, y_prim, lam, df_train)

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