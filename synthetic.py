import pandas as pd
import matplotlib.pyplot as plt
import re
import numpy as np


from BO import BayesianOptimization
from src.features.CF_acq import CF_acquisition
from ALT_opt import ComparisonOptimizers

def visualize(df_train):
    """Visualize data"""
    # plot the data points, in 2D if one input feature, in 3D if two input features
    output_data = df_train["LoanAmount"]
    input_data = df_train.drop("LoanAmount", axis=1)
    plt.scatter(input_data, output_data)
    plt.xlabel(df_train.columns[0])
    plt.ylabel(output_data.name)

def visualize_synthetic(model):
    # plot model predictions on a grid of points between 15 and 80
    x = np.linspace(15, 90, 200)
    y = model.predict(x.reshape(-1, 1))
    plt.plot(x, y, color="red", label="loan model")
    plt.axhline(y=800, color='green', linestyle='dotted', label="desired loan amount")
    # plot line showing current loan amount for income of 60, the line shoudl go from 0 to the model's prediction for 60
    point = 45
    y = model.predict(point)
    plt.plot([point, point], [0, y], color="black", linestyle='dotted', label="current loan amount")
    plt.plot([0, point], [y, y], color="black", linestyle='dotted')
    # set window limits
    plt.xlim(12, 93)
    plt.ylim(0, 1300)

    plt.legend()

    plt.xlabel("Income")
    plt.ylabel("Loan Amount in Thousands")

def synthetic_model():
    # make a model class that implements the predict method
    # return an instance of the model class
    class Model:
        def predict(self, X):
            # return 100*(X+10) + 1000*np.sin(X/10) - 0.1*X**2
            # return 1000*np.log(X-20) + 100*np.sin(X/10)
            # return 262.8235 - 10.56857*X - 0.3960956*X**2 + 0.0477614*X**3 - 0.0008690937*X**4 + 0.000004640347*X**5
            # return -0.0000000008575*X**7 + 0.0000000700809*X**6 + 0.0000197931944*X**5 - 0.003217025319*X**4 + 0.172741851887*X**3 - 3.6954003727141*X**2 + 34.0745482930487*X
            return 0.0000000038232*X**7 - 0.0000014199948*X**6 + 0.0002078597378*X**5 - 0.0151606312126*X**4 + 0.571270644072*X**3 -10.2471567393055*X**2 + 75.0351163349937*X
    return Model()

def visualize_f_syn(f, model, x, des_out, lam, data):
    # plot model predictions on a grid of points between 15 and 80
    std = data.drop("LoanAmount", axis=1).std().to_numpy().reshape(-1)
    fig = plt.figure()
    print("std", std)
    x_min, x_max = data.drop("LoanAmount", axis=1).min(), data.drop("LoanAmount", axis=1).max()
    x_prim = np.linspace(x_min, x_max, 100).reshape(-1, 1)
    y = f(x, x_prim, des_out, model.predict(x_prim).reshape(-1,1), lam, std=std)
    plt.plot(x_prim, y, color="magenta")
    plt.xlabel("Income")
    plt.ylabel("counterfactual value")

def separated_BO(func, obj_func, acq, df_train, y_prim):
    # handle cases of 1 and 2 input features and
    # define bounds from data and add a margin which is 10% of the range of the data in each dimension
    # output_data = df_train["Income"]
    input_data = df_train.drop("LoanAmount", axis=1)
    min1 = input_data.iloc[:, 0].min()
    max1 = input_data.iloc[:, 0].max()
    margin1 = (max1 - min1) * 0.1
    bounds = np.array([[min1 - margin1, max1 + margin1]])
    bo = BayesianOptimization(f = func, obj_func=obj_func, acquisition=acq, dim = input_data.shape[1], 
                              bounds = bounds, n_iter=10*input_data.shape[1]**2, n_init=2*input_data.shape[1], noise_std=0, normalize_Y=True)
    bo.run_BO()
    print(bo.number_of_evaluations_f)
    bo.make_plots(y_prim=y_prim)

def BO_on_regressor(model, df_train):
    # handle cases of 1 and 2 input features and
    # define bounds from data and add a margin which is 10% of the range of the data in each dimension
    # output_data = df_train["Income"]
    input_data = df_train.drop("LoanAmount", axis=1)
    min1 = input_data.iloc[:, 0].min()
    max1 = input_data.iloc[:, 0].max()
    margin1 = (max1 - min1) * 0.1
    bounds = np.array([[min1 - margin1, max1 + margin1]])
    bo = BayesianOptimization(f = model, dim = input_data.shape[1],bounds = bounds, n_iter=10*input_data.shape[1]**2, n_init=2*input_data.shape[1], noise_std=0, normalize_Y=True)
    bo.run_BO()
    print(bo.number_of_evaluations_f)
    bo.make_plots()

def f(x, x_prim, y_prim, computed_val, lam, std=None):
    if std is not None:
        # check std right dimension
        if std.shape[0] != x.shape[1]:
            raise Exception("std has wrong dimension")
        x = np.divide(x, std)
        x_prim = np.divide(x_prim, std)
    val = np.linalg.norm(x - x_prim, axis=1).reshape(-1,1) + (lam * (computed_val - y_prim)**2).reshape(-1,1)
    return val

def main():
    data_path = "src\data\synthetic.csv"
    # read data
    df_train = pd.read_csv(data_path)
    df_train = df_train.rename(columns = lambda x:re.sub('[^A-Za-z0-9_]+', '', x))
    df_train = df_train.drop(columns=['Unnamed0'])
    print(df_train.head())
    std = df_train.drop("LoanAmount", axis=1).std().to_numpy().reshape(-1)


    model = synthetic_model()
    # visualize_synthetic(model)
    x = np.array([[45]])
    y_prim = 800
    lam = 100 / y_prim**2
    print("x, punkt i fråga", x)
    print("y_prim, önskad output", y_prim)
    print("model.predict(x), nuvarande output", model.predict(x))
    visualize_f_syn(f, model, x, y_prim, lam, df_train)

    # objective function for BO
    x = np.array([[45]])
    y_prim = 800
    lam = 100/y_prim**2
    func = lambda x_prim: f(x, x_prim, y_prim, model.predict(x_prim), lam, std=std)
    print("x, punkt i fråga", x)
    print("y_prim, önskad output", y_prim)
    print("f(x), nuvarande output", func(x))
    BO_on_regressor(func, df_train)

    # # For testing own acquisition function
    # # define functions
    # x = np.array([[45]])
    # y_prim = 800
    # lam = 10/y_prim**2
    # objective_func = lambda x_prim, func_val: f(x, x_prim, y_prim, func_val, lam, std=std)
    # dist_func = lambda x_prim, current_point: np.linalg.norm(np.divide(x_prim-current_point, std), axis=1)
    # acq_func = CF_acquisition(dist_func, y_prim, x, lam).get_CF_EI()
    # separated_BO(model.predict, objective_func, acq_func, df_train, y_prim=y_prim)


    plt.show()

if __name__ == "__main__":
    main()