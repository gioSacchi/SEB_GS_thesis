import numpy as np
import scipy.stats as ss
import matplotlib.pyplot as plt

def test_3_moment():
    mus = np.linspace(-4, 4, 10)
    sigmas = np.linspace(0, 5, 10)
    
    nr_points = 1_000_0
    diffs = []
    for mu in mus:
        for sigma in sigmas:
            # calc approx
            rv = np.random.normal(mu, sigma, nr_points)
            approx = np.mean(rv**3)

            # exact
            exact = mu**3 + 3*sigma**2*mu
            diffs.append(np.abs(approx-exact))
    print(np.mean(diffs))
    print(np.max(diffs))
    plt.plot(range(len(diffs)), diffs)
    plt.show()

def test_4_moment():
    mus = np.linspace(-4, 4, 10)
    sigmas = np.linspace(0, 5, 10)
    
    nr_points = 1_000_000_0
    diffs = []
    for mu in mus:
        for sigma in sigmas:
            # calc approx
            rv = np.random.normal(mu, sigma, nr_points)
            approx = np.mean(rv**4)

            # exact
            exact = mu**4 + 3*sigma**4 + 6*mu**2*sigma**2
            diffs.append(np.abs(approx-exact))
    print(np.mean(diffs))
    print(np.max(diffs))
    plt.plot(range(len(diffs)), diffs)
    plt.show()

def diff_sqaure():
    mus = np.linspace(-4, 4, 10)
    sigmas = np.linspace(0, 5, 10)
    y_prim = 3.5
    nr_points = 1_000_000_0
    diffs = []
    for mu in mus:
        for sigma in sigmas:
            # calc approx
            rv = np.random.normal(mu, sigma, nr_points)
            approx = np.mean((rv-y_prim)**2)

            # exact
            exact = y_prim**2 - 2*y_prim*mu + mu**2 + sigma**2
            diffs.append(np.abs(approx-exact))
    print(np.mean(diffs))
    plt.plot(diffs)
    plt.show()

def diff_quad():
    mus = np.linspace(-4, 4, 10)
    sigmas = np.linspace(0, 5, 10)
    y_prim = 3.5
    nr_points = 1_000_000_0
    diffs = []
    for mu in mus:
        for sigma in sigmas:
            # calc approx
            rv = np.random.normal(mu, sigma, nr_points)
            approx = np.mean((rv-y_prim)**4)

            # exact
            f1 = mu**4 + 6*mu**2*sigma**2 + 3*sigma**4
            f2 = mu**3 + 3*mu*sigma**2
            f3 = mu**2 + sigma**2
            f4 = mu
            exact = f1 - 4*f2*y_prim + 6*f3*y_prim**2 - 4*f4*y_prim**3 + y_prim**4
            diffs.append(np.abs(approx-exact))
    print(np.mean(diffs))
    print(np.max(diffs))
    plt.plot(diffs)
    plt.show()

if __name__ == "__main__":
    # diff_sqaure()
    # test_3_moment()
    # test_4_moment()
    diff_quad()