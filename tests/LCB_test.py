import numpy as np
import scipy.stats as ss
import matplotlib.pyplot as plt

def exact(mu, sigma, dist, lam, y_prim, beta):
    f1 = y_prim**2 - 2*y_prim*mu + mu**2 + sigma**2
    f2 = y_prim**4 - 4*y_prim**3*mu + 6*y_prim**2*(mu**2 + sigma**2) - 4*y_prim*mu*(mu**2 + 3*sigma**2) + mu**4 + 6*mu**2*sigma**2 + 3*sigma**4
    mean = dist + lam*f1
    var = dist**2 + 2*dist*lam*f1 + f2*lam**2 - mean**2
    return mean - beta*np.sqrt(var)

def short_exact(mu, sigma, dist, lam, y_prim, beta):
    f1 = (y_prim - mu)**2 + sigma**2
    mean = dist + lam*f1
    var = 2*lam**2*sigma**2*(2*(y_prim - mu)**2+sigma**2)
    return mean - beta*np.sqrt(var)

def approx(dist, lam, y_prim, rv, beta):
    c = dist + lam*(rv-y_prim)**2
    mean = np.mean(c)
    std = np.std(c)
    return mean - beta*std

def test():
    y_prim = 3.5
    x_prim = 11
    
    lam = 400
    x = 10

    mus = np.linspace(-4, 4, 10)
    sigmas = np.linspace(0.1, 4, 10)
    diffs = []
    analytic = []
    for mu in mus:
        for sigma in sigmas:
            rv = np.random.normal(mu, sigma, 1000_0)
            dist = np.linalg.norm(x-x_prim)
            beta = 1
            EI_exact = exact(mu, sigma, dist, lam, y_prim, beta)
            EI_exact_short = short_exact(mu, sigma, dist, lam, y_prim, beta)
            EI_mc = approx(dist, lam, y_prim, rv, beta)
            # print("EI_exact: ", EI_exact)
            # print("EI_mc: ", EI_mc)
            # print("EI_exact_short: ", EI_exact_short)
            # print("EI_mc: ", EI_mc)
            diffs.append(abs(EI_exact - EI_mc))
            analytic.append(abs(EI_exact_short - EI_exact_short))
    print(np.mean(analytic))
    print(np.mean(diffs))
    print(np.max(diffs))
    plt.figure()
    plt.plot(diffs)
    plt.xlabel("Lambda")
    plt.ylabel("Difference between exact and MC approximation")
    plt.show()

test()