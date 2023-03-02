import numpy as np
import scipy.stats as ss
import matplotlib.pyplot as plt

def test():
    # Define values
    c_OPT = 3
    y_prim = 3.5
    x_prim = 11
    
    lam = 400
    x = 10
    
    # Define the distribution and generate random values
    mu = 3.45
    sigma = 0.01
    nr_points = 1000000
    rv = np.random.normal(mu, sigma, nr_points)

    mus = [3.3, 3.33, 3.36, 3.39, 3.42, 3.45, 3.48, 3.51, 3.54, 3.57, 3.6, 3.63, 3.66, 3.69, 3.72, 3.75, 3.78]
    diffs = []
    for mu in mus:
        rv = np.random.normal(mu, sigma, nr_points)
        # Calculate improtant data
        dist = np.linalg.norm(x-x_prim)
        UB = np.sqrt((c_OPT-dist)/lam) + y_prim
        LB = -np.sqrt((c_OPT-dist)/lam) + y_prim
        UB_arg = (UB - mu)/sigma
        LB_arg = (LB - mu)/sigma
        CDF_diff = ss.norm.cdf(UB_arg) - ss.norm.cdf(LB_arg)
        PDF_UB = ss.norm.pdf(UB_arg)
        PDF_LB = ss.norm.pdf(LB_arg)

        EI_exact = explicit(c_OPT, y_prim, lam, mu, sigma, dist, CDF_diff, PDF_UB, PDF_LB, UB, LB)
        EI_mc = approx(c_OPT, y_prim, lam, dist, rv)
        print("EI_exact: ", EI_exact)
        print("EI_mc: ", EI_mc)
        
        diffs.append(abs(EI_exact - EI_mc))
    
    plt.figure()
    plt.plot(mus, diffs)
    plt.xlabel("Lambda")
    plt.ylabel("Difference between exact and MC approximation")
    plt.show()

def OG_test():
    # Define values
    c_OPT = 3    
    
    # Define the distribution and generate random values
    mu = 3
    sigma = 2
    nr_points = 1000000
    rv = np.random.normal(mu, sigma, nr_points)


    EI_exact = OG_explicit(c_OPT, mu, sigma)
    EI_mc = OG_approx(c_OPT, rv)

    print("EI_exact: ", EI_exact)
    print("EI_mc: ", EI_mc)

def explicit(c_OPT, y_prim, lam, mu, sigma, dist, CDF_diff, PDF_UB, PDF_LB, UB, LB):
    if sigma == 0:
        return 0
    first_term = c_OPT-dist+ lam*(2*y_prim*mu-y_prim**2-mu**2-sigma**2)
    second_term = lam*sigma*(mu+UB-2*y_prim)
    third_term = lam*sigma*(2*y_prim-mu-LB)
    EI_exact = first_term*CDF_diff + second_term*PDF_UB + third_term*PDF_LB
    return EI_exact

def approx(c_OPT, y_prim, lam, dist, rv):
    c_hats = dist+ lam*(rv-y_prim)**2
    c_diff = c_OPT - c_hats
    c_diff[c_diff<0] = 0
    EI_mc = np.mean(c_diff)
    return EI_mc

def OG_approx(c_OPT, rv):
    c_diff = c_OPT - rv
    c_diff[c_diff<0] = 0
    EI_mc = np.mean(c_diff)
    return EI_mc

def OG_explicit(c_OPT, mu, sigma):
    if sigma == 0:
        return 0
    EI_exact = (c_OPT-mu)*ss.norm.cdf((c_OPT-mu)/sigma) + sigma*ss.norm.pdf((c_OPT-mu)/sigma)
    return EI_exact

if __name__ == "__main__":
    test()