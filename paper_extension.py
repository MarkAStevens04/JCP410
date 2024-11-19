import h5py
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import h5pandas as h5pd
import main


# ---------------------------------------------------------------------------------------------------------------------
# Get GFP up and running
def dYdt2(t, Y, p):
    # p: (lambda_m, lambda_p, beta_m, beta_p, h, K)
    # Y: [m1, P1, m2, P2, m3, P3, m4, P4]
    m1 = (p[0] * (p[5] ** p[4])) / (p[5] ** p[4] + Y[5] ** p[4]) - Y[0] * (p[2] + p[3])
    p1 = Y[0] * p[1] - Y[1] * p[3]
    m2 = (p[0] * (p[5] ** p[4])) / (p[5] ** p[4] + Y[1] ** p[4]) - Y[2] * (p[2] + p[3])
    p2 = Y[2] * p[1] - Y[3] * p[3]
    m3 = (p[0] * (p[5] ** p[4])) / (p[5] ** p[4] + Y[3] ** p[4]) - Y[4] * (p[2] + p[3])
    p3 = Y[4] * p[1] - Y[5] * p[3]
    m4 = (p[0] * (p[5] ** p[4])) / (p[5] ** p[4] + Y[5] ** p[4]) - Y[6] * (p[2] + p[3])
    p4 = Y[6] * p[1] - Y[7] * p[3]
    return [m1, p1, m2, p2, m3, p3, m4, p4]

def rvf_gfp(x, p):
    """
    Calculates the rates of rxns using current quantities of each species
    :param x: [m1, P1, m2, P2, m3, P3, m4, P4] (array of molecule numbers)
    :param p: [lambda_m, lambda_p, beta_m, beta_p, h, K] Parameter vector for ODE solver
    :return:
    """
    # This is where you put in your RVF!
    # return a 1x12 matrix
    # print(f'running rvf {x}')
    # print(f'x at 0: {x[0]}')
    # POSSIBLE ERROR:
    # deg_m might be (beta_m * x[]) NOT (beta_m + beta_p) * x[]
    # print(f'p: {p}')
    lambda_m = p[0]
    lambda_p = p[1]
    beta_m = p[2]
    beta_p = p[3]
    h = p[4]
    K = p[5]


    prod_m1 = (lambda_m * K**h) / (K**h + x[5] ** h)
    deg_m1 = (beta_m + beta_p) * x[0]

    prod_p1 = lambda_p * x[0]
    deg_p1 = beta_p * x[1]

    prod_m2 = (lambda_m * K ** h) / (K ** h + x[1] ** h)
    deg_m2 = (beta_m + beta_p) * x[2]

    prod_p2 = lambda_p * x[2]
    deg_p2 = beta_p * x[3]

    prod_m3 = (lambda_m * K ** h) / (K ** h + x[3] ** h)
    deg_m3 = (beta_m + beta_p) * x[4]

    prod_p3 = lambda_p * x[4]
    deg_p3 = beta_p * x[5]

    prod_m4 = (lambda_m * K ** h) / (K ** h + x[5] ** h)
    deg_m4 = (beta_m + beta_p) * x[6]

    prod_p4 = lambda_p * x[6]
    deg_p4 = beta_p * x[7]
    return [prod_m1, deg_m1, prod_p1, deg_p1, prod_m2, deg_m2, prod_p2, deg_p2, prod_m3, deg_m3, prod_p3, deg_p3, prod_m4, deg_m4, prod_p4, deg_p4]


# ---------------------------------------------------------------------------------------------------------------------
# Test degradation interference


def rvf_gfp(x, p):
    """
    Calculates the rates of rxns using current quantities of each species
    :param x: [m1, P1, m2, P2, m3, P3, m4, P4] (array of molecule numbers)
    :param p: [lambda_m, lambda_p, beta_m, beta_p, h, K] Parameter vector for ODE solver
    :return:
    """
    # This is where you put in your RVF!
    # return a 1x12 matrix
    # print(f'running rvf {x}')
    # print(f'x at 0: {x[0]}')
    # POSSIBLE ERROR:
    # deg_m might be (beta_m * x[]) NOT (beta_m + beta_p) * x[]
    # print(f'p: {p}')
    lambda_m = p[0]
    lambda_p = p[1]
    beta_m = p[2]
    beta_p = p[3]
    h = p[4]
    K = p[5]


    prod_m1 = (lambda_m * K**h) / (K**h + x[5] ** h)
    deg_m1 = (beta_m + beta_p) * x[0]

    prod_p1 = lambda_p * x[0]
    deg_p1 = beta_p * x[1]

    prod_m2 = (lambda_m * K ** h) / (K ** h + x[1] ** h)
    deg_m2 = (beta_m + beta_p) * x[2]

    prod_p2 = lambda_p * x[2]
    deg_p2 = beta_p * x[3]

    prod_m3 = (lambda_m * K ** h) / (K ** h + x[3] ** h)
    deg_m3 = (beta_m + beta_p) * x[4]

    prod_p3 = lambda_p * x[4]
    deg_p3 = beta_p * x[5]

    prod_m4 = (lambda_m * K ** h) / (K ** h + x[5] ** h)
    deg_m4 = (beta_m + beta_p) * x[6]

    prod_p4 = lambda_p * x[6]
    deg_p4 = beta_p * x[7]
    return [prod_m1, deg_m1, prod_p1, deg_p1, prod_m2, deg_m2, prod_p2, deg_p2, prod_m3, deg_m3, prod_p3, deg_p3, prod_m4, deg_m4, prod_p4, deg_p4]





def plot_gfp_det():
    """
       Plots the deterministic model for GFP
       :return:
    """
    t = np.linspace(0, 50, 1000)
    x0 = [10, 0, 0, 0, 0, 0, 0, 0]
    sol = main.deterministic(0.277, 380, t, x0=x0, ode=dYdt2, p=("cat",))
    # [m1, P1, m2, P2, m3, P3, m4, p4]

    p1_sol = sol.y[1]
    p2_sol = sol.y[3]
    p3_sol = sol.y[5]
    p4_sol = sol.y[7]
    t = sol.t

    plt.figure(figsize=(8, 8))

    plt.plot(t, p1_sol, label="P1", color="#f94144")
    plt.plot(t, p2_sol, label="P2", color="#f9c74f")
    plt.plot(t, p3_sol, label="P3", color="#577590")
    plt.plot(t, p4_sol, label="P4", color="#06d6a0")

    plt.xlabel("Time", fontsize=15)
    plt.ylabel("Copy Number", fontsize=15)
    plt.title("GFP Deterministic", fontsize=15)

    plt.legend(loc="upper left")
    plt.subplots_adjust(bottom=0.1, left=0.1)

    plt.show()




def plot_gfp_stoch():
    """
    Create time trace graph from algorithm run.
    :return:
    """
    # reactions: prod_m1, deg_m1, prod_P1, deg_P1, prod_m2, deg_m2, prod_P2, deg_P2, prod_m3, deg_m3, prod_P3, deg_P3, prod_m4, deg_m4, prod_P4, deg_P4
    stoich_mat = [
        [1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, -1, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 1, -1, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, -1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, -1, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, -1]]

    x0_g = np.zeros((8, 1))
    x0_g[0, 0] = 10

    # *** NOTE: H should be set to 2, but is set to 1 by default!! ***
    # *** NOTE: H should be set to 2, but is set to 1 by default!! ***
    rt, rx, peaks, autoc = main.single_pass(0.277, 380, 2, 100000, K=7, stoich_mat=stoich_mat, rvf=rvf_gfp, x0_g=x0_g, p=["cat"])
    # - Regular Graph -
    p1_r = rx[1, :]
    p2_r = rx[3, :]
    p3_r = rx[5, :]
    p4_r = rx[7, :]

    plt.figure(figsize=(8, 8))

    # plt.plot(rt, p1_r, label="P1", color="#f94144")
    # plt.plot(rt, p2_r, label="P2", color="#f9c74f")
    # plt.plot(rt, p3_r, label="P3", color="#577590")
    plt.plot(rt, p4_r, label="P4", color="#06d6a0")

    plt.xlabel("Time", fontsize=15)
    plt.ylabel("Copy Number", fontsize=15)
    plt.title("GFP Stochastic", fontsize=15)

    plt.legend(loc="upper left")
    plt.subplots_adjust(bottom=0.1, left=0.1)

    # - Autocorrelation graph -
    # plt.plot(rt, autoc)

    plt.show()



# *** NOTE: H should be set to 2, but is set to 1 by default!! ***




if __name__ == "__main__":
    # plot_gfp_det()
    plot_gfp_stoch()