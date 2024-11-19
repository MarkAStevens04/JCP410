import h5py
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import h5pandas as h5pd
import main

# = General Layout of Experiments =
# RVFs are for stochastic simulations, dYdt are for deterministic.
# We define wrappers to help quickly set up parameters.



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




def gfp_wrapper_stoch(plt):
    """
    Performs a run including gfp.
    Uses rvf_gfp
    :return: Updated plt
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
    rt, rx, peaks, autoc = main.single_pass(0.277, 380, 2, 100000, K=7, stoich_mat=stoich_mat, rvf=rvf_gfp, x0_g=x0_g)

    # - Regular Graph -
    p1_r = rx[1, :]
    p2_r = rx[3, :]
    p3_r = rx[5, :]
    p4_r = rx[7, :]

    plt.plot(rt, p1_r, label="P1", color="#f94144")
    plt.plot(rt, p2_r, label="P2", color="#f9c74f")
    plt.plot(rt, p3_r, label="P3", color="#577590")
    plt.plot(rt, p4_r, label="P4", color="#06d6a0")

    return plt



# ---------------------------------------------------------------------------------------------------------------------
# Adding Hydrogen Peroxide

def rvf_h2o2(x, p):
    """
    Calculates the rates of rxns using current quantities of each species
    :param x: [m1, P1, m2, P2, m3, P3, m4, P4, h] (array of molecule numbers)
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

    # * 60 b/c we're in molecules / sec and we need to be in molecules/min
    # stoich is number of hydrogen peroxide generated per GFP.
    stoich = 10
    prod_h = (5835 * 60) + stoich * x[7]
    V_max = 1.2 * (10 ** 6) * 60
    K = 7.2*(10**17) * 60
    deg_h = 82.6 * x[8] * 60 + (x[8] * V_max) / (x[8] + K)

    return [prod_m1, deg_m1, prod_p1, deg_p1, prod_m2, deg_m2, prod_p2, deg_p2,
            prod_m3, deg_m3, prod_p3, deg_p3, prod_m4, deg_m4, prod_p4, deg_p4,
            prod_h, deg_h]




def h2o2_wrapper_stoch(plt):
    """
    Performs a run including gfp.
    Uses rvf_gfp
    :return: Updated plt
    """
    # reactions: prod_m1, deg_m1, prod_P1, deg_P1, prod_m2, deg_m2, prod_P2, deg_P2, prod_m3, deg_m3, prod_P3, deg_P3, prod_m4, deg_m4, prod_P4, deg_P4, prod_h2o2, deg_h2o2
    stoich_mat = [
        [1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 1, -1, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, -1, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, -1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, -1, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, -1]]

    x0_g = np.zeros((9, 1))
    x0_g[0, 0] = 10

    # *** NOTE: H should be set to 2, but is set to 1 by default!! ***
    # *** NOTE: H should be set to 2, but is set to 1 by default!! ***
    rt, rx, peaks, autoc = main.single_pass(0.277, 380, 2, 300000, K=7, stoich_mat=stoich_mat, rvf=rvf_h2o2, x0_g=x0_g, p=["cat"])

    # - Regular Graph -
    p1_r = rx[1, :]
    p2_r = rx[3, :]
    p3_r = rx[5, :]
    p4_r = rx[7, :]
    p5_r = rx[8, :]

    # plt.plot(rt, p1_r, label="P1", color="#f94144")
    # plt.plot(rt, p2_r, label="P2", color="#f9c74f")
    # plt.plot(rt, p3_r, label="P3", color="#577590")
    plt.plot(rt, p4_r, label="P4", color="#06d6a0")
    plt.plot(rt, p5_r, label="P5", color="#c1121f")

    return plt


# ---------------------------------------------------------------------------------------------------------------------
# Idea of Experiment 1A:
# TetR increases in concentration in response to Oxidative stress.
# Use regular rate constant to increase concentration of m3

def rvf_exp1a(x, p):
    """
    Calculates the rates of rxns using current quantities of each species
    :param x: [m1, P1, m2, P2, m3, P3, m4, P4, h] (array of molecule numbers)
    :param p: [lambda_m, lambda_p, beta_m, beta_p, h, K, alpha] Parameter vector for ODE solver
    :return:
    """
    # alpha is the parameter we should change!
    # alpha = 0.01 semi stable, 0.1 unstable
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

    prod_m3 = (lambda_m * K ** h) / (K ** h + x[3] ** h) + p[6] * x[8]
    deg_m3 = (beta_m + beta_p) * x[4]

    prod_p3 = lambda_p * x[4]
    deg_p3 = beta_p * x[5]

    prod_m4 = (lambda_m * K ** h) / (K ** h + x[5] ** h)
    deg_m4 = (beta_m + beta_p) * x[6]

    prod_p4 = lambda_p * x[6]
    deg_p4 = beta_p * x[7]

    # * 60 b/c we're in molecules / sec and we need to be in molecules/min
    # stoich is number of hydrogen peroxide generated per GFP.
    stoich = 19
    # h_generation = 5835 * 60
    # V_max = 1.2 * (10 ** 6) * 60
    # K = 7.2 * (10 ** 17) * 60
    # h_degrade = 82.6 * 60
    h_generation = 5835
    V_max = 1.2 * (10 ** 6)
    K = 7.2 * (10 ** 17)
    h_degrade = 82.6

    prod_h = h_generation + stoich * x[7]
    deg_h = h_degrade * x[8] + (x[8] * V_max) / (x[8] + K)

    return [prod_m1, deg_m1, prod_p1, deg_p1, prod_m2, deg_m2, prod_p2, deg_p2,
            prod_m3, deg_m3, prod_p3, deg_p3, prod_m4, deg_m4, prod_p4, deg_p4,
            prod_h, deg_h]




def exp1a_wrapper_stoch(plt):
    """
    Performs a run including gfp.
    Uses rvf_gfp
    :return: Updated plt
    """
    # reactions: prod_m1, deg_m1, prod_P1, deg_P1, prod_m2, deg_m2, prod_P2, deg_P2, prod_m3, deg_m3, prod_P3, deg_P3, prod_m4, deg_m4, prod_P4, deg_P4, prod_h2o2, deg_h2o2
    stoich_mat = [
        [1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 1, -1, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, -1, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, -1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, -1, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, -1]]

    x0_g = np.zeros((9, 1))
    x0_g[0, 0] = 10

    # *** NOTE: H should be set to 2, but is set to 1 by default!! ***
    # *** NOTE: H should be set to 2, but is set to 1 by default!! ***
    rt, rx, peaks, autoc = main.single_pass(0.277, 380, 2, 1000000, K=7, stoich_mat=stoich_mat, rvf=rvf_exp1a, x0_g=x0_g, p=[0.1])

    # - Regular Graph -
    p1_r = rx[1, :]
    p2_r = rx[3, :]
    p3_r = rx[5, :]
    p4_r = rx[7, :]
    p5_r = rx[8, :]

    plt.plot(rt, p5_r, label="P5", color="#c1121f")
    plt.plot(rt, p1_r, label="P1", color="#f94144")
    plt.plot(rt, p2_r, label="P2", color="#f9c74f")
    plt.plot(rt, p3_r, label="P3", color="#577590")
    plt.plot(rt, p4_r, label="P4", color="#06d6a0")


    return plt



# ---------------------------------------------------------------------------------------------------------------------
# Idea of Experiment 1B:
# TetR increases in concentration in response to Oxidative stress.
# Use Hill Equation to increase concentration of m3

def rvf_exp1b(x, p):
    """
    Calculates the rates of rxns using current quantities of each species
    :param x: [m1, P1, m2, P2, m3, P3, m4, P4, h] (array of molecule numbers)
    :param p: [lambda_m, lambda_p, beta_m, beta_p, h, K, lambda_h] Parameter vector for ODE solver
    :return:
    """
    # lambda_h is the parameter we should change!
    # lmabda_h = 1000 unstable, = 100 stable
    lambda_m = p[0]
    lambda_p = p[1]
    beta_m = p[2]
    beta_p = p[3]
    h = p[4]
    K = p[5]

    lambda_h = p[6]


    prod_m1 = (lambda_m * K**h) / (K**h + x[5] ** h)
    deg_m1 = (beta_m + beta_p) * x[0]

    prod_p1 = lambda_p * x[0]
    deg_p1 = beta_p * x[1]

    prod_m2 = (lambda_m * K ** h) / (K ** h + x[1] ** h)
    deg_m2 = (beta_m + beta_p) * x[2]

    prod_p2 = lambda_p * x[2]
    deg_p2 = beta_p * x[3]

    prod_m3 = (lambda_m * K ** h) / (K ** h + x[3] ** h) + (lambda_h * K ** h) / (K ** h + x[8] ** h)
    deg_m3 = (beta_m + beta_p) * x[4]

    prod_p3 = lambda_p * x[4]
    deg_p3 = beta_p * x[5]

    prod_m4 = (lambda_m * K ** h) / (K ** h + x[5] ** h)
    deg_m4 = (beta_m + beta_p) * x[6]

    prod_p4 = lambda_p * x[6]
    deg_p4 = beta_p * x[7]

    # * 60 b/c we're in molecules / sec and we need to be in molecules/min
    # stoich is number of hydrogen peroxide generated per GFP.
    stoich = 19
    # h_generation = 5835 * 60
    # V_max = 1.2 * (10 ** 6) * 60
    # K = 7.2 * (10 ** 17) * 60
    # h_degrade = 82.6 * 60
    h_generation = 5835
    V_max = 1.2 * (10 ** 6)
    K = 7.2 * (10 ** 17)
    h_degrade = 82.6

    prod_h = h_generation + stoich * x[7]
    deg_h = h_degrade * x[8] + (x[8] * V_max) / (x[8] + K)

    return [prod_m1, deg_m1, prod_p1, deg_p1, prod_m2, deg_m2, prod_p2, deg_p2,
            prod_m3, deg_m3, prod_p3, deg_p3, prod_m4, deg_m4, prod_p4, deg_p4,
            prod_h, deg_h]




def exp1b_wrapper_stoch(plt):
    """
    Performs a run including gfp.
    Uses rvf_gfp
    :return: Updated plt
    """
    # reactions: prod_m1, deg_m1, prod_P1, deg_P1, prod_m2, deg_m2, prod_P2, deg_P2, prod_m3, deg_m3, prod_P3, deg_P3, prod_m4, deg_m4, prod_P4, deg_P4, prod_h2o2, deg_h2o2
    stoich_mat = [
        [1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 1, -1, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, -1, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, -1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, -1, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, -1]]

    x0_g = np.zeros((9, 1))
    x0_g[0, 0] = 10

    # *** NOTE: H should be set to 2, but is set to 1 by default!! ***
    # *** NOTE: H should be set to 2, but is set to 1 by default!! ***
    rt, rx, peaks, autoc = main.single_pass(0.277, 380, 2, 1000000, K=7, stoich_mat=stoich_mat, rvf=rvf_exp1b, x0_g=x0_g, p=[100])

    # - Regular Graph -
    p1_r = rx[1, :]
    p2_r = rx[3, :]
    p3_r = rx[5, :]
    p4_r = rx[7, :]
    p5_r = rx[8, :]

    plt.plot(rt, p5_r, label="P5", color="#c1121f")
    plt.plot(rt, p1_r, label="P1", color="#f94144")
    plt.plot(rt, p2_r, label="P2", color="#f9c74f")
    plt.plot(rt, p3_r, label="P3", color="#577590")
    plt.plot(rt, p4_r, label="P4", color="#06d6a0")


    return plt

# ---------------------------------------------------------------------------------------------------------------------
# Idea of Experiment 2A:
# E Coli decreases global translation under oxidative stress.
# Simulate this by increasing degradation of mRNA in response to H2O2.
# DOES NOT include degradation of GFP mRNA

def rvf_exp2a(x, p):
    """
    Calculates the rates of rxns using current quantities of each species
    :param x: [m1, P1, m2, P2, m3, P3, m4, P4, h] (array of molecule numbers)
    :param p: [lambda_m, lambda_p, beta_m, beta_p, h, K, alpha] Parameter vector for ODE solver
    :return:
    """
    # alpha is the parameter we should change!
    # alpha = 0.01 semi unstable, 0.1 unstable, 0.001 semi unstable, 0.0001 semi unstable
    lambda_m = p[0]
    lambda_p = p[1]
    beta_m = p[2]
    beta_p = p[3]
    h = p[4]
    K = p[5]

    alpha = p[6]


    prod_m1 = (lambda_m * K**h) / (K**h + x[5] ** h)
    deg_m1 = (beta_m + beta_p) * x[0] + x[8] * alpha

    prod_p1 = lambda_p * x[0]
    deg_p1 = beta_p * x[1]

    prod_m2 = (lambda_m * K ** h) / (K ** h + x[1] ** h)
    deg_m2 = (beta_m + beta_p) * x[2] + x[8] * alpha

    prod_p2 = lambda_p * x[2]
    deg_p2 = beta_p * x[3]

    prod_m3 = (lambda_m * K ** h) / (K ** h + x[3] ** h) + p[6] * x[8]
    deg_m3 = (beta_m + beta_p) * x[4] + x[8] * alpha

    prod_p3 = lambda_p * x[4]
    deg_p3 = beta_p * x[5]

    prod_m4 = (lambda_m * K ** h) / (K ** h + x[5] ** h)
    deg_m4 = (beta_m + beta_p) * x[6]

    prod_p4 = lambda_p * x[6]
    deg_p4 = beta_p * x[7]

    # * 60 b/c we're in molecules / sec and we need to be in molecules/min
    # stoich is number of hydrogen peroxide generated per GFP.
    stoich = 19
    # h_generation = 5835 * 60
    # V_max = 1.2 * (10 ** 6) * 60
    # K = 7.2 * (10 ** 17) * 60
    # h_degrade = 82.6 * 60
    h_generation = 5835
    V_max = 1.2 * (10 ** 6)
    K = 7.2 * (10 ** 17)
    h_degrade = 82.6

    prod_h = h_generation + stoich * x[7]
    deg_h = h_degrade * x[8] + (x[8] * V_max) / (x[8] + K)

    return [prod_m1, deg_m1, prod_p1, deg_p1, prod_m2, deg_m2, prod_p2, deg_p2,
            prod_m3, deg_m3, prod_p3, deg_p3, prod_m4, deg_m4, prod_p4, deg_p4,
            prod_h, deg_h]




def exp2a_wrapper_stoch(plt):
    """
    Performs a run including gfp.
    Uses rvf_gfp
    :return: Updated plt
    """
    # reactions: prod_m1, deg_m1, prod_P1, deg_P1, prod_m2, deg_m2, prod_P2, deg_P2, prod_m3, deg_m3, prod_P3, deg_P3, prod_m4, deg_m4, prod_P4, deg_P4, prod_h2o2, deg_h2o2
    stoich_mat = [
        [1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 1, -1, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, -1, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, -1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, -1, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, -1]]

    x0_g = np.zeros((9, 1))
    x0_g[0, 0] = 10

    # *** NOTE: H should be set to 2, but is set to 1 by default!! ***
    # *** NOTE: H should be set to 2, but is set to 1 by default!! ***
    rt, rx, peaks, autoc = main.single_pass(0.277, 380, 2, 1000000, K=7, stoich_mat=stoich_mat, rvf=rvf_exp2a, x0_g=x0_g, p=[0.0001])

    # - Regular Graph -
    p1_r = rx[1, :]
    p2_r = rx[3, :]
    p3_r = rx[5, :]
    p4_r = rx[7, :]
    p5_r = rx[8, :]

    plt.plot(rt, p5_r, label="P5", color="#c1121f")
    plt.plot(rt, p1_r, label="P1", color="#f94144")
    plt.plot(rt, p2_r, label="P2", color="#f9c74f")
    plt.plot(rt, p3_r, label="P3", color="#577590")
    plt.plot(rt, p4_r, label="P4", color="#06d6a0")


    return plt



# ---------------------------------------------------------------------------------------------------------------------
# Idea of Experiment 2B:
# E Coli decreases global translation under oxidative stress.
# Simulate this by increasing degradation of mRNA in response to H2O2.
# DOES include degradation of GFP mRNA

def rvf_exp2b(x, p):
    """
    Calculates the rates of rxns using current quantities of each species
    :param x: [m1, P1, m2, P2, m3, P3, m4, P4, h] (array of molecule numbers)
    :param p: [lambda_m, lambda_p, beta_m, beta_p, h, K, alpha] Parameter vector for ODE solver
    :return:
    """
    # alpha is the parameter we should change!
    # alpha = 0.01 semi stable, 0.1 unstable, 0.001 mostly stable
    lambda_m = p[0]
    lambda_p = p[1]
    beta_m = p[2]
    beta_p = p[3]
    h = p[4]
    K = p[5]

    alpha = p[6]


    prod_m1 = (lambda_m * K**h) / (K**h + x[5] ** h)
    deg_m1 = (beta_m + beta_p) * x[0] + x[8] * alpha

    prod_p1 = lambda_p * x[0]
    deg_p1 = beta_p * x[1]

    prod_m2 = (lambda_m * K ** h) / (K ** h + x[1] ** h)
    deg_m2 = (beta_m + beta_p) * x[2] + x[8] * alpha

    prod_p2 = lambda_p * x[2]
    deg_p2 = beta_p * x[3]

    prod_m3 = (lambda_m * K ** h) / (K ** h + x[3] ** h) + p[6] * x[8]
    deg_m3 = (beta_m + beta_p) * x[4] + x[8] * alpha

    prod_p3 = lambda_p * x[4]
    deg_p3 = beta_p * x[5]

    prod_m4 = (lambda_m * K ** h) / (K ** h + x[5] ** h)
    deg_m4 = (beta_m + beta_p) * x[6] + x[8] * alpha

    prod_p4 = lambda_p * x[6]
    deg_p4 = beta_p * x[7]

    # * 60 b/c we're in molecules / sec and we need to be in molecules/min
    # stoich is number of hydrogen peroxide generated per GFP.
    stoich = 19
    # h_generation = 5835 * 60
    # V_max = 1.2 * (10 ** 6) * 60
    # K = 7.2 * (10 ** 17) * 60
    # h_degrade = 82.6 * 60
    h_generation = 5835
    V_max = 1.2 * (10 ** 6)
    K = 7.2 * (10 ** 17)
    h_degrade = 82.6

    prod_h = h_generation + stoich * x[7]
    deg_h = h_degrade * x[8] + (x[8] * V_max) / (x[8] + K)

    return [prod_m1, deg_m1, prod_p1, deg_p1, prod_m2, deg_m2, prod_p2, deg_p2,
            prod_m3, deg_m3, prod_p3, deg_p3, prod_m4, deg_m4, prod_p4, deg_p4,
            prod_h, deg_h]




def exp2b_wrapper_stoch(plt):
    """
    Performs a run including gfp.
    Uses rvf_gfp
    :return: Updated plt
    """
    # reactions: prod_m1, deg_m1, prod_P1, deg_P1, prod_m2, deg_m2, prod_P2, deg_P2, prod_m3, deg_m3, prod_P3, deg_P3, prod_m4, deg_m4, prod_P4, deg_P4, prod_h2o2, deg_h2o2
    stoich_mat = [
        [1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 1, -1, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, -1, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, -1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, -1, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, -1]]

    x0_g = np.zeros((9, 1))
    x0_g[0, 0] = 10

    # *** NOTE: H should be set to 2, but is set to 1 by default!! ***
    # *** NOTE: H should be set to 2, but is set to 1 by default!! ***
    rt, rx, peaks, autoc = main.single_pass(0.277, 380, 2, 1000000, K=7, stoich_mat=stoich_mat, rvf=rvf_exp2b, x0_g=x0_g, p=[0.0001])

    # - Regular Graph -
    p1_r = rx[1, :]
    p2_r = rx[3, :]
    p3_r = rx[5, :]
    p4_r = rx[7, :]
    p5_r = rx[8, :]

    plt.plot(rt, p5_r, label="P5", color="#c1121f")
    plt.plot(rt, p1_r, label="P1", color="#f94144")
    plt.plot(rt, p2_r, label="P2", color="#f9c74f")
    plt.plot(rt, p3_r, label="P3", color="#577590")
    plt.plot(rt, p4_r, label="P4", color="#06d6a0")


    return plt

# ---------------------------------------------------------------------------------------------------------------------
# Idea of Experiment 3A:
# Lambda cI increases in response to Oxidative stress.
# Use regular rate constant to increase concentration of m1

def rvf_exp3a(x, p):
    """
    Calculates the rates of rxns using current quantities of each species
    :param x: [m1, P1, m2, P2, m3, P3, m4, P4, h] (array of molecule numbers)
    :param p: [lambda_m, lambda_p, beta_m, beta_p, h, K, alpha] Parameter vector for ODE solver
    :return:
    """
    # alpha is the parameter we should change!
    # alpha = 0.1 unstable, 0.001 pretty much stable
    lambda_m = p[0]
    lambda_p = p[1]
    beta_m = p[2]
    beta_p = p[3]
    h = p[4]
    K = p[5]


    prod_m1 = (lambda_m * K**h) / (K**h + x[5] ** h) + p[6] * x[8]
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

    # * 60 b/c we're in molecules / sec and we need to be in molecules/min
    # stoich is number of hydrogen peroxide generated per GFP.
    stoich = 19
    # h_generation = 5835 * 60
    # V_max = 1.2 * (10 ** 6) * 60
    # K = 7.2 * (10 ** 17) * 60
    # h_degrade = 82.6 * 60
    h_generation = 5835
    V_max = 1.2 * (10 ** 6)
    K = 7.2 * (10 ** 17)
    h_degrade = 82.6

    prod_h = h_generation + stoich * x[7]
    deg_h = h_degrade * x[8] + (x[8] * V_max) / (x[8] + K)

    return [prod_m1, deg_m1, prod_p1, deg_p1, prod_m2, deg_m2, prod_p2, deg_p2,
            prod_m3, deg_m3, prod_p3, deg_p3, prod_m4, deg_m4, prod_p4, deg_p4,
            prod_h, deg_h]




def exp3a_wrapper_stoch(plt):
    """
    Performs a run including gfp.
    Uses rvf_gfp
    :return: Updated plt
    """
    # reactions: prod_m1, deg_m1, prod_P1, deg_P1, prod_m2, deg_m2, prod_P2, deg_P2, prod_m3, deg_m3, prod_P3, deg_P3, prod_m4, deg_m4, prod_P4, deg_P4, prod_h2o2, deg_h2o2
    stoich_mat = [
        [1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 1, -1, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, -1, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, -1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, -1, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, -1]]

    x0_g = np.zeros((9, 1))
    x0_g[0, 0] = 10

    # *** NOTE: H should be set to 2, but is set to 1 by default!! ***
    # *** NOTE: H should be set to 2, but is set to 1 by default!! ***
    rt, rx, peaks, autoc = main.single_pass(0.277, 380, 2, 1000000, K=7, stoich_mat=stoich_mat, rvf=rvf_exp3a, x0_g=x0_g, p=[0.01])

    # - Regular Graph -
    p1_r = rx[1, :]
    p2_r = rx[3, :]
    p3_r = rx[5, :]
    p4_r = rx[7, :]
    p5_r = rx[8, :]

    plt.plot(rt, p5_r, label="P5", color="#c1121f")
    plt.plot(rt, p1_r, label="P1", color="#f94144")
    plt.plot(rt, p2_r, label="P2", color="#f9c74f")
    plt.plot(rt, p3_r, label="P3", color="#577590")
    plt.plot(rt, p4_r, label="P4", color="#06d6a0")


    return plt



# ---------------------------------------------------------------------------------------------------------------------
# Idea of Experiment 3B:
# Lambda cI increases in concentration in response to Oxidative stress.
# Use Hill Equation to increase concentration of m1

def rvf_exp3b(x, p):
    """
    Calculates the rates of rxns using current quantities of each species
    :param x: [m1, P1, m2, P2, m3, P3, m4, P4, h] (array of molecule numbers)
    :param p: [lambda_m, lambda_p, beta_m, beta_p, h, K, lambda_h] Parameter vector for ODE solver
    :return:
    """
    # lambda_h is the parameter we should change!
    # lmabda_h = 1000 unstable, = 100 stable
    lambda_m = p[0]
    lambda_p = p[1]
    beta_m = p[2]
    beta_p = p[3]
    h = p[4]
    K = p[5]

    lambda_h = p[6]


    prod_m1 = (lambda_m * K**h) / (K**h + x[5] ** h) + (lambda_h * K ** h) / (K ** h + x[8] ** h)
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

    # * 60 b/c we're in molecules / sec and we need to be in molecules/min
    # stoich is number of hydrogen peroxide generated per GFP.
    stoich = 19
    # h_generation = 5835 * 60
    # V_max = 1.2 * (10 ** 6) * 60
    # K = 7.2 * (10 ** 17) * 60
    # h_degrade = 82.6 * 60
    h_generation = 5835
    V_max = 1.2 * (10 ** 6)
    K = 7.2 * (10 ** 17)
    h_degrade = 82.6

    prod_h = h_generation + stoich * x[7]
    deg_h = h_degrade * x[8] + (x[8] * V_max) / (x[8] + K)

    return [prod_m1, deg_m1, prod_p1, deg_p1, prod_m2, deg_m2, prod_p2, deg_p2,
            prod_m3, deg_m3, prod_p3, deg_p3, prod_m4, deg_m4, prod_p4, deg_p4,
            prod_h, deg_h]




def exp3b_wrapper_stoch(plt):
    """
    Performs a run including gfp.
    Uses rvf_gfp
    :return: Updated plt
    """
    # reactions: prod_m1, deg_m1, prod_P1, deg_P1, prod_m2, deg_m2, prod_P2, deg_P2, prod_m3, deg_m3, prod_P3, deg_P3, prod_m4, deg_m4, prod_P4, deg_P4, prod_h2o2, deg_h2o2
    stoich_mat = [
        [1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 1, -1, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, -1, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, -1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, -1, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, -1]]

    x0_g = np.zeros((9, 1))
    x0_g[0, 0] = 10

    # *** NOTE: H should be set to 2, but is set to 1 by default!! ***
    # *** NOTE: H should be set to 2, but is set to 1 by default!! ***
    rt, rx, peaks, autoc = main.single_pass(0.277, 380, 2, 1000000, K=7, stoich_mat=stoich_mat, rvf=rvf_exp3b, x0_g=x0_g, p=[1000])

    # - Regular Graph -
    p1_r = rx[1, :]
    p2_r = rx[3, :]
    p3_r = rx[5, :]
    p4_r = rx[7, :]
    p5_r = rx[8, :]

    plt.plot(rt, p5_r, label="P5", color="#c1121f")
    plt.plot(rt, p1_r, label="P1", color="#f94144")
    plt.plot(rt, p2_r, label="P2", color="#f9c74f")
    plt.plot(rt, p3_r, label="P3", color="#577590")
    plt.plot(rt, p4_r, label="P4", color="#06d6a0")


    return plt



# ---------------------------------------------------------------------------------------------------------------------
# Idea of Experiment 4A:
# Most vulnerable to a feedback loop
# Use regular rate constant to increase concentration of m2

def rvf_exp4a(x, p):
    """
    Calculates the rates of rxns using current quantities of each species
    :param x: [m1, P1, m2, P2, m3, P3, m4, P4, h] (array of molecule numbers)
    :param p: [lambda_m, lambda_p, beta_m, beta_p, h, K, alpha] Parameter vector for ODE solver
    :return:
    """
    # alpha is the parameter we should change!
    # alpha = 0.1 unstable, 0.001 pretty much stable
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

    prod_m2 = (lambda_m * K ** h) / (K ** h + x[1] ** h) + p[6] * x[8]
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

    # * 60 b/c we're in molecules / sec and we need to be in molecules/min
    # stoich is number of hydrogen peroxide generated per GFP.
    stoich = 19
    # h_generation = 5835 * 60
    # V_max = 1.2 * (10 ** 6) * 60
    # K = 7.2 * (10 ** 17) * 60
    # h_degrade = 82.6 * 60
    h_generation = 5835
    V_max = 1.2 * (10 ** 6)
    K = 7.2 * (10 ** 17)
    h_degrade = 82.6

    prod_h = h_generation + stoich * x[7]
    deg_h = h_degrade * x[8] + (x[8] * V_max) / (x[8] + K)

    return [prod_m1, deg_m1, prod_p1, deg_p1, prod_m2, deg_m2, prod_p2, deg_p2,
            prod_m3, deg_m3, prod_p3, deg_p3, prod_m4, deg_m4, prod_p4, deg_p4,
            prod_h, deg_h]




def exp4a_wrapper_stoch(plt):
    """
    Performs a run including gfp.
    Uses rvf_gfp
    :return: Updated plt
    """
    # reactions: prod_m1, deg_m1, prod_P1, deg_P1, prod_m2, deg_m2, prod_P2, deg_P2, prod_m3, deg_m3, prod_P3, deg_P3, prod_m4, deg_m4, prod_P4, deg_P4, prod_h2o2, deg_h2o2
    stoich_mat = [
        [1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 1, -1, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, -1, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, -1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, -1, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, -1]]

    x0_g = np.zeros((9, 1))
    x0_g[0, 0] = 10

    # *** NOTE: H should be set to 2, but is set to 1 by default!! ***
    # *** NOTE: H should be set to 2, but is set to 1 by default!! ***
    rt, rx, peaks, autoc = main.single_pass(0.277, 380, 2, 10000000, K=7, stoich_mat=stoich_mat, rvf=rvf_exp4a, x0_g=x0_g, p=[0.1])

    # - Regular Graph -
    p1_r = rx[1, :]
    p2_r = rx[3, :]
    p3_r = rx[5, :]
    p4_r = rx[7, :]
    p5_r = rx[8, :]

    plt.plot(rt, p5_r, label="P5", color="#c1121f")
    plt.plot(rt, p1_r, label="P1", color="#f94144")
    plt.plot(rt, p2_r, label="P2", color="#f9c74f")
    plt.plot(rt, p3_r, label="P3", color="#577590")
    plt.plot(rt, p4_r, label="P4", color="#06d6a0")


    return plt



# ---------------------------------------------------------------------------------------------------------------------
# Idea of Experiment 4B:
# Most vulnerable to a feedback loop
# Use Hill Equation to increase concentration of m2

def rvf_exp4b(x, p):
    """
    Calculates the rates of rxns using current quantities of each species
    :param x: [m1, P1, m2, P2, m3, P3, m4, P4, h] (array of molecule numbers)
    :param p: [lambda_m, lambda_p, beta_m, beta_p, h, K, lambda_h] Parameter vector for ODE solver
    :return:
    """
    # lambda_h is the parameter we should change!
    # lmabda_h = 1000 unstable, = 100 stable
    lambda_m = p[0]
    lambda_p = p[1]
    beta_m = p[2]
    beta_p = p[3]
    h = p[4]
    K = p[5]

    lambda_h = p[6]


    prod_m1 = (lambda_m * K**h) / (K**h + x[5] ** h)
    deg_m1 = (beta_m + beta_p) * x[0]

    prod_p1 = lambda_p * x[0]
    deg_p1 = beta_p * x[1]

    prod_m2 = (lambda_m * K ** h) / (K ** h + x[1] ** h) + (lambda_h * K ** h) / (K ** h + x[8] ** h)
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

    # * 60 b/c we're in molecules / sec and we need to be in molecules/min
    # stoich is number of hydrogen peroxide generated per GFP.
    stoich = 19
    # h_generation = 5835 * 60
    # V_max = 1.2 * (10 ** 6) * 60
    # K = 7.2 * (10 ** 17) * 60
    # h_degrade = 82.6 * 60
    h_generation = 5835
    V_max = 1.2 * (10 ** 6)
    K = 7.2 * (10 ** 17)
    h_degrade = 82.6

    prod_h = h_generation + stoich * x[7]
    deg_h = h_degrade * x[8] + (x[8] * V_max) / (x[8] + K)

    return [prod_m1, deg_m1, prod_p1, deg_p1, prod_m2, deg_m2, prod_p2, deg_p2,
            prod_m3, deg_m3, prod_p3, deg_p3, prod_m4, deg_m4, prod_p4, deg_p4,
            prod_h, deg_h]




def exp4b_wrapper_stoch(plt):
    """
    Performs a run including gfp.
    Uses rvf_gfp
    :return: Updated plt
    """
    # reactions: prod_m1, deg_m1, prod_P1, deg_P1, prod_m2, deg_m2, prod_P2, deg_P2, prod_m3, deg_m3, prod_P3, deg_P3, prod_m4, deg_m4, prod_P4, deg_P4, prod_h2o2, deg_h2o2
    stoich_mat = [
        [1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 1, -1, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, -1, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, -1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, -1, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, -1]]

    x0_g = np.zeros((9, 1))
    x0_g[0, 0] = 10

    # *** NOTE: H should be set to 2, but is set to 1 by default!! ***
    # *** NOTE: H should be set to 2, but is set to 1 by default!! ***
    rt, rx, peaks, autoc = main.single_pass(0.277, 380, 2, 1000000, K=7, stoich_mat=stoich_mat, rvf=rvf_exp4b, x0_g=x0_g, p=[1000])

    # - Regular Graph -
    p1_r = rx[1, :]
    p2_r = rx[3, :]
    p3_r = rx[5, :]
    p4_r = rx[7, :]
    p5_r = rx[8, :]

    plt.plot(rt, p5_r, label="P5", color="#c1121f")
    plt.plot(rt, p1_r, label="P1", color="#f94144")
    plt.plot(rt, p2_r, label="P2", color="#f9c74f")
    plt.plot(rt, p3_r, label="P3", color="#577590")
    plt.plot(rt, p4_r, label="P4", color="#06d6a0")


    return plt



# ---------------------------------------------------------------------------------------------------------------------


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



    plt.figure(figsize=(8, 8))
    # h2o2_wrapper_stoch(plt)
    # gfp_wrapper_stoch(plt)
    # exp1a_wrapper_stoch(plt)
    # exp1b_wrapper_stoch(plt)
    # exp2a_wrapper_stoch(plt)
    # exp2b_wrapper_stoch(plt)
    # exp3a_wrapper_stoch(plt)
    # exp3b_wrapper_stoch(plt)
    exp4a_wrapper_stoch(plt)
    # exp4b_wrapper_stoch(plt)

    # plt.plot(rt, p1_r, label="P1", color="#f94144")
    # plt.plot(rt, p2_r, label="P2", color="#f9c74f")
    # plt.plot(rt, p3_r, label="P3", color="#577590")
    # plt.plot(rt, p4_r, label="P4", color="#06d6a0")

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