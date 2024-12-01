import multi
import numpy as np
import traceback
import time

ERROR_DIRECTORY = 'errors.txt'




# ---------------------------------------------------------------------------------------------------------------------
# Idea of Experiment 1A:
# TetR increases in concentration in response to Oxidative stress.
# Use regular rate constant to increase concentration of m3

def rvf_exp1a(x, p):
    """
    Calculates the rates of rxns using current quantities of each species
    :param x: [m1, P1, m2, P2, m3, P3, m4, P4, h] (array of molecule numbers)
    :param p: [lambda_m, lambda_p, beta_m, beta_p, h, K, alpha, beta] Parameter vector for ODE solver
    :return:
    """
    # beta is the parameter we should change! Corresponds to production of mRNA-3
    # beta = 0.01 semi stable, 0.1 unstable
    lambda_m = p[0]
    lambda_p = p[1]
    beta_m = p[2]
    beta_p = p[3]
    h = p[4]
    K = p[5]
    alpha = p[6]
    beta = p[7]

    # Example parameter values:
    # known: [160.02406557883805, 60.00902459206427, 3.6101083032490973, 1, 2, 7, 10, 0.01]


    prod_m1 = (lambda_m * K**h) / (K**h + x[5] ** h)
    deg_m1 = (beta_m + beta_p) * x[0]

    prod_p1 = lambda_p * x[0]
    deg_p1 = beta_p * x[1]

    prod_m2 = (lambda_m * K ** h) / (K ** h + x[1] ** h)
    deg_m2 = (beta_m + beta_p) * x[2]

    prod_p2 = lambda_p * x[2]
    deg_p2 = beta_p * x[3]

    prod_m3 = (lambda_m * K ** h) / (K ** h + x[3] ** h) + beta * x[8]
    deg_m3 = (beta_m + beta_p) * x[4]

    prod_p3 = lambda_p * x[4]
    deg_p3 = beta_p * x[5]

    prod_m4 = (lambda_m * K ** h) / (K ** h + x[5] ** h)
    deg_m4 = (beta_m + beta_p) * x[6]

    prod_p4 = lambda_p * x[6]
    deg_p4 = beta_p * x[7]

    # * 60 b/c we're in molecules / sec and we need to be in molecules/min
    # stoich is number of hydrogen peroxide generated per GFP.
    stoich = alpha
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



# ---------------------------------------------------------------------------------------------------------------------
# Idea of Experiment 1B:
# TetR increases in concentration in response to Oxidative stress.
# Use Hill Equation to increase concentration of m3

def rvf_exp1b(x, p):
    """
    Calculates the rates of rxns using current quantities of each species
    :param x: [m1, P1, m2, P2, m3, P3, m4, P4, h] (array of molecule numbers)
    :param p: [lambda_m, lambda_p, beta_m, beta_p, h, K, alpha, beta] Parameter vector for ODE solver
    :return:
    """
    # beta is the parameter we should change! Corresponds to production of mRNA-3
    # beta = 1000 semi stable, 100 stable
    lambda_m = p[0]
    lambda_p = p[1]
    beta_m = p[2]
    beta_p = p[3]
    h = p[4]
    K = p[5]
    alpha = p[6]
    beta = p[7]


    prod_m1 = (lambda_m * K**h) / (K**h + x[5] ** h)
    deg_m1 = (beta_m + beta_p) * x[0]

    prod_p1 = lambda_p * x[0]
    deg_p1 = beta_p * x[1]

    prod_m2 = (lambda_m * K ** h) / (K ** h + x[1] ** h)
    deg_m2 = (beta_m + beta_p) * x[2]

    prod_p2 = lambda_p * x[2]
    deg_p2 = beta_p * x[3]

    prod_m3 = (lambda_m * K ** h) / (K ** h + x[3] ** h) + (beta * K ** h) / (K ** h + x[8] ** h)
    deg_m3 = (beta_m + beta_p) * x[4]

    prod_p3 = lambda_p * x[4]
    deg_p3 = beta_p * x[5]

    prod_m4 = (lambda_m * K ** h) / (K ** h + x[5] ** h)
    deg_m4 = (beta_m + beta_p) * x[6]

    prod_p4 = lambda_p * x[6]
    deg_p4 = beta_p * x[7]

    # * 60 b/c we're in molecules / sec and we need to be in molecules/min
    # stoich is number of hydrogen peroxide generated per GFP.
    stoich = alpha
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




# ---------------------------------------------------------------------------------------------------------------------
# Idea of Experiment 2A:
# E Coli decreases global translation under oxidative stress.
# Simulate this by increasing degradation of mRNA in response to H2O2.
# DOES NOT include degradation of GFP mRNA

def rvf_exp2a(x, p):
    """
    Calculates the rates of rxns using current quantities of each species
    :param x: [m1, P1, m2, P2, m3, P3, m4, P4, h] (array of molecule numbers)
    :param p: [lambda_m, lambda_p, beta_m, beta_p, h, K, alpha, beta] Parameter vector for ODE solver
    :return:
    """
    # beta is the parameter we should change! Corresponds to mRNA degradation rate
    # beta = 0.01 semi unstable, 0.1 unstable, 0.001 semi unstable, 0.0001 semi unstable
    lambda_m = p[0]
    lambda_p = p[1]
    beta_m = p[2]
    beta_p = p[3]
    h = p[4]
    K = p[5]

    alpha = p[6]
    beta = p[7]


    prod_m1 = (lambda_m * K**h) / (K**h + x[5] ** h)
    deg_m1 = (beta_m + beta_p) * x[0] + x[8] * beta

    prod_p1 = lambda_p * x[0]
    deg_p1 = beta_p * x[1]

    prod_m2 = (lambda_m * K ** h) / (K ** h + x[1] ** h)
    deg_m2 = (beta_m + beta_p) * x[2] + x[8] * beta

    prod_p2 = lambda_p * x[2]
    deg_p2 = beta_p * x[3]

    prod_m3 = (lambda_m * K ** h) / (K ** h + x[3] ** h) + p[6] * x[8]
    deg_m3 = (beta_m + beta_p) * x[4] + x[8] * beta

    prod_p3 = lambda_p * x[4]
    deg_p3 = beta_p * x[5]

    prod_m4 = (lambda_m * K ** h) / (K ** h + x[5] ** h)
    deg_m4 = (beta_m + beta_p) * x[6]

    prod_p4 = lambda_p * x[6]
    deg_p4 = beta_p * x[7]

    # * 60 b/c we're in molecules / sec and we need to be in molecules/min
    # stoich is number of hydrogen peroxide generated per GFP.
    stoich = alpha
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




# ---------------------------------------------------------------------------------------------------------------------
# Idea of Experiment 2B:
# E Coli decreases global translation under oxidative stress.
# Simulate this by increasing degradation of mRNA in response to H2O2.
# DOES include degradation of GFP mRNA

def rvf_exp2b(x, p):
    """
    Calculates the rates of rxns using current quantities of each species
    :param x: [m1, P1, m2, P2, m3, P3, m4, P4, h] (array of molecule numbers)
    :param p: [lambda_m, lambda_p, beta_m, beta_p, h, K, alpha, beta] Parameter vector for ODE solver
    :return:
    """
    # beta is the parameter we should change! Corresponds to mRNA degradation rate
    # beta = 0.01 semi stable, 0.1 unstable, 0.001 mostly stable
    lambda_m = p[0]
    lambda_p = p[1]
    beta_m = p[2]
    beta_p = p[3]
    h = p[4]
    K = p[5]

    alpha = p[6]
    beta = p[7]


    prod_m1 = (lambda_m * K**h) / (K**h + x[5] ** h)
    deg_m1 = (beta_m + beta_p) * x[0] + x[8] * beta

    prod_p1 = lambda_p * x[0]
    deg_p1 = beta_p * x[1]

    prod_m2 = (lambda_m * K ** h) / (K ** h + x[1] ** h)
    deg_m2 = (beta_m + beta_p) * x[2] + x[8] * beta

    prod_p2 = lambda_p * x[2]
    deg_p2 = beta_p * x[3]

    prod_m3 = (lambda_m * K ** h) / (K ** h + x[3] ** h) + p[6] * x[8]
    deg_m3 = (beta_m + beta_p) * x[4] + x[8] * beta

    prod_p3 = lambda_p * x[4]
    deg_p3 = beta_p * x[5]

    prod_m4 = (lambda_m * K ** h) / (K ** h + x[5] ** h)
    deg_m4 = (beta_m + beta_p) * x[6] + x[8] * beta

    prod_p4 = lambda_p * x[6]
    deg_p4 = beta_p * x[7]

    # * 60 b/c we're in molecules / sec and we need to be in molecules/min
    # stoich is number of hydrogen peroxide generated per GFP.
    stoich = alpha
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



# ---------------------------------------------------------------------------------------------------------------------
# Idea of Experiment 3A:
# Lambda cI increases in response to Oxidative stress.
# Use regular rate constant to increase concentration of m1

def rvf_exp3a(x, p):
    """
    Calculates the rates of rxns using current quantities of each species
    :param x: [m1, P1, m2, P2, m3, P3, m4, P4, h] (array of molecule numbers)
    :param p: [lambda_m, lambda_p, beta_m, beta_p, h, K, alpha, beta] Parameter vector for ODE solver
    :return:
    """
    # beta is the parameter we should change! Corresponds to mRNA production in relation to H2O2
    # beta = 0.1 unstable, 0.001 pretty much stable
    lambda_m = p[0]
    lambda_p = p[1]
    beta_m = p[2]
    beta_p = p[3]
    h = p[4]
    K = p[5]

    alpha = p[6]
    beta = p[7]


    prod_m1 = (lambda_m * K**h) / (K**h + x[5] ** h) + beta * x[8]
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
    stoich = alpha
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




# ---------------------------------------------------------------------------------------------------------------------
# Idea of Experiment 3B:
# Lambda cI increases in concentration in response to Oxidative stress.
# Use Hill Equation to increase concentration of m1

def rvf_exp3b(x, p):
    """
    Calculates the rates of rxns using current quantities of each species
    :param x: [m1, P1, m2, P2, m3, P3, m4, P4, h] (array of molecule numbers)
    :param p: [lambda_m, lambda_p, beta_m, beta_p, h, K, alpha, beta] Parameter vector for ODE solver
    :return:
    """
    # beta is the parameter we should change! Corresponds to mRNA-1 transcription in response to H2O2
    # beta = 1000 unstable, = 100 stable
    lambda_m = p[0]
    lambda_p = p[1]
    beta_m = p[2]
    beta_p = p[3]
    h = p[4]
    K = p[5]

    alpha = p[6]
    beta = p[7]


    prod_m1 = (lambda_m * K**h) / (K**h + x[5] ** h) + (beta * K ** h) / (K ** h + x[8] ** h)
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
    stoich = alpha
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




# ---------------------------------------------------------------------------------------------------------------------
# Idea of Experiment 4A:
# Most vulnerable to a feedback loop
# Use regular rate constant to increase concentration of m2

def rvf_exp4a(x, p):
    """
    Calculates the rates of rxns using current quantities of each species
    :param x: [m1, P1, m2, P2, m3, P3, m4, P4, h] (array of molecule numbers)
    :param p: [lambda_m, lambda_p, beta_m, beta_p, h, K, alpha, beta] Parameter vector for ODE solver
    :return:
    """
    # beta is the parameter we should change! Corresponds to mRNA-2 transcript rate in response to H2O2
    # beta = 0.1 unstable, 0.001 pretty much stable
    lambda_m = p[0]
    lambda_p = p[1]
    beta_m = p[2]
    beta_p = p[3]
    h = p[4]
    K = p[5]

    alpha = p[6]
    beta = p[7]


    prod_m1 = (lambda_m * K**h) / (K**h + x[5] ** h)
    deg_m1 = (beta_m + beta_p) * x[0]

    prod_p1 = lambda_p * x[0]
    deg_p1 = beta_p * x[1]

    prod_m2 = (lambda_m * K ** h) / (K ** h + x[1] ** h) + beta * x[8]
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
    stoich = alpha
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



# ---------------------------------------------------------------------------------------------------------------------
# Idea of Experiment 4B:
# Most vulnerable to a feedback loop
# Use Hill Equation to increase concentration of m2

def rvf_exp4b(x, p):
    """
    Calculates the rates of rxns using current quantities of each species
    :param x: [m1, P1, m2, P2, m3, P3, m4, P4, h] (array of molecule numbers)
    :param p: [lambda_m, lambda_p, beta_m, beta_p, h, K, alpha, beta] Parameter vector for ODE solver
    :return:
    """
    # beta is the parameter we should change! Corresponds to mRNA-2 transcript rate in response to H2O2
    # beta = 1000 unstable, = 100 stable
    lambda_m = p[0]
    lambda_p = p[1]
    beta_m = p[2]
    beta_p = p[3]
    h = p[4]
    K = p[5]

    alpha = p[6]
    beta = p[7]


    prod_m1 = (lambda_m * K**h) / (K**h + x[5] ** h)
    deg_m1 = (beta_m + beta_p) * x[0]

    prod_p1 = lambda_p * x[0]
    deg_p1 = beta_p * x[1]

    prod_m2 = (lambda_m * K ** h) / (K ** h + x[1] ** h) + (beta * K ** h) / (K ** h + x[8] ** h)
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
    stoich = alpha
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



def run_experiment(exp_name=''):
    """
    Runs a given experiment!
    Possible experiment names:
    1a, 1b, 2a, 2b, 3a, 3b, 4a, 4b
    """
    f = open('experiments/trial_tracker.txt', 'r+')
    attempt_num = int(f.readline())
    f = open('experiments/trial_tracker.txt', 'r+')
    f.writelines(f'{attempt_num + 1}\n')
    f.close()

    if exp_name == '1a':
        file_name = 'experiments/attempt' + str(attempt_num) + '_1a'
        rvf = rvf_exp1a
        beta_range = (0.01, 0.1)

    elif exp_name == '1b':
        file_name = 'experiments/attempt' + str(attempt_num) + '_1b'
        rvf = rvf_exp1b
        beta_range = (100, 1000)


    elif exp_name == '2a':
        file_name = 'experiments/attempt' + str(attempt_num) + '_2a'
        rvf = rvf_exp2a
        beta_range = (0.001, 0.01)

    elif exp_name == '2b':
        file_name = 'experiments/attempt' + str(attempt_num) + '_2b'
        rvf = rvf_exp2b
        beta_range = (0.001, 0.01)


    elif exp_name == '3a':
        file_name = 'experiments/attempt' + str(attempt_num) + '_3a'
        rvf = rvf_exp3a
        beta_range = (0.001, 0.01)

    elif exp_name == '3b':
        file_name = 'experiments/attempt' + str(attempt_num) + '_3b'
        rvf = rvf_exp3b
        beta_range = (100, 1000)


    elif exp_name == '4a':
        file_name = 'experiments/attempt' + str(attempt_num) + '_4a'
        rvf = rvf_exp4a
        beta_range = (0.001, 0.01)

    elif exp_name == '4b':
        file_name = 'experiments/attempt' + str(attempt_num) + '_4b'
        rvf = rvf_exp4b
        beta_range = (100, 1000)

    else:
        raise Exception(f"Invalid experiment name: '{exp_name}'")

    print(f'experiment: {file_name}')

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

    print(f"file name: {file_name + '.h5'}")
    # p (stoich_mat, rvf, x0_g, alpha_p, beta_p) [alpha_c and beta_c are constant alpha and beta]
    p = (stoich_mat, rvf, x0_g, 380, 0.277)
    multi.grid_run(beta_range, p, file_name=file_name + '.h5')

    # *** NOTE: H should be set to 2, but is set to 1 by default!! ***
    # *** NOTE: H should be set to 2, but is set to 1 by default!! ***
    # rt, rx, peaks, autoc = main.single_pass(0.277, 380, 2, 1000000, K=7, stoich_mat=stoich_mat, rvf=rvf_exp1a,
    #                                         x0_g=x0_g, p=[0.1])


if __name__ == '__main__':
    # run_experiment('1b')

    for name in ['1a', '1b', '2a', '2b', '3a', '3b', '4a', '4b']:
        try:
            e = open(ERROR_DIRECTORY, 'a')
            first_line = '*' * 300 + '\n'
            e.write(first_line)
            e.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - trial: {name}\n")
            e.write(f'\n')
            e.write(f'\n')
            e.close()
            run_experiment(name)
        except Exception as error:
            print(f'uh oh! error with entry {name}')
            e = open(ERROR_DIRECTORY, 'a')
            e.write(f'ERROR ERROR ERROR ERROR ERROR ERROR ERROR ERROR \n')
            e.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')}")
            e.write(f'{traceback.format_exc()}')
            e.write(f'\n')
            e.close()
            traceback.print_exc()
            e.close()