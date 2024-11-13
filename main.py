import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
from scipy.integrate import odeint
from scipy.integrate import solve_ivp
from scipy.signal import find_peaks
import math
# pip install numpy
# pip install matplotlib
# python -m pip install scipy

# following tutorial
# https://www.youtube.com/watch?v=MM3cBamj1Ms
#
# Book if you need help
# https://pythonnumericalmethods.studentorg.berkeley.edu/notebooks/chapter22.06-Python-ODE-Solvers.html



def dYdt(t, Y):
    m1 = (p[0] * (p[5] ** p[4])) / (p[5] ** p[4] + Y[5] ** p[4]) - Y[0] * (p[2] + p[3])
    p1 = Y[0] * p[1] - Y[1] * p[3]
    m2 = (p[0] * (p[5] ** p[4])) / (p[5] ** p[4] + Y[1] ** p[4]) - Y[2] * (p[2] + p[3])
    p2 = Y[2] * p[1] - Y[3] * p[3]
    m3 = (p[0] * (p[5] ** p[4])) / (p[5] ** p[4] + Y[3] ** p[4]) - Y[4] * (p[2] + p[3])
    p3 = Y[4] * p[1] - Y[5] * p[3]
    return [m1, p1, m2, p2, m3, p3]





# reactions: prod_m1, deg_m1, prod_P1, deg_P1, prod_m2, deg_m2, prod_P2, deg_P2, prod_m3, deg_m3,prod_P3, deg_P3
stoich_mat=[
    [1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 1, -1, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, -1, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 1, -1, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 1, -1, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, -1]]

# Rows: m1, pf1, m2, pf2, m3, pf3
# Shape: (6 x 12) 6 species, 12 reactions.
stoich_mat = np.array(stoich_mat)
n = 1000000




def calc_rate(x, p):
    """
    Calculates the rates of rxns using current quantities of each species
    :param x: [m1, P1, m2, P2, m3, P3] (array of molecule numbers)
    :param p: [lambda_m, lambda_p, beta_m, beta_p, h, K] Parameter vector for ODE solver
    :return:
    """
    # This is where you put in your RVF!
    # return a 1x12 matrix
    # print(f'running rvf {x}')
    # print(f'x at 0: {x[0]}')
    # POSSIBLE ERROR:
    # deg_m might be (beta_m * x[]) NOT (beta_m + beta_p) * x[]
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
    return [prod_m1, deg_m1, prod_p1, deg_p1, prod_m2, deg_m2, prod_p2, deg_p2, prod_m3, deg_m3, prod_p3, deg_p3]





def full_gillespie(rvf, stoich_mat, p, n):
    """

    :param x0: Column vector of initial conditions
    :param rvf: a rate-value function. Should be a function which takes array as input.
    :param stoich_mat: Matrix storing changes to each reaction
    :param n: Number of iterations
    :return:
    """
    x0 = np.zeros((6, 1))
    x0[0, 0] = 10
    # x0 = [10, 0, 0, 0, 0, 0]  # [m1, P1, m2, P2, m3, P3]
    # print(f'x0: {x0}')


    # print(f'running full gillespie')

    # Setup matrix to store our output data.
    x = np.hstack([x0, np.zeros((x0.size, n))])
    # Matrix to store times
    t = np.zeros((1, n+1))
    # Matrix to store time intervals
    tau = np.zeros((1, n))

    for i in range(n):
        # print(i)
        lamd = rvf(x[:, i], p)
        lamd_tot = np.sum(lamd)
        r_t = np.random.random()
        # Calculate time to next interval
        T = -1 * math.log(r_t) / lamd_tot
        # print(f'T: {T}')

        # normalize lamd btwn 0 and 1
        lamd = lamd/lamd_tot
        # make lamd a cumulative sum ([0.7, 0.1, 0.2] -> [0.7, 0.8, 1.0])
        # Find which reaction rate has been picked!
        lamd = np.cumsum(lamd)
        # print(f'lamd: {lamd}')
        r = np.random.random()
        # print(f'r: {r}')
        I = 0
        while lamd[I] < r:
            I += 1

        # Update our time!
        t[0, i+1] = t[0, i] + T
        # Update our stoich.
        # print(f'stoich mat: {stoich_mat}')
        # print(f'selected: {stoich_mat[:, I]}')
        # print(f'pre: {x}')
        x[:, i+ 1] = x[:, i] + stoich_mat[:, I]
        tau[0, i] = T
        # print(f'post: {x}')
    return t, x, tau




def resample(t, x, t_mean):
    """
    We "resample" our data so that performing further analysis is easier.
    Our timesteps (tau) are not constant, but we need constant time intervals
    to calculate the autocorrelation. So, we will resample with constant timesteps!

    :param t: Original time vector
    :param x: Original amounts of reactants
    :param t_mean: Mean time step size
    :return:
    """
    # Make new time vector!
    # Expressions are equivalent, however second provides more stability with np.arange
    # rt = np.arange(0, t[0, -1], t_mean)
    rt = np.arange(0, math.floor(t[-1] / t_mean)) * t_mean
    rx = np.zeros((x.shape[0], rt.shape[0]))
    # initialize with x initial conditions
    rx[:, 0] = x[:, 0]

    I = 0
    for i in range(1, rt.shape[0]):
        # Essentially, find the index with the closest value to our desired.
        while t[I] < rt[i]:
            I += 1
        rx[:, i] = x[:, I-1]

    return rt, rx



# def autocorrelate(rt, rx):
#     """
#     Determines the autocorrelation of our concentrations and times.
#     :param rt:
#     :param rx:
#     :return:
#     """
#     # rx[0, :] = rx[0, :] - np.mean(rx[0, :])
#     # print(f'rx: {rx}')
#     result = np.correlate(rx[0, :], rx[0, :], mode='full')
#     return result[result.size//2:]


def autocorrelate(rt, rx):
    """
    Determines the autocorrelation of our concentrations and times.
    :param rt:
    :param rx:
    :return:
    """
    n_points = rx.shape[1]
    n_species = rx.shape[0]

    rx = rx - np.mean(rx, axis=1)[:, np.newaxis]
    ac = np.zeros(2 * n_points, dtype=complex)


    for i in range(n_species):
        # for every species,,,
        frx = np.fft.fft(rx[i, :], n=2 * n_points)
        power_spectrum = frx * np.conj(frx)

        ac += np.fft.ifft(power_spectrum)

    ac = ac / n_species
    ac = np.fft.fftshift(ac)
    divisor = np.concatenate(([1], np.arange(1, n_points), np.arange(n_points, 0, -1)))
    ac = ac / divisor / np.std(rx) ** 2
    ac = np.fft.fftshift(ac)

    ac_half = ac[:ac.size//2]
    ac_half = np.real(ac_half)
    return ac_half






def single_pass(Beta, alpha, Hill, num_iterations):
    """
    Performs a single pass of Gillespie Simulation!
    :param Beta:
    :param alpha:
    :return:
    """
    beta_p = 1  # protein elimination rate
    # Implicity math.log is ln
    # beta_m = 0.1 * (25 / math.log(2))
    beta_m = 1 / Beta
    h = 2  # Hill coefficient of cooperativity
    K = 7  # Repression threshold (when 1/2 of all repressors are bound)

    c = 4.8/1.8
    lambda_p = math.sqrt((alpha * beta_p * beta_m * K) / c)
    lambda_m = math.sqrt((alpha * beta_p * beta_m * K) * c)
    # print(f'lambda_p: {lambda_p}')
    # print(f'lambda_m: {lambda_m}')



    # lambda_m = 4.1 * (25 / math.log(2))  # max transcription rate constant
    # lambda_p = 1.8 * (25 / math.log(2))  # Translation rate constant

    p = [lambda_m, lambda_p, beta_m, beta_p, h, K]  # Parameter vector for ODE solver

    # alpha = lambda_p * lambda_m / (beta_m * beta_p * k)

    # reactions: prod_m1, deg_m1, prod_P1, deg_P1, prod_m2, deg_m2, prod_P2, deg_P2, prod_m3, deg_m3,prod_P3, deg_P3
    stoich_mat = [
        [1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, -1, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, -1, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, -1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 1, -1, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, -1]]

    # Rows: m1, pf1, m2, pf2, m3, pf3
    # Shape: (6 x 12) 6 species, 12 reactions.
    stoich_mat = np.array(stoich_mat)

    x0_g = np.zeros((6, 1))
    x0_g[0, 0] = 10


    # update calc_rate to take our constants as an input parameter

    t, x, tau = full_gillespie(calc_rate, stoich_mat, p, num_iterations)
    rt, rx = resample(t[0, :], x, tau.mean())

    autoc = autocorrelate(rt, rx)
    offset = 100
    [peaks, locs] = find_peaks(autoc[offset:])

    if len(peaks) == 0:
        # Unable to find any peaks
        print(f'no peaks :(')
        period = 0
        precision = 0
    else:
        # Found some peaks!
        # print(f'periods: {rt[peaks]}')
        period = rt[peaks[0]]
        precision = autoc[peaks[0]]


    return rt, rx, peaks, autoc




def deterministic(Beta, alpha, t):
    """
    Run one deterministic pass
    :return:
    """
    beta_p = 1  # protein elimination rate
    beta_m = 1 / Beta # mRNA elimination rate (25min protein half-life)
    h = 2  # Hill coefficient of cooperativity
    K = 7  # Repression threshold (when 1/2 of all repressors are bound)

    c = 4.8 / 1.8
    lambda_p = math.sqrt((alpha * beta_p * beta_m * K) / c) # Translation rate constant
    lambda_m = math.sqrt((alpha * beta_p * beta_m * K) * c) # max transcription rate constant

    global p
    p = [lambda_m, lambda_p, beta_m, beta_p, h, K]  # Parameter vector for ODE solver
    x0 = [10, 0, 0, 0, 0, 0]  # [m1, P1, m2, P2, m3, P3]
    sol = solve_ivp(dYdt, t_span=(0, max(t)), y0=x0, t_eval=t)
    return sol





if __name__ == "__main__":
    print(f'inside main')
    # setup what x will be!
    # x = np.linspace(0, 1, 100)
    # x0 = [10, 0, 0, 0, 0, 0]  # [m1, P1, m2, P2, m3, P3]
    t = np.linspace(0, 50, 1000)


    # ODE Int more oldschool, uses Fortran
    # solve_ivp more flexible

    # sol = odeint(dSdx, y0=S_0, t=x, tfirst=True)
    # [m1, P1, m2, P2, m3, P3]

    # p1_sol = sol.y[1]
    # p2_sol = sol.y[3]
    # p3_sol = sol.y[5]
    # t = sol.t

    # first parameter is velocity
    # print(sol_m1.T[0])

    # print(sol)

    # plt.plot(t, p1_sol)
    # plt.plot(t, p2_sol)
    # plt.plot(t, p3_sol)
    # plt.ylabel('$v(t)$', fontsize=22)
    # plt.xlabel('$t$', fontsize=22)

    # plt.show()

    print(f'')
    print(f'down here')
    # --------------------- Parameters for Gillespie ------------------------
    beta_p = 1  # protein elimination rate
    beta_m = 0.1 * (25 / math.log(2))  # mRNA elimination rate (25min protein half-life)

    lambda_m = 4.1 * (25 / math.log(2))  # max transcription rate constant
    lambda_p = 1.8 * (25 / math.log(2))  # Translation rate constant

    h = 2  # Hill coefficient of cooperativity
    K = 7  # Repression threshold (when 1/2 of all repressors are bound)
    p = [lambda_m, lambda_p, beta_m, beta_p, h, K]  # Parameter vector for ODE solver
    # ---------------------------------------------------------------------


    t, x, tau = full_gillespie(calc_rate, stoich_mat, p, 100000)
    rt, rx = resample(t[0, :], x, tau.mean())

    autoc = autocorrelate(rt, rx)
    print(f'generated autoc')

    print(f'autoc shape: {autoc.shape}')
    print(f'auto correlation: {autoc}')
    offset = 100
    [peaks, locs] = find_peaks(autoc[offset:])
    print(f'return: {find_peaks(autoc[offset:])}')
    # [peaks, locs] = find_peaks(np.zeros(10))
    print(f'peaks: {peaks}')
    print(f'locs: {locs}')
    if len(peaks) == 0:
        # Unable to find any peaks
        print(f'no peaks :(')
        period = 0
        precision = 0
    else:
        # Found some peaks!
        print(f'peaks here')
        print(f'peaks[0]: {peaks[0]}')
        # print(f'periods: {rt[peaks]}')
        period = rt[peaks[0]]
        precision = autoc[peaks[0]]

    print(f'period: {period}')
    print(f'precision: {precision}')



    p1_g = x[1, :]
    # p2_g = x[3, :]
    # p3_g = x[5, :]

    # - Autocorrelation -
    p1_r = rx[1, :]
    t = t[0, :]
    # plt.plot(rt, autoc)



    # # print(f'p3_g: {p3_g}')
    # # print(f't: {t}')
    # plt.plot(t, p1_g)

    # plt.plot(t, p2_g)
    # plt.plot(t, p3_g)

    # plt.plot(rt, p1_r)
    # # Peaks will be slightly offset. Peaks is the period after 1 oscillation,
    # # it might not start at the correct position!
    # plt.scatter(rt[peaks], rx[1, peaks])

    # - Display concentrations of Gillespie -
    p1_r = rx[1, :]
    p2_r = rx[3, :]
    p3_r = rx[5, :]

    plt.plot(rt, p1_r)
    plt.plot(rt, p2_r)
    plt.plot(rt, p3_r)

    plt.show()

