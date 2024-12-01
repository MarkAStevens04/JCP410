import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
from scipy.integrate import odeint
from scipy.integrate import solve_ivp
from scipy.signal import find_peaks
from scipy.optimize import curve_fit
import math
import time
# pip install numpy
# pip install matplotlib
# python -m pip install scipy


# following tutorial
# https://www.youtube.com/watch?v=MM3cBamj1Ms
#
# Book if you need help
# https://pythonnumericalmethods.studentorg.berkeley.edu/notebooks/chapter22.06-Python-ODE-Solvers.html



def dYdt(t, Y, p):
    # p: (lambda_m, lambda_p, beta_m, beta_p, h, K)
    # [m1, P1, m2, P2, m3, P3]
    # print(p)
    m1 = (p[0] * (p[5] ** p[4])) / (p[5] ** p[4] + Y[5] ** p[4]) - Y[0] * (p[2] + p[3])
    p1 = Y[0] * p[1] - Y[1] * p[3]
    m2 = (p[0] * (p[5] ** p[4])) / (p[5] ** p[4] + Y[1] ** p[4]) - Y[2] * (p[2] + p[3])
    p2 = Y[2] * p[1] - Y[3] * p[3]
    m3 = (p[0] * (p[5] ** p[4])) / (p[5] ** p[4] + Y[3] ** p[4]) - Y[4] * (p[2] + p[3])
    p3 = Y[4] * p[1] - Y[5] * p[3]
    return [m1, p1, m2, p2, m3, p3]



# reactions: prod_m1, deg_m1, prod_P1, deg_P1, prod_m2, deg_m2, prod_P2, deg_P2, prod_m3, deg_m3,prod_P3, deg_P3
# stoich_mat=[
#     [1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#     [0, 0, 1, -1, 0, 0, 0, 0, 0, 0, 0, 0],
#     [0, 0, 0, 0, 1, -1, 0, 0, 0, 0, 0, 0],
#     [0, 0, 0, 0, 0, 0, 1, -1, 0, 0, 0, 0],
#     [0, 0, 0, 0, 0, 0, 0, 0, 1, -1, 0, 0],
#     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, -1]]

# Rows: m1, pf1, m2, pf2, m3, pf3
# Shape: (6 x 12) 6 species, 12 reactions.
# stoich_mat = np.array(stoich_mat)
# n = 1000000




def calc_rate(x, p):
    """
    Calculates the rates of rxns using current quantities of each species
    :param x: [m1, P1, m2, P2, m3, P3] (array of molecule numbers)
    :param p: [lambda_m, lambda_p, beta_m, beta_p, h, K] Parameter vector for ODE solver
    :return:
    """
    # This is where you put in your RVF!
    # return a 1x12 matrix

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



def full_gillespie(rvf, stoich_mat, p, n, x0=None):
    """
    Performs entire Gillespie simulation
    :param x0: Column vector of initial conditions
    :param rvf: a rate-value function. Should be a function which takes array as input.
    :param stoich_mat: Matrix storing changes to each reaction
    :param n: Number of iterations
    :param p: Parameters to feed to rvf
    :return: t, x, tau (Time steps, concentration values, and delta-t for each timestep)
    """
    if x0 is None:
        x0 = np.zeros((6, 1))
        x0[0, 0] = 10
        # x0 = [10, 0, 0, 0, 0, 0]  # [m1, P1, m2, P2, m3, P3]


    # Setup matrix to store our output data.
    x = np.hstack([x0, np.zeros((x0.size, n))])
    # Matrix to store times
    t = np.zeros((1, n+1))
    # Matrix to store time intervals
    tau = np.zeros((1, n))

    timer = np.zeros((1, n+1))
    # Total: 24459

    for i in range(n):
        # s = time.perf_counter_ns()
        lamd = rvf(x[:, i], p)
        lamd_tot = sum(lamd)
        # lamd_tot = np.sum(lamd)
        r_t = np.random.random()
        # Calculate time to next interval
        T = -1 * math.log(r_t) / lamd_tot

        # normalize lamd btwn 0 and 1
        lamd = lamd/lamd_tot
        # make lamd a cumulative sum ([0.7, 0.1, 0.2] -> [0.7, 0.8, 1.0])
        # Find which reaction rate has been picked!
        lamd = np.cumsum(lamd)
        r = np.random.random()
        I = 0
        while lamd[I] < r:
            I += 1

        # Update our time!
        t[0, i+1] = t[0, i] + T
        # Update our stoich.
        x[:, i+ 1] = x[:, i] + stoich_mat[:, I]
        tau[0, i] = T

        # e = time.perf_counter_ns()
        # print(f'time taken: {e - s}')
        # timer[0, i] = e - s
    # print(f'timer: {timer}')
    # print(f'average: {np.average(timer)}')
    # print(f'percentage of total time: {(np.average(timer) / 22000) * 100}')

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


def freq_amp_spec(rx, n_points):
    """
    Creates the frequency-amplitude spectrum
    :return:
    """
    frx = np.fft.fft(rx, n=2 * n_points)

    power_spectrum = frx * np.conj(frx)
    # needs to be squared to add power spectra together
    # Curves tend to fit better for non-squared version
    power_spectrum_half = np.abs(frx) ** 2
    return power_spectrum_half[:power_spectrum_half.size // 2 + 1]




def fourier_analysis(rt, rx, n_points):
    """
    Performs all fourier analysis
    :return:
    """
    # For optimizing, remove:
    # y_bell
    # plt stuff
    # prints

    power_spectrum_half = freq_amp_spec(rx, n_points)
    # power_spectrum_half = power_spectrum_half


    print(f'power_spectrum_half shape: {power_spectrum_half.shape}')
    print(f'rt shape: {rt.shape}')

    # rt_adj = rt[1000:power_spectrum_half.size // 2]
    rt_adj = rt
    # rx_adj = rx[i, 1000:]
    plt.figure(figsize=(6, 6))
    time_step = rt_adj[2] - rt_adj[1]

    freqs = np.fft.fftfreq(power_spectrum_half.size, time_step)

    print(f'freqs size: {freqs.shape}')
    idx = np.argsort(freqs)[freqs.size // 2 + 1:freqs.size // 2 + 100]
    # idx = np.argsort(freqs)[:]

    # avg = np.average(freqs[idx], weights=power_spectrum_half[idx])
    # var = np.average((freqs[idx] - avg) ** 2, weights=power_spectrum_half[idx])
    # var = var * sum(power_spectrum_half[idx]) / (sum(power_spectrum_half[idx]) - 1)
    # std = math.sqrt(var)
    std2 = np.sqrt(np.cov(freqs[idx], aweights=power_spectrum_half[idx], ddof=0))
    amp = max(power_spectrum_half[idx])
    mean2_index = np.argmax(power_spectrum_half[idx])
    mean_2 = freqs[mean2_index]

    y_bell = gaussian(freqs[idx], amp, mean_2, std2)
    popt, pcov = curve_fit(gaussian, xdata=freqs[idx], ydata=power_spectrum_half[idx], p0=[amp, mean_2, std2],
                           method='dogbox')
    y_bell_dog = gaussian(freqs[idx], *popt)

    pdf_freq = power_spectrum_half[idx] / sum(power_spectrum_half[idx])
    cdf_freq = np.cumsum(pdf_freq)
    indices_99 = np.where((cdf_freq >= 0.005) & (cdf_freq <= 0.995))[0]
    indices_95 = np.where((cdf_freq >= 0.025) & (cdf_freq <= 0.975))[0]
    indices_68 = np.where((cdf_freq >= 0.16) & (cdf_freq <= 0.84))[0]
    indices_50 = np.where((cdf_freq >= 0.25) & (cdf_freq <= 0.75))[0]
    min_index_99, max_index_99 = indices_99[0], indices_99[-1]
    min_index_95, max_index_95 = indices_95[0], indices_95[-1]
    min_index_68, max_index_68 = indices_68[0], indices_68[-1]
    min_index_50, max_index_50 = indices_50[0], indices_50[-1]

    spread_99 = freqs[max_index_99] - freqs[min_index_99]
    spread_95 = freqs[max_index_95] - freqs[min_index_95]
    spread_68 = freqs[max_index_68] - freqs[min_index_68]
    spread_50 = freqs[max_index_50] - freqs[min_index_50]
    # power = [power_spectrum, power_spectrum_half]
    power = power_spectrum_half

    findings = [spread_99, spread_95, spread_68, spread_50, popt, std2, amp, mean_2]

    plt.plot(freqs[idx], power_spectrum_half[idx])
    plt.plot(freqs[idx], y_bell)
    plt.plot(freqs[idx], y_bell_dog, label='dog')

    plt.legend(loc='upper left')

    plt.show()

    return power, findings






def gaussian(x, amplitude, mean, stddev):
    return amplitude * np.exp(-((x - mean)**2) / (2 * stddev**2))

def autocorrelate(rt, rx):
    """
    Determines the autocorrelation of our concentrations and times.
    MUST ALREADY BE RESAMPLED!!!
    :param rt:
    :param rx:
    :return:
    """
    n_points = rx.shape[1]
    n_species = rx.shape[0]

    rx = rx - np.mean(rx, axis=1)[:, np.newaxis]
    ac = np.zeros(2 * n_points, dtype=complex)
    ac_p = np.zeros(2 * n_points, dtype=complex)

    stored_data = []

    for i in range(n_species):
        # for every species,,,
        # Helpful stack overflow
        # https://stackoverflow.com/questions/15382076/plotting-power-spectrum-in-python
        # frx = np.fft.fft(rx[i, :], n=2 * n_points)



        # frx = np.fft.fft(rx[i, :], n=2 * n_points)
        #
        # power_spectrum = frx * np.conj(frx)
        # # power_spectrum_half = np.abs(np.real(frx[1:frx.size // 2])) ** 2
        # # power_spectrum_half = np.abs(frx[1000:frx.size // 2]) ** 2
        # power_spectrum_half = np.abs(frx) ** 2
        #
        # rt_adj = rt[1000:frx.size // 2]
        # # rx_adj = rx[i, 1000:]
        # plt.figure(figsize=(6, 6))
        # time_step = rt_adj[2] - rt_adj[1]
        #
        # freqs = np.fft.fftfreq(power_spectrum_half.size, time_step)
        # idx = np.argsort(freqs)[freqs.size // 2 + 1:freqs.size // 2 + 100]
        # avg = np.average(freqs[idx], weights=power_spectrum_half[idx])
        # var = np.average((freqs[idx] - avg) ** 2, weights=power_spectrum_half[idx])
        # var = var * sum(power_spectrum_half[idx]) / (sum(power_spectrum_half[idx]) - 1)
        # std = math.sqrt(var)
        # std2 = np.sqrt(np.cov(freqs[idx], aweights=power_spectrum_half[idx], ddof=0))
        # amp = max(power_spectrum_half[idx])
        # mean2_index = np.argmax(power_spectrum_half[idx])
        # mean_2 = freqs[mean2_index]
        #
        # y_bell = gaussian(freqs[idx], amp, mean_2, std)
        # popt, pcov = curve_fit(gaussian, xdata=freqs[idx], ydata=power_spectrum_half[idx], p0=[amp, mean_2, std],
        #                        method='dogbox')
        # y_bell_dog = gaussian(freqs[idx], *popt)
        #
        # pdf_freq = power_spectrum_half[idx] / sum(power_spectrum_half[idx])
        # cdf_freq = np.cumsum(pdf_freq)
        # indices_99 = np.where((cdf_freq >= 0.005) & (cdf_freq <= 0.995))[0]
        # indices_95 = np.where((cdf_freq >= 0.025) & (cdf_freq <= 0.975))[0]
        # indices_68 = np.where((cdf_freq >= 0.16) & (cdf_freq <= 0.84))[0]
        # indices_50 = np.where((cdf_freq >= 0.25) & (cdf_freq <= 0.75))[0]
        # min_index_99, max_index_99 = indices_99[0], indices_99[-1]
        # min_index_95, max_index_95 = indices_95[0], indices_95[-1]
        # min_index_68, max_index_68 = indices_68[0], indices_68[-1]
        # min_index_50, max_index_50 = indices_50[0], indices_50[-1]
        #
        # spread_99 = freqs[max_index_99] - freqs[min_index_99]
        # spread_95 = freqs[max_index_95] - freqs[min_index_95]
        # spread_68 = freqs[max_index_68] - freqs[min_index_68]
        # spread_50 = freqs[max_index_50] - freqs[min_index_50]

        # print(f'cum_sum cdf: {cdf_freq}')
        # print(f'spread_99: {spread_99}')
        # print(f'spread_95: {spread_95}')
        # print(f'spread_68: {spread_68}')
        # print(f'spread_50: {spread_50}')

        # # print(f'popt: {popt}')
        # # print(f'pcov: {pcov}')
        # print(f'amp: {amp}')
        # print(f'mean_2: {mean_2}')
        #
        # print(f'std2: {std2}')
        # print(f'std: {std}')
        # print(f'var: {var}')
        # print(f'avg: {avg}')
        # # idx = np.argsort(power_spectrum_half)
        # print(f'power spectrum shape: {power_spectrum_half}')
        # print(f'rt shape: {rt_adj.shape}')
        # # print(f'rx shape: {rx_adj.shape}')
        # print(f'time_step: {time_step}')
        # print(f'freqs: {freqs}')
        # print(power_spectrum_half[-10:])
        #
        # # plt.plot(rt_adj, power_spectrum_half)
        # print(freqs[idx])
        # print(power_spectrum_half[idx])
        # print(idx)
        # print(f'---')
        # plt.plot(freqs[idx], power_spectrum_half[idx])
        # plt.plot(freqs[idx], y_bell)
        # plt.plot(freqs[idx], y_bell_dog, label='dog')
        #
        # plt.legend(loc='upper left')
        #
        # # plt.plot(rt_adj, power_spectrum_half)
        #
        # plt.show()

        power, findings = fourier_analysis(rt, rx[i, :], n_points)

        # ac += np.fft.ifft(power_spectrum)
        # ac_p += power_spectrum_half
        # stored_data.append([spread_99, spread_95, spread_68, spread_50, popt, std2, amp, mean_2])

        # # ac += np.fft.ifft(power[0])
        # ac_p += power
        stored_data.append([*findings])

    print(f'stored_data: {stored_data}')
    # plt.plot(freqs[idx], power_spectrum_half[idx])
    power, findings = fourier_analysis(rt, rx[i, :], n_points)

    ac = ac / n_species
    ac = np.fft.fftshift(ac)
    divisor = np.concatenate(([1], np.arange(1, n_points), np.arange(n_points, 0, -1)))
    ac = ac / divisor / np.std(rx) ** 2
    ac = np.fft.fftshift(ac)

    ac_half = ac[:ac.size // 2]
    ac_half = np.real(ac_half)

    plt.plot(rt, ac_half)
    plt.show()

    return ac_half


def single_pass(Beta, alpha, h, num_iterations, K=None, rvf=calc_rate, stoich_mat=None, x0_g=None, p=None):
    """
    Performs a single pass of Gillespie Simulation!
    Calculates peaks and autocorrelation
    :param Beta: Degradation ratio (protein elimination rate / mRNA elimination rate)
    :param alpha: Creation / Elimination ratio
    :param Hill: Hill coefficient of cooperativity
    :param num_iterations: Number of iterations for our simulation to run
    :param rvf: Rate-Vector function (Gillespie rvf)
    :param stoich_mat: Stoichiometric matrix for rvf
    :param x0_g: initial conditions
    :return: rt, rx, peaks, autoc
    """
    beta_p = 1  # protein elimination rate
    # Implicity math.log is ln
    # beta_m = 0.1 * (25 / math.log(2))
    beta_m = 1 / Beta
    h = h  # Hill coefficient of cooperativity (default is 2)
    # *** NOTE: H should be set to 2, but is set to 1 by default!! ***
    # *** NOTE: H should be set to 2, but is set to 1 by default!! ***
    # *** NOTE: H should be set to 2, but is set to 1 by default!! ***
    if K is None:
        K = 7  # Repression threshold (when 1/2 of all repressors are bound)

    c = 4.8/1.8
    lambda_p = math.sqrt((alpha * beta_p * beta_m * K) / c)
    lambda_m = math.sqrt((alpha * beta_p * beta_m * K) * c)


    # lambda_m = 4.1 * (25 / math.log(2))  # max transcription rate constant
    # lambda_p = 1.8 * (25 / math.log(2))  # Translation rate constant

    # Adds parameters to p if they are given
    if p is None:
        p = np.array([lambda_m, lambda_p, beta_m, beta_p, h, K, 10, 0.01])  # Parameter vector for ODE solver
        # p = [lambda_m, lambda_p, beta_m, beta_p, h, K, 10, 0.01]
    else:
        p = [lambda_m, lambda_p, beta_m, beta_p, h, K, *p]

    # alpha = lambda_p * lambda_m / (beta_m * beta_p * k)

    # Default values for stoich_mat if none given
    if not stoich_mat:
        # reactions: prod_m1, deg_m1, prod_P1, deg_P1, prod_m2, deg_m2, prod_P2, deg_P2, prod_m3, deg_m3, prod_P3, deg_P3
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

        # Rows: m1, pf1, m2, pf2, m3, pf3
        # Shape: (6 x 12) 6 species, 12 reactions.
    stoich_mat = np.array(stoich_mat)

    # Default values for x0_g if none given
    if x0_g is None:
        x0_g = np.zeros((9, 1))
        x0_g[0, 0] = 10


    # update calc_rate to take our constants as an input parameter

    s = time.time()
    print(f'starting gillespie')
    t, x, tau = full_gillespie(rvf, stoich_mat, p, num_iterations, x0_g)
    e = time.time()
    print(f'gillespie took {round(e - s, 2)}')
    rt, rx = resample(t[0, :], x, tau.mean())
    print(f'resample took {round(time.time() - e, 2)}')

    autoc = autocorrelate(rt, rx)
    offset = 100
    [peaks, locs] = find_peaks(autoc[offset:])

    return rt, rx, peaks, autoc




def deterministic(Beta, alpha, t, h=None, K=None, ode=dYdt, p=None, x0=None):
    """
    Run one deterministic pass
    :return:
    """
    beta_p = 1  # protein elimination rate
    beta_m = 1 / Beta # mRNA elimination rate (25min protein half-life)

    # Default values of h and k
    if h is None:
        h = 2  # Hill coefficient of cooperativity
    if K is None:
        K = 7  # Repression threshold (when 1/2 of all repressors are bound)

    c = 4.8 / 1.8
    lambda_p = math.sqrt((alpha * beta_p * beta_m * K) / c) # Translation rate constant
    lambda_m = math.sqrt((alpha * beta_p * beta_m * K) * c) # max transcription rate constant

    # Adds given parameters p to solver
    if not p:
        p = (lambda_m, lambda_p, beta_m, beta_p, h, K)  # Parameter vector for ODE solver
    else:
        p = (lambda_m, lambda_p, beta_m, beta_p, h, K, *p)

    # Default value of x0, otherwise can be specified
    if not x0:
        x0 = [10, 0, 0, 0, 0, 0]  # [m1, P1, m2, P2, m3, P3]

    # p must be passed as a tuple
    sol = solve_ivp(ode, t_span=(0, max(t)), y0=x0, t_eval=t, args=(p,))
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

