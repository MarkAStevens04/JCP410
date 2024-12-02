import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
from scipy.integrate import odeint
from scipy.integrate import solve_ivp
from scipy.signal import find_peaks
from scipy.optimize import curve_fit
import math
import time
import traceback
import logging
import multiprocessing

logger = logging.getLogger(__name__)




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

    for i in range(n):
        lamd = rvf(x[:, i], p)
        lamd_tot = sum(lamd)
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



def freq_amp_spec(rx, n_points):
    """
    Creates the frequency-amplitude spectrum
    :return:
    """
    frx = np.fft.fft(rx, n=2 * n_points, axis=1)

    # power_spectrum = frx * np.conj(frx)
    # needs to be squared to add power spectra together
    # Curves tend to fit better for non-squared version
    power_spectrum_half = np.abs(frx) ** 2
    return power_spectrum_half[:, :power_spectrum_half.shape[1] // 2 + 1]




def fourier_analysis(power_spectrum_half, n_points, rt_adj):
    """
    Performs all fourier analysis
    :return:
    """
    # For optimizing, remove:
    # y_bell
    # plt stuff
    # prints
    # Parallelize fourier transformation


    # print(f'power_spectrum_half shape: {power_spectrum_half.shape}')
    # print(f'rt shape: {rt.shape}')


    # plt.figure(figsize=(6, 6))
    time_step = rt_adj[2] - rt_adj[1]

    freqs = np.fft.fftfreq(n_points, time_step)

    # print(f'freqs size: {freqs.shape}')
    # print(f'n points: {n_points}')
    idx = np.argsort(freqs)[n_points // 2 + 1:n_points // 2 + 1000]

    # avg = np.average(freqs[idx], weights=power_spectrum_half[idx])
    # var = np.average((freqs[idx] - avg) ** 2, weights=power_spectrum_half[idx])
    # var = var * sum(power_spectrum_half[idx]) / (sum(power_spectrum_half[idx]) - 1)
    # std = math.sqrt(var)
    try:
        std2 = np.sqrt(np.cov(freqs[idx], aweights=power_spectrum_half[idx], ddof=0))
    except KeyboardInterrupt as e:
        print(f'quitting...')
        raise Exception(KeyboardInterrupt) from e
    except:
        logger.warning(
            f'----------------------------------------------------------------------------------------------------')
        logger.exception(f'Error with calculating std! Will try to prevail...')
        logger.warning(
            f'----------------------------------------------------------------------------------------------------')

        traceback.print_exc()
        print(f'error with calculating std2! Will try to prevail...')
        std2 = 0

    amp = max(power_spectrum_half[idx])
    mean2_index = np.argmax(power_spectrum_half[idx])
    mean_2 = freqs[mean2_index]

    # y_bell = gaussian(freqs[idx], amp, mean_2, std2)

    try:
        popt, pcov = curve_fit(gaussian, xdata=freqs[idx], ydata=power_spectrum_half[idx], p0=[amp, mean_2, std2],
                               method='dogbox')
    except KeyboardInterrupt as e:
        print(f'quitting...')
        raise Exception(KeyboardInterrupt) from e
    except:

        logger.warning(
            f'----------------------------------------------------------------------------------------------------')
        logger.exception(f'Error with calculating popt! Will try to prevail...')
        logger.warning(
            f'----------------------------------------------------------------------------------------------------')

        traceback.print_exc()
        print(f'error with calculating popt! Will try to prevail...')
        popt = [0.0, 0.0, 0.0]
        # popt = (None, None, None)

    # y_bell_dog = gaussian(freqs[idx], *popt)

    pdf_freq = power_spectrum_half[idx] / sum(power_spectrum_half[idx])
    cdf_freq = np.cumsum(pdf_freq)
    indices_99 = np.where((cdf_freq >= 0.005) & (cdf_freq <= 0.995))[0]
    indices_95 = np.where((cdf_freq >= 0.025) & (cdf_freq <= 0.975))[0]
    indices_68 = np.where((cdf_freq >= 0.16) & (cdf_freq <= 0.84))[0]
    indices_50 = np.where((cdf_freq >= 0.25) & (cdf_freq <= 0.75))[0]
    if len(indices_99) != 2:
        logger.log(f'Unable to find two indices with 99 spread!')
        logger.log(f'cdf_freq: {cdf_freq}')
        indices_99 = [0, len(cdf_freq) - 1]
    if len(indices_95) != 2:
        logger.log(f'Unable to find two indices with 95 spread!')
        logger.log(f'cdf_freq: {cdf_freq}')
        indices_95 = [0, len(cdf_freq) - 1]
    if len(indices_68) != 2:
        logger.log(f'Unable to find two indices with 68 spread!')
        logger.log(f'cdf_freq: {cdf_freq}')
        indices_68 = [0, len(cdf_freq) - 1]
    if len(indices_50) != 2:
        logger.log(f'Unable to find two indices with 50 spread!')
        logger.log(f'cdf_freq: {cdf_freq}')
        indices_50 = [0, len(cdf_freq) - 1]

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

    findings = [spread_99, spread_95, spread_68, spread_50, *popt, std2, amp, mean_2]

    # plt.plot(freqs[idx], power_spectrum_half[idx])
    # plt.plot(freqs[idx], y_bell)
    # plt.plot(freqs[idx], y_bell_dog, label='dog')
    #
    # plt.legend(loc='upper left')
    #
    # plt.show()

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
    ac_p = np.zeros(n_points + 1, dtype=float)

    stored_data = []
    power_a = freq_amp_spec(rx, n_points)

    for i in range(n_species):
        # for every species,,,
        # Helpful stack overflow
        # https://stackoverflow.com/questions/15382076/plotting-power-spectrum-in-python
        # frx = np.fft.fft(rx[i, :], n=2 * n_points)

        power, findings = fourier_analysis(power_a[i, :], n_points, rt)

        ac_p += power
        stored_data.append([*findings])

    # print(f'stored_data: {stored_data}')
    power, findings = fourier_analysis(ac_p, n_points, rt)
    stored_data.append([*findings])
    # len(stored_data) should be 1 + n_species

    return stored_data



# def autocorrelate(rt, rx):
#     """
#     Determines the autocorrelation of our concentrations and times.
#     MUST ALREADY BE RESAMPLED!!!
#     :param rt:
#     :param rx:
#     :return:
#     """
#     n_points = rx.shape[1]
#     n_species = rx.shape[0]
#
#     rx = rx - np.mean(rx, axis=1)[:, np.newaxis]
#     ac = np.zeros(2 * n_points, dtype=complex)
#
#
#     for i in range(n_species):
#         # for every species,,,
#         frx = np.fft.fft(rx[i, :], n=2 * n_points)
#         power_spectrum = frx * np.conj(frx)
#
#         ac += np.fft.ifft(power_spectrum)
#
#     ac = ac / n_species
#     ac = np.fft.fftshift(ac)
#     divisor = np.concatenate(([1], np.arange(1, n_points), np.arange(n_points, 0, -1)))
#     ac = ac / divisor / np.std(rx) ** 2
#     ac = np.fft.fftshift(ac)
#
#     ac_half = ac[:ac.size//2]
#     ac_half = np.real(ac_half)
#     return ac_half






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
        p = [lambda_m, lambda_p, beta_m, beta_p, h, K]  # Parameter vector for ODE solver
    else:
        p = [lambda_m, lambda_p, beta_m, beta_p, h, K, *p]

    # alpha = lambda_p * lambda_m / (beta_m * beta_p * k)

    # Default values for stoich_mat if none given
    if not stoich_mat:
        # reactions: prod_m1, deg_m1, prod_P1, deg_P1, prod_m2, deg_m2, prod_P2, deg_P2, prod_m3, deg_m3, prod_P3, deg_P3
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

    # Default values for x0_g if none given
    if x0_g is None:
        x0_g = np.zeros((6, 1))
        x0_g[0, 0] = 10


    # update calc_rate to take our constants as an input parameter

    t, x, tau = full_gillespie(rvf, stoich_mat, p, num_iterations, x0_g)
    rt, rx = resample(t[0, :], x, tau.mean())

    save_data = autocorrelate(rt, rx)
    # offset = 100
    # [peaks, locs] = find_peaks(autoc[offset:])

    return rt, rx, save_data




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

