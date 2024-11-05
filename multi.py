import numpy as np
import matplotlib.pyplot as plt
print(f'matplotlib and np imported')
import scipy as sp
from scipy.integrate import odeint
from scipy.integrate import solve_ivp
from scipy.signal import find_peaks
print(f'scipy imported')
import math
print(f'imported')
import main
import os
import datetime


# Beta should go from 1 to 0.01
# Alpha goes from 1 to 100
import main

def open_directories(a, b):
    """
    Opens the directory and creates an output file
    :param alpha:
    :param beta:
    :return:
    """
    dir_name = f'Trials/Test_1/Trial {str(b).replace('.', '-')} {str(a).replace('.', '-')}'
    try:
        os.mkdir(dir_name)
        print(f"Directory '{dir_name}' created successfully.")
    except FileExistsError:
        print(f"Directory '{dir_name}' already exists.")

    main_tracker = f'{dir_name}/output.txt'
    m = open(main_tracker, 'a+')
    m.writelines(f'Starting trials: {datetime.datetime.now()} \n')
    return m, dir_name

def grid_run():
    """
    Runs a grid of parameters & attempts to solve
    :return:
    """
    betas = [10 ** (-1 * (i / 5)) for i in range(10 + 1)]
    alphas = [10 ** (i / 5) for i in range(10 + 1)]
    trials_per_bin = 1000
    for b in betas:
        for a in alphas:
            # Start by opening the directories
            # m is our output.txt file writer!
            m, dir_name = open_directories(a, b)
            for n in range(trials_per_bin):
                rt, rx, peaks, autoc = main.single_pass(b, a, 1, 100000)
                if len(peaks) == 0:
                    # Unable to find any peaks
                    print(f'no peaks :(')
                    period = 0
                    precision = 0
                else:
                    # Found some peaks!
                    print(f'periods: {rt[peaks]}')
                    period = rt[peaks[0]]
                    precision = autoc[peaks[0]]
                m.writelines(f'{str(n).ljust(5)}: {period}, {precision} \n')



            m.close()
        #     break
        # break

    print(f'betas: {betas}')


if __name__ == "__main__":
    rt, rx, peaks, autoc = main.single_pass(0.277, 380, 1, 100000)

    if len(peaks) == 0:
        # Unable to find any peaks
        print(f'no peaks :(')
        period = 0
        precision = 0
    else:
        # Found some peaks!
        print(f'periods: {rt[peaks]}')
        period = rt[peaks[0]]
        precision = autoc[peaks[0]]

    print(f'period: {period}')
    print(f'precision: {precision}')

    # - Regular Graph -
    p1_r = rx[1, :]
    p2_r = rx[3, :]
    p3_r = rx[5, :]

    plt.plot(rt, p1_r)
    plt.plot(rt, p2_r)
    plt.plot(rt, p3_r)

    # - Autocorrelation graph -
    # plt.plot(rt, autoc)

    plt.show()

    grid_run()
