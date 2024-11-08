import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
from scipy.integrate import odeint
from scipy.integrate import solve_ivp
from scipy.signal import find_peaks
import math
import main
import os
import datetime
import h5py
import multiprocessing
from multiprocessing import Process
# pip install h5py


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
    dir_name = f'Trials/Test_1/Trial {str(b).replace(".", "-")} {str(a).replace(".", "-")}'
    try:
        os.mkdir(dir_name)
        print(f"Directory '{dir_name}' created successfully.")
    except FileExistsError:
        print(f"Directory '{dir_name}' already exists.")

    main_tracker = f'{dir_name}/output.txt'
    m = open(main_tracker, 'a+')
    m.writelines(f'Starting trials: {datetime.datetime.now()} \n')
    return m, dir_name


def read_data():
    """
    open the h5py data!
    :return:
    """
    with h5py.File('experimental_results.h5', 'r') as f:
        g_names = [name for name in f if isinstance(f[name], h5py.Group)]
        group = f['param_1-0_1-0']
        period_data = group['period'][:]
        precision_data = group['precision'][:]
        concentration_data = group['trial_0_concentrations']

        print(f'period data: {period_data}')
        print(f'precision data: {precision_data}')
        print(f'names: {g_names}')
        print(f'avg period: {np.average(period_data)}')
        print(f'avg precision: {np.average(precision_data)}')



def thread_run(trial_num,alpha,beta,group,queue):
    """
    Runs a single thread of parameters
    """
    rt, rx, peaks, autoc = main.single_pass(beta, alpha, 1, 100000)
    group.create_dataset(f'trial_{trial_num}_concentrations', data=rx)
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
    queue.put((period,precision))
    print(f' - trial {trial_num} complete!')





def grid_run():
    """
    Runs a grid of parameters & attempts to solve
    :return:
    """
    betas = [10 ** (-1 * (i / 5)) for i in range(10 + 1)]
    alphas = [10 ** (i / 5) for i in range(10 + 1)]
    trials_per_bin = 1000

    with h5py.File('experimental_results.h5', 'w') as f:
        for b in betas:
            for a in alphas:
                group_name = f'param_{str(b).replace(".", "-")}_{str(a).replace(".", "-")}'
                group = f.create_group(group_name)

                periods = []
                precisions = []

                queue = multiprocessing.Queue()
                procs = []
                for trial in range(trials_per_bin):
                    proc = Process(target=thread_run, args=(trial,a,b,group,queue))
                    procs.append(proc)
                    proc.start()

                for p in procs:
                    ret = queue.get()
                    periods.append(ret[0])
                    precisions.append(ret[1])
                print(f'Param complete! {a}  {b}')

                group.create_dataset('period', data=periods)
                group.create_dataset('precision', data=precisions)






                # # Start by opening the directories
                # # m is our output.txt file writer!
                # m, dir_name = open_directories(a, b)
                # for n in range(trials_per_bin):
                #     rt, rx, peaks, autoc = main.single_pass(b, a, 1, 100000)
                #     if len(peaks) == 0:
                #         # Unable to find any peaks
                #         print(f'no peaks :(')
                #         period = 0
                #         precision = 0
                #     else:
                #         # Found some peaks!
                #         print(f'periods: {rt[peaks]}')
                #         period = rt[peaks[0]]
                #         precision = autoc[peaks[0]]
                #     m.writelines(f'{str(n).ljust(5)}: {period}, {precision} \n')
                #
                #
                #
                # m.close()
        #     break
        # break

    print(f'betas: {betas}')


if __name__ == "__main__":
    # rt, rx, peaks, autoc = main.single_pass(0.277, 380, 1, 100000)
    #
    # if len(peaks) == 0:
    #     # Unable to find any peaks
    #     print(f'no peaks :(')
    #     period = 0
    #     precision = 0
    # else:
    #     # Found some peaks!
    #     print(f'periods: {rt[peaks]}')
    #     period = rt[peaks[0]]
    #     precision = autoc[peaks[0]]
    #
    # print(f'period: {period}')
    # print(f'precision: {precision}')
    #
    # # - Regular Graph -
    # p1_r = rx[1, :]
    # p2_r = rx[3, :]
    # p3_r = rx[5, :]
    #
    # plt.plot(rt, p1_r)
    # plt.plot(rt, p2_r)
    # plt.plot(rt, p3_r)

    # - Autocorrelation graph -
    # plt.plot(rt, autoc)

    # plt.show()

    # read_data()
    grid_run()
