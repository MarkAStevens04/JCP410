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
from multiprocessing import Pool
import time
import traceback
import logging
from functools import partial

logger = logging.getLogger(__name__)

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
        logger.exception(f"Directory '{dir_name}' already exists.")
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
    with h5py.File('experimental_results_long_four.h5', 'r') as f:
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
        print(f'concentrations: {concentration_data}')



def thread_run_no_queue(seed, alpha, beta, p, trial_num):
    """
    Runs a single thread of parameters.
    :param p: Holds all experiment related parameters. (stoich_mat, rvf, x0_g, alpha_c, beta_c) [alpha_c and beta_c are constant alpha and beta]
    :param alpha: Our first varying parameter. Corresponds to H2O2 stoichiometry in GFP
    :param beta: Our experimental related parameter.
    """
    # print(f'trial num: {trial_num}')
    seed = seed * (trial_num + 1)
    # print(f'seed: {seed}')
    stoich_mat = p[0]
    rvf = p[1]
    x0_g = p[2]
    alpha_c = p[3]
    beta_c = p[4]

    np.random.seed(seed)
    # print(f'would run with parameters: {p} and {alpha},{beta}.')



    try:
        rt, rx, save_data = main.single_pass(beta_c, alpha_c, 2, 300000, K=7, stoich_mat=stoich_mat, rvf=rvf, x0_g=x0_g, p=[alpha, beta])
        # group.create_dataset(f'trial_{trial_num}_concentrations', data=rx)

        # if dset_queue:
        #     dset_queue.put((f'trial_{trial_num}_concentrations', rx))
        #     dset_queue.put((f'trial_{trial_num}_times', rt))


        # if len(peaks) == 0:
        #     # Unable to find any peaks
        #     print(f'no peaks :(')
        #     period = 0
        #     precision = 0
        # else:
        #     # Found some peaks!
        #     period = rt[peaks[0]]
        #     precision = autoc[peaks[0]]

        if trial_num % 10 == 0:
            print(f' - trial {trial_num} complete!')
            logging.info(f' - trial {trial_num} complete!')

        if trial_num % 100 == 0:
            # print(f'putting: {(period, precision)}')
            print(f'putting: {[*save_data]}')

        # queue.put((period, precision))
        # queue.put([])
        # queue.put([*save_data], timeout=300)
        return [*save_data]

    except KeyboardInterrupt as e:
        print(f'quitting...')
        raise Exception(KeyboardInterrupt) from e
    except:
        logger.warning(f'----------------------------------------------------------------------------------------------------')
        logger.exception(f'Error with SINGLE TRIAL! beta {beta}, alpha {alpha}, trial num: {trial_num}')
        logger.warning(f'----------------------------------------------------------------------------------------------------')
        print(f'error! with beta {beta}, alpha {alpha}')
        print(f'trial num: {trial_num}')

        traceback.print_exc()
        # queue.put([])







def grid_run(beta_range=(0.01, 1), p=None, file_name='experiments/experimental_results_long.h5'):
    """
    Runs a grid of parameters & attempts to solve
    :param beta_range: lower and upper values for betas.
    :param p: Holds all experiment related parameters. (stoich_mat, rvf, x0_g, alpha_p, beta_p) [alpha_c and beta_c are constant alpha and beta]
    :return:
    """
    n_bins = 10
    trials_per_bin = 100

    low = int(math.log10(beta_range[0]))
    high = int(math.log10(beta_range[1]))

    betas = np.logspace(low, high, n_bins + 1)
    # CAREFUL! MAKE SURE THERE'S NO DUPLICATES!
    alphas = np.logspace(0, 2, n_bins + 1).astype(dtype=int)
    alphas[0] = 0
    # alphas = np.int()

    print(f'alphas: {alphas}')
    print(f'betas:{betas}')

    total_params = len(betas) * len(alphas)
    t = 0
    start = time.time()

    if not p:
        raise Exception(f'Parameter p needs to be not None in grid_run')

    all_procs = []

    n_reactants = len(p[0]) + 1
    logger.info(f'Beginning grid run...')

    with h5py.File(file_name, 'w') as f:
            for b in betas:
                for a in alphas:
                    try:

                        t += 1
                        group_name = f'param_{str(a).replace(".", "-")}_{str(b).replace(".", "-")}'
                        group = f.create_group(group_name)



                        # queue = multiprocessing.Queue()
                        # dset_queue = multiprocessing.Queue()
                        dset_queue = None

                        procs = []

                        with Pool() as pool:
                            s = int(np.random.random() * 10000000)
                            r = pool.imap_unordered(partial(thread_run_no_queue, s, a, b, p), range(trials_per_bin))

                            # print(f'r: {r}')

                            # for trial in range(trials_per_bin):
                            #     s = int(np.random.random() * 10000000)
                            #     proc = Process(target=thread_run, args=(trial, s, queue, a, b, p, dset_queue))
                            #     procs.append(proc)
                            #     all_procs.append(proc)
                            #     proc.start()

                            data_findings = []
                            for k in range(n_reactants):
                                data_findings.append([])

                            for ret in r:
                                # print(f'ret: {ret}')

                                if dset_queue and not dset_queue.empty():
                                    # Do it twice because we added two!
                                    dset_ret = dset_queue.get()
                                    group.create_dataset(dset_ret[0], data=dset_ret[1])

                                    dset_ret = dset_queue.get()
                                    group.create_dataset(dset_ret[0], data=dset_ret[1])

                                # store trials in each reactant compartment!
                                for d in range(len(ret)):
                                    data_findings[d].append(ret[d])


                            if dset_queue:
                                while not dset_queue.empty():
                                    print(f'missed something...')
                                    dset_ret = dset_queue.get()
                                    group.create_dataset(dset_ret[0], data=dset_ret[1])


                        print(f'active children:')
                        print(f'{multiprocessing.active_children()}')
                        print(f'')


                        # print(f'all procs: {all_procs}')
                        print()
                        print()
                        print(f'************************ Param complete! {round(a, 2)}  {round(b, 2)} ************************')
                        logger.info(f'************************ Param complete! {round(a, 2)}  {round(b, 2)} ************************')
                        percent = t / total_params
                        print(f'Keep going! {round(percent * 100, 2)}% there')
                        logger.info(f'Keep going! {round(percent * 100, 2)}% there')

                        print()
                        print()

                        # Create a dataset for each reactant, add corresponding data
                        for d in range(n_reactants):
                            group.create_dataset(f'reactant_{d}', data=data_findings[d], compression='gzip')

                        # while not queue.empty():
                        #
                        #     print(f'ope running in queue.')
                        #     ret = queue.get()

                        if dset_queue:
                            while not dset_queue.empty():
                                print(f'ope running in dset queue')
                                ret = dset_ret.get()

                        # queue.close()
                        # if dset_queue:
                        #     dset_queue.close()




                    except KeyboardInterrupt as e:
                        print(f'quitting...')
                        raise Exception(KeyboardInterrupt) from e
                    except:
                        logger.warning(
                            f'----------------------------------------------------------------------------------------------------')
                        logger.exception(f'********** error with ENTIRE PARAMETER!!!! alpha {a}, beta {b} **********')
                        logger.warning(
                            f'----------------------------------------------------------------------------------------------------')
                        print(f'error! with alpha {a}, beta {b}')

                        traceback.print_exc()



    end = time.time()
    print(f'took {end - start} seconds!')


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

    # read_data
    grid_run()
