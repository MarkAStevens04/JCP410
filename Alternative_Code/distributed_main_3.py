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
import time
import traceback
# pip install h5py

ERROR_DIRECTORY = 'errors.txt'

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



def thread_run(trial_num, seed, queue, alpha, beta, p, dset_queue=None):
    """
    Runs a single thread of parameters.
    :param p: Holds all experiment related parameters. (stoich_mat, rvf, x0_g, alpha_c, beta_c) [alpha_c and beta_c are constant alpha and beta]
    :param alpha: Our first varying parameter. Corresponds to H2O2 stoichiometry in GFP
    :param beta: Our experimental related parameter.
    """
    stoich_mat = p[0]
    rvf = p[1]
    x0_g = p[2]
    alpha_c = p[3]
    beta_c = p[4]

    np.random.seed(seed)
    # print(f'would run with parameters: {p} and {alpha},{beta}.')
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
        if dset_queue:
            dset_queue.put((f'trial_{trial_num}_concentrations', rx), block=False)
            dset_queue.put((f'trial_{trial_num}_times', rt), block=False)
    if trial_num % 100 == 0:
        print(f'putting: {(period, precision)}')

    # queue.put((period, precision))
    queue.put([*save_data])







def grid_run(beta_range=(0.01, 1), p=None, file_name='experiments/experimental_results_long.h5'):
    """
    Runs a grid of parameters & attempts to solve
    :param beta_range: lower and upper values for betas.
    :param p: Holds all experiment related parameters. (stoich_mat, rvf, x0_g, alpha_p, beta_p) [alpha_c and beta_c are constant alpha and beta]
    :return:
    """
    n_bins = 10
    trials_per_bin = 3

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

    with h5py.File(file_name, 'w') as f:
        try:
            for b in betas:
                for a in alphas:
                    t += 1
                    group_name = f'param_{str(a).replace(".", "-")}_{str(b).replace(".", "-")}'
                    group = f.create_group(group_name)



                    queue = multiprocessing.Queue()
                    dset_queue = multiprocessing.Queue()
                    # dset_queue = None

                    procs = []
                    print(f'num reactants: {n_reactants}')

                    for trial in range(trials_per_bin):
                        s = int(np.random.random() * 10000000)
                        proc = Process(target=thread_run, args=(trial, s, queue, a, b, p, dset_queue))
                        procs.append(proc)
                        all_procs.append(proc)
                        proc.start()

                    data_findings = []
                    for r in range(n_reactants):
                        data_findings.append([])

                    # spread_99 = []
                    # spred_95 = []
                    # spread_68 = []


                    for proc in procs:
                        ret = queue.get()
                        print(f'ret: {ret}')

                        if not dset_queue.empty():
                            # Do it twice because we added two!
                            dset_ret = dset_queue.get()
                            group.create_dataset(dset_ret[0], data=dset_ret[1])

                            dset_ret = dset_queue.get()
                            group.create_dataset(dset_ret[0], data=dset_ret[1])

                        for d in range(n_reactants):
                            data_findings[d].append(ret[d])


                        # periods.append(ret[0])
                        # precisions.append(ret[1])


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
                    percent = t / total_params
                    print(f'Keep going! {round(percent * 100, 2)}% there')
                    print()
                    print()


                    # group.create_dataset('period', data=periods)
                    # group.create_dataset('precision', data=precisions)

                    for d in range(n_reactants):
                        group.create_dataset(f'reactant_{d}', data=data_findings[d])

                    while not queue.empty():

                        print(f'ope running in queue.')
                        ret = queue.get()

                    while not dset_queue.empty():
                        print(f'ope running in dset queue')
                        ret = dset_ret.get()





        except:
            e = open(ERROR_DIRECTORY, 'a')
            e.write(f'---------------------------------------------------------------------------------------------------- \n')
            e.write(f'\n')
            e.write(f'ERROR ERROR ERROR ERROR ERROR ERROR ERROR ERROR \n')
            e.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            e.write(f'alpha: {a}, beta: {b}\n')

            e.write(f'\n')
            e.write(f'{traceback.format_exc()}')
            e.write(f'\n')

            e.write(f'---------------------------------------------------------------------------------------------------- \n')
            e.write(f'\n')

            e.write(f'\n')
            print(f'error! with beta {b}, alpha {a}')

            e.close()
            traceback.print_exc()



    end = time.time()
    print(f'took {end - start} seconds!')





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
