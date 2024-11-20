import h5py
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import h5pandas as h5pd
import main

EXP_DIRECTORY = 'Trials/Paper_Extension/attempt15_1b.h5'

def parse_params(name, reversed=True):
    """
    Parses the name of the param
    :param name:
    :param reversed: Whether to use old version (true) or updated (false)
    :return:
    """
    name = name.replace('-', '.')
    params = name.split('_')
    try:
        if reversed:
            # Previous version
            beta = float(params[1])
            alpha = float(params[2])
        else:
            # Updated version
            alpha = float(params[1])
            beta = float(params[2])
        return alpha, beta
    except:
        print(f'invalid name!')
        print(f'{name}.')
        return None, None


def read_grid():
    """
    open the h5py data!
    :return:
    """
    with h5py.File(EXP_DIRECTORY, 'r') as f:
        g_names = [name for name in f if isinstance(f[name], h5py.Group)]
        # group = f['param_1-0_1-0']
        # period_data = group['period'][:]
        # precision_data = group['precision'][:]
        # # concentration_data = group['trial_0_concentrations']
        #
        # print(f'period data: {period_data}')
        # print(f'precision data: {precision_data}')
        # print(f'names: {g_names}')
        # print(f'avg period: {np.average(period_data)}')
        # print(f'avg precision: {np.average(precision_data)}')
        # print(f'')
        #
        # print(f'')
        # parse the dataset
        all_alphas = set()
        all_betas = set()

        for name in g_names:
            alpha, beta = parse_params(name, reversed=False)
            all_alphas.add(alpha)
            all_betas.add(beta)

        all_alphas = [a for a in all_alphas]
        all_betas = [b for b in all_betas]

        all_alphas.sort()
        all_betas.sort()

        print(f'all alphas: {all_alphas}')
        print(f'all betas: {all_betas}')


        df = pd.DataFrame(np.random.randn(len(all_alphas) * len(all_betas), 4), columns=["Alpha", "Beta", "Period", "Precision"])
        print(df)
        for i, name in enumerate(g_names):
            alpha, beta = parse_params(name, reversed=False)
            group = f[name]
            period_data = group['period'][:]
            precision_data = group['precision'][:]
            avg_period = np.average(period_data)
            avg_precision = np.average(precision_data)

            df.iloc[i] = [alpha, beta, avg_period, avg_precision]

        print(df)

        glue = df.pivot(index="Beta", columns="Alpha", values="Precision")
        # glue = df.pivot(index="Beta", columns="Alpha", values="Period")

        # Put names of rows and columns in scientific notation
        if min(glue.columns) < 0.1:
            glue.columns = pd.Index([f"{x:.4f}" for x in glue.columns])
        else:
            glue.columns = pd.Index([f"{x:.1f}" for x in glue.columns])

        if min(glue.index) < 0.1:
            glue.index = pd.Index([f"{x:.4f}" for x in glue.index])
        else:
            glue.index = pd.Index([f"{x:.1f}" for x in glue.index])

        # sns.set(font_scale=1.5, rc={'text.usetex': True})
        plt.figure(figsize=(10, 10))
        g = sns.heatmap(glue, annot=True, fmt=".2f")
        # g2 = sns.heatmap(glue, annot=True, cmap="crest_r")

        # sns.set_context("notebook", font_scale=2)
        plt.xlabel("$\\alpha$", fontsize=20)
        plt.ylabel("$\\beta$", fontsize=20)
        # plt.title("Mean Period from Variations in $\\alpha$ and $\\beta$", fontsize=20)
        plt.title("Mean Precision from Variations in $\\alpha$ and $\\beta$", fontsize=20)

        # Add padding
        plt.subplots_adjust(bottom=0.2, left=0.2)

        # glue_2 = df.pivot(index="Beta", columns="Alpha", values="Period")
        # g2 = sns.heatmap(glue_2, annot=True, cmap="crest")

        plt.show()


def plot_stochastic():
    """
    Create time trace graph from algorithm run.
    :return:
    """
    rt, rx, peaks, autoc = main.single_pass(0.277, 380, 1, 100000)
    # - Regular Graph -
    p1_r = rx[1, :]
    p2_r = rx[3, :]
    p3_r = rx[5, :]

    plt.figure(figsize=(8, 8))

    plt.plot(rt, p1_r, label="P1", color="#f94144")
    plt.plot(rt, p2_r, label="P2", color="#f9c74f")
    plt.plot(rt, p3_r, label="P3", color="#577590")

    plt.xlabel("Time", fontsize=15)
    plt.ylabel("Copy Number", fontsize=15)
    plt.title("Stochastic", fontsize=15)

    plt.legend(loc="upper left")
    plt.subplots_adjust(bottom=0.1, left=0.1)

    # - Autocorrelation graph -
    # plt.plot(rt, autoc)

    plt.show()


def plot_deterministic():
    """
    Plots the deterministic model
    :return:
    """
    t = np.linspace(0, 50, 1000)
    sol = main.deterministic(0.277, 380, t)
    # [m1, P1, m2, P2, m3, P3]

    p1_sol = sol.y[1]
    p2_sol = sol.y[3]
    p3_sol = sol.y[5]
    t = sol.t

    plt.figure(figsize=(8, 8))

    plt.plot(t, p1_sol, label="P1", color="#f94144")
    plt.plot(t, p2_sol, label="P2", color="#f9c74f")
    plt.plot(t, p3_sol, label="P3", color="#577590")

    plt.xlabel("Time", fontsize=15)
    plt.ylabel("Copy Number", fontsize=15)
    plt.title("Deterministic", fontsize=15)

    plt.legend(loc="upper left")
    plt.subplots_adjust(bottom=0.1, left=0.1)

    plt.show()


# *********************************************************************************************************************

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

    # plt.plot(t, p1_sol, label="P1", color="#f94144")
    # plt.plot(t, p2_sol, label="P2", color="#f9c74f")
    # plt.plot(t, p3_sol, label="P3", color="#577590")
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

    rt, rx, peaks, autoc = main.single_pass(0.277, 380, 2, 100000, stoich_mat=stoich_mat, rvf=rvf_gfp, x0_g=x0_g, p=["cat"])


    # - Regular Graph -
    p1_r = rx[1, :]
    p2_r = rx[3, :]
    p3_r = rx[5, :]
    p4_r = rx[7, :]

    plt.figure(figsize=(8, 8))

    plt.plot(rt, p1_r, label="P1", color="#f94144")
    plt.plot(rt, p2_r, label="P2", color="#f9c74f")
    plt.plot(rt, p3_r, label="P3", color="#577590")
    plt.plot(rt, p4_r, label="P4", color="#06d6a0")

    plt.xlabel("Time", fontsize=15)
    plt.ylabel("Copy Number", fontsize=15)
    plt.title("GFP Stochastic", fontsize=15)

    plt.legend(loc="upper left")
    plt.subplots_adjust(bottom=0.1, left=0.1)

    # - Autocorrelation graph -
    # plt.plot(rt, autoc)

    plt.show()


def read_full_gfp_stoch():
    """
    Create time trace graph from algorithm run.
    :return:
    """
    # reactions: prod_m1, deg_m1, prod_P1, deg_P1, prod_m2, deg_m2, prod_P2, deg_P2, prod_m3, deg_m3, prod_P3, deg_P3, prod_m4, deg_m4, prod_P4, deg_P4
    # stoich_mat = [
    #     [1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #     [0, 0, 1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #     [0, 0, 0, 0, 1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #     [0, 0, 0, 0, 0, 0, 1, -1, 0, 0, 0, 0, 0, 0, 0, 0],
    #     [0, 0, 0, 0, 0, 0, 0, 0, 1, -1, 0, 0, 0, 0, 0, 0],
    #     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, -1, 0, 0, 0, 0],
    #     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, -1, 0, 0],
    #     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, -1]]
    #
    # x0_g = np.zeros((8, 1))
    # x0_g[0, 0] = 10
    #
    # rt, rx, peaks, autoc = main.single_pass(0.277, 380, 2, 100000, stoich_mat=stoich_mat, rvf=rvf_gfp, x0_g=x0_g, p=["cat"])
    #

    with h5py.File(EXP_DIRECTORY, 'r') as f:
        g_names = [name for name in f if isinstance(f[name], h5py.Group)]
        print(f'g_names: {g_names}')
        group = f['param_10-0_100-0']
        period_data = group['period'][:]
        precision_data = group['precision'][:]

        rx = np.array(group.get('trial_0_concentrations'))
        rt = np.array(group.get('trial_0_times'))

        print(f'period data: {period_data}')
        print(f'precision data: {precision_data}')
        print(f'names: {g_names}')
        print(f'avg period: {np.average(period_data)}')
        print(f'avg precision: {np.average(precision_data)}')
        print(f'concentrations: {rx}')
        print(f'')

        print(f'')
        # parse the dataset
        all_alphas = set()
        all_betas = set()

        for name in g_names:
            alpha, beta = parse_params(name, reversed=False)
            all_alphas.add(alpha)
            all_betas.add(beta)

        all_alphas = [a for a in all_alphas]
        all_betas = [b for b in all_betas]

        all_alphas.sort()
        all_betas.sort()

        print(f'all alphas: {all_alphas}')
        print(f'all betas: {all_betas}')

        print(f'rt: {rt.shape}')
        print(f'rx: {rx.shape}')

        # - Regular Graph -
        p1_r = rx[1, :]
        p2_r = rx[3, :]
        p3_r = rx[5, :]
        p4_r = rx[7, :]
        p5_r = rx[8, :]

        plt.figure(figsize=(8, 8))

        plt.plot(rt, p5_r, label="P5", color="#c1121f")
        plt.plot(rt, p1_r, label="P1", color="#f94144")
        plt.plot(rt, p2_r, label="P2", color="#f9c74f")
        plt.plot(rt, p3_r, label="P3", color="#577590")
        plt.plot(rt, p4_r, label="P4", color="#06d6a0")

        plt.xlabel("Time", fontsize=15)
        plt.ylabel("Copy Number", fontsize=15)
        plt.title("GFP Stochastic", fontsize=15)

        plt.legend(loc="upper left")
        plt.subplots_adjust(bottom=0.1, left=0.1)

        # - Autocorrelation graph -
        # plt.plot(rt, autoc)

        plt.show()








if __name__ == "__main__":
    # plot_stochastic()
    # plot_gfp_stoch()

    # read_grid()
    read_full_gfp_stoch()