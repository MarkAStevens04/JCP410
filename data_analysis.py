import h5py
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import h5pandas as h5pd
import main

EXP_DIRECTORY = 'Trials/Replicate_Existing/experimental_results_long.h5'

def parse_params(name):
    """
    Parses the name of the param
    :param name:
    :return:
    """
    name = name.replace('-', '.')
    params = name.split('_')
    try:
        beta = float(params[1])
        alpha = float(params[2])
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
        group = f['param_1-0_1-0']
        period_data = group['period'][:]
        precision_data = group['precision'][:]
        # concentration_data = group['trial_0_concentrations']

        print(f'period data: {period_data}')
        print(f'precision data: {precision_data}')
        print(f'names: {g_names}')
        print(f'avg period: {np.average(period_data)}')
        print(f'avg precision: {np.average(precision_data)}')
        print(f'')

        print(f'')
        # parse the dataset
        all_alphas = set()
        all_betas = set()

        for name in g_names:
            alpha, beta = parse_params(name)
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
            alpha, beta = parse_params(name)
            group = f[name]
            period_data = group['period'][:]
            precision_data = group['precision'][:]
            avg_period = np.average(period_data)
            avg_precision = np.average(precision_data)

            df.iloc[i] = [alpha, beta, avg_period, avg_precision]

        print(df)

        # glue = df.pivot(index="Beta", columns="Alpha", values="Precision")
        glue = df.pivot(index="Beta", columns="Alpha", values="Period")

        # Put names of rows and columns in scientific notation
        glue.columns = pd.Index([f"{x:.1e}" for x in glue.columns])
        glue.index = pd.Index([f"{x:.1e}" for x in glue.index])

        # sns.set(font_scale=1.5, rc={'text.usetex': True})
        plt.figure(figsize=(10, 10))
        # g = sns.heatmap(glue, annot=True, fmt=".2f")
        g2 = sns.heatmap(glue, annot=True, cmap="crest_r")

        # sns.set_context("notebook", font_scale=2)
        plt.xlabel("$\\alpha$", fontsize=20)
        plt.ylabel("$\\beta$", fontsize=20)
        plt.title("Mean Period from Variations in $\\alpha$ and $\\beta$", fontsize=20)
        # plt.title("Mean Precision from Variations in $\\alpha$ and $\\beta$", fontsize=20)

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

if __name__ == "__main__":
    plot_gfp_det()