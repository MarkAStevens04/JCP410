import h5py
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import h5pandas as h5pd

EXP_DIRECTORY = 'Trials/Replicate_Existing/experimental_results.h5'

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


def read_data():
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

        all_entries = []
        all_alphas = set()
        all_betas = set()

        for name in g_names:
            alpha, beta = parse_params(name)
            all_alphas.add(alpha)
            all_betas.add(beta)

        all_alphas = [a for a in all_alphas]
        all_betas = [b for b in all_betas]
        # all_alphas = all_alphas.sort()
        # all_betas = all_betas.sort()
        all_alphas.sort()
        all_betas.sort()


        # all_alphas = np.array(all_alphas, dtype='str')
        # all_betas = np.array(all_betas, dtype='str')

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

            # sets the row precision at column (alpha, beta) to avg_precision
            # df.loc["precision", (alpha, beta)] = avg_precision
            df.iloc[i] = [alpha, beta, avg_period, avg_precision]
            # df.iloc[i] = [alpha, beta, avg_precision]




        # df_period = pd.DataFrame(index=all_alphas, columns=all_betas)
        # df_precision = pd.DataFrame(index=all_alphas, columns=all_betas)
        # for name in g_names:
        #     alpha, beta = parse_params(name)
        #     group = f[name]
        #     period_data = group['period'][:]
        #     precision_data = group['precision'][:]
        #     avg_period = np.average(period_data)
        #     avg_precision = np.average(precision_data)
        #
        #     # sets the row precision at column (alpha, beta) to avg_precision
        #     # df.loc["precision", (alpha, beta)] = avg_precision
        #     df_period.loc[str(alpha), str(beta)] = avg_period
        #     df_precision.loc[str(alpha), str(beta)] = avg_precision
        # print(df_period)




        # for name in g_names:
        #     alpha, beta = parse_params(name)
        #     all_alphas.add(alpha)
        #     all_betas.add(beta)
        # print(f'sizes: {len(all_alphas), len(all_betas)}')
        # df = pd.MultiIndex.from_product([all_alphas, all_betas], names=['alpha', 'beta'])
        # print(df)
        # print(f' -- pause -- ')
        # df = pd.DataFrame(np.random.randn(1, 121), index=["period"], columns=df)
        # print(df)
        # print(f'slay')
        # print(df['10.0'])
        # for name in g_names:
        #     alpha, beta = parse_params(name)
        #     group = f[name]
        #     period_data = group['period'][:]
        #     # precision_data = group['precision'][:]
        #     avg_period = np.average(period_data)
        #     avg_precision = np.average(precision_data)
        #
        #     # sets the row precision at column (alpha, beta) to avg_precision
        #     # df.loc["precision", (alpha, beta)] = avg_precision
        #     df.loc["period", (alpha, beta)] = avg_period
        #
        #
        #
        #
        #
        # # all_betas = [b for b in all_betas]
        # # all_betas.insert(0, "alpha")
        # # df = pd.DataFrame(columns=["Alpha", all_betas])
        # # print(df)
        # # print(f'slay')
        # # for name in g_names:
        # #     alpha, beta = parse_params(name)
        # #     group = f[name]
        # #     period_data = group['period'][:]
        # #     precision_data = group['precision'][:]
        # #
        # #     avg_period = np.average(period_data)
        # #     avg_precision = np.average(precision_data)
        # #     if beta not in df:
        # #         df[beta]
        # #     df[beta]["period"] = period_data
        # # print(df)
        #
        #
        #
        #
        #
        # # entry = (alpha, beta, avg_period, avg_precision)
        # # print(f'entry: {entry}')
        # # all_entries.append(entry)
        #
        # # df = pd.MultiIndex.from_tuples(all_entries, names=["alpha", "beta", "period", "precision"])
        # # print(df)
        # # print(df[np.float64(1.0)])
        # #     # print(f'{alpha}, {beta}')
        # #
        # #
        #
        # print(f'unstacking')
        # # df2 = df.reset_index().pivot(columns='alpha', index='beta', values='period')
        # # df2 = df.unstack()
        # df2 = df.pivot_table(values='alpha', index)
        # print(df2)
        # sns.set_theme(style="ticks")
        # # f, ax = plt.subplots(figsize=(9, 6))
        # print(f'--- df_period ---')
        # print(df_period)
        # # sns.heatmap(data=df_period)
        #
        # print(df_period.index)
        # print(df_period.columns)
        # print(f'-- pivoting --')
        # df_period = df_period.melt(
        #     df_period.reset_index(),
        #     id_vars='alpha',
        #     var_name='beta',
        #     value_vars=all_betas,
        #     value_name='period'
        # )
        #
        #
        print(df)
        # print(f'len: {}')

        # g = sns.JointGrid(data=df, x='Alpha', y='Beta', marginal_ticks=True)
        # g.ax_joint.set(yscale="log")
        # g.ax_joint.set(xscale="log")
        # # g.plot(sns.heatmap, sns.histplot)
        # #
        #
        #
        #
        #
        # # # Create an inset legend for the histogram colorbar
        # cax = g.figure.add_axes([.15, .55, .02, .2])
        # #
        # # # Add the joint and marginal histogram plots
        # g.plot_joint(
        #     sns.histplot, discrete=(False, False),
        #     cmap="light:#03012d", pmax=.8, cbar=True, cbar_ax=cax, bins=len(all_alphas), stat='percent'
        # )
        # g.plot_marginals(sns.histplot, element="step", color="#03012d")

        glue = df.pivot(index="Beta", columns="Alpha", values="Precision")
        sns.heatmap(glue, annot=True)

        plt.show()


if __name__ == "__main__":
    read_data()