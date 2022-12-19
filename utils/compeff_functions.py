import numpy as np
import pandas as pd
import seaborn as sns
import time
from utils.simple_ar_univariate import *
from utils.simple_ar_multivariate import *
from utils.adaptive_ar_univariate import *
from utils.mh_univariate import *
from utils.mh_multivariate import *

def computational_efficiency_univariate1(N, n_rep, target, target_name, proposal_simple, M, proposal_ind_MH, proposal_rw_MH,
                                                     log_prob, T_k, z_limits, x0_arms, x0_MH, burn_in, latex_table=False):

    time_array = np.zeros((n_rep, 4, len(N)))
    accept_array = np.zeros((n_rep, 4, len(N)))

    for j, n in enumerate(N):
        for i in range(n_rep):

            # Set time and sample
            t0 = time.time()
            samples, accept_prob = naive_simple_accept_reject_univariate(n, M, target, proposal_simple)
            t1 = time.time()
            t = t1 - t0
            time_array[i, 0, j] = t
            accept_array[i, 0, j] = accept_prob
            t0 = time.time()              
            samples, arms_accept_prob, arms_mh_accept_prob = naive_adaptive_rejection_metropolis_sampling(n, x0_arms, burn_in, log_prob, xs=T_k, z_limits=z_limits)
            t1 = time.time()
            t = t1 - t0
            time_array[i, 1, j] = t
            accept_array[i, 1, j] = arms_mh_accept_prob
            t0 = time.time()
            samples, ind_MH_accept_prob = naive_independent_metropolis_hastings_univariate(x0_MH, n, burn_in, target, proposal_ind_MH)
            t1 = time.time()
            t = t1 - t0
            time_array[i, 2, j] = t
            accept_array[i, 2, j] = ind_MH_accept_prob
            t0 = time.time()
            samples, rw_MH_accept_prob = naive_rw_metropolis_hastings_univariate(x0_MH, n, burn_in, target, proposal_rw_MH)
            t1 = time.time()
            t = t1 - t0
            time_array[i, 3, j] = t
            accept_array[i, 3, j] = rw_MH_accept_prob
    # Create Pandas Dataframe
    algorithm = ['sar', 'arms', 'ind_mh', 'rw_mh']
    dfs = []
    for j, n in enumerate(N):
        for i, alg in enumerate(algorithm):
            df = pd.DataFrame({'Time': time_array[:, i, j], 'Accept': accept_array[:, i, j]})
            df['algorithm'] = alg
            df['N'] = n
            df['N'] = df['N'].astype('category')
            dfs.append(df)

    final_df = pd.concat(dfs)

    ax = sns.catplot(x="algorithm", y="Time", hue="N", data=final_df, size=3)
    ax.set(xlabel='Algorithm', ylabel='Time in seconds');
    ax.fig.suptitle(f'{target_name}, {n_rep} repetitions')
    ax.tight_layout()

    mean_df = final_df.groupby(['algorithm', 'N']).mean()
    mean_df = mean_df.rename(columns={'Time': 'Avg. Time'})
    mean_df = mean_df.transpose()
    mean_df = mean_df[algorithm]

    if latex_table == True:
        print(mean_df.to_latex(index=True))


    return ax, mean_df

def computational_efficiency_univariate2(N, n_rep, target, target_name, proposal_simple, M, proposal_ind_MH, proposal_rw_MH,
                                                     log_prob, T_k_ars, T_k_fars, z_limits, x0_MH, burn_in, latex_table=False):

    time_array = np.zeros((n_rep, 5, len(N)))
    accept_array = np.zeros((n_rep, 5, len(N)))

    for j, n in enumerate(N):
        for i in range(n_rep):

            # Set time and sample
            t0 = time.time()
            samples, accept_prob = naive_simple_accept_reject_univariate(n, M, target, proposal_simple)
            t1 = time.time()
            t = t1 - t0
            time_array[i, 0, j] = t
            accept_array[i, 0, j] = accept_prob
            t0 = time.time()                          
            samples, ars_accept_prob, = naive_adaptive_rejection_sampling(n, log_prob, xs=T_k_ars, z_limits=z_limits)
            t1 = time.time()
            t = t1 - t0
            time_array[i, 1, j] = t
            accept_array[i, 1, j] = ars_accept_prob
            t0 = time.time()                          
            samples, ars_accept_prob, = naive_fixed_adaptive_rejection_sampling(n, log_prob, xs=T_k_fars, z_limits=z_limits)
            t1 = time.time()
            t = t1 - t0
            time_array[i, 2, j] = t
            accept_array[i, 2, j] = ars_accept_prob
            t0 = time.time()
            samples, ind_MH_accept_prob = naive_independent_metropolis_hastings_univariate(x0_MH, n, burn_in, target, proposal_ind_MH)
            t1 = time.time()
            t = t1 - t0
            time_array[i, 3, j] = t
            accept_array[i, 3, j] = ind_MH_accept_prob
            t0 = time.time()
            samples, rw_MH_accept_prob = naive_rw_metropolis_hastings_univariate(x0_MH, n, burn_in, target, proposal_rw_MH)
            t1 = time.time()
            t = t1 - t0
            time_array[i, 4, j] = t
            accept_array[i, 4, j] = rw_MH_accept_prob
    # Create Pandas Dataframe
    algorithm = ['sar', 'ars', 'fars', 'ind_mh', 'rw_mh']
    dfs = []
    for j, n in enumerate(N):
        for i, alg in enumerate(algorithm):
            df = pd.DataFrame({'Time': time_array[:, i, j], 'Accept': accept_array[:, i, j]})
            df['algorithm'] = alg
            df['N'] = n
            df['N'] = df['N'].astype('category')
            dfs.append(df)

    final_df = pd.concat(dfs)

    ax = sns.catplot(x="algorithm", y="Time", hue="N", data=final_df, size=3)
    ax.set(xlabel='Algorithm', ylabel='Time in seconds');
    ax.fig.suptitle(f'{target_name}, {n_rep} repetitions')
    ax.tight_layout()

    mean_df = final_df.groupby(['algorithm', 'N']).mean()
    mean_df = mean_df.rename(columns={'Time': 'Avg. Time'})
    mean_df = mean_df.transpose()
    mean_df = mean_df[algorithm]

    if latex_table == True:
        print(mean_df.to_latex(index=True))


    return ax, mean_df

def computational_efficiency_multivariate(N, n_rep, D, target, target_name, proposal_simple, M, proposal_ind_MH, proposal_rw_MH,
                                                    x0, burn_in, latex_table=False):
    if D == 2:
        k = 3
    else:
        k = 2
    time_array = np.zeros((n_rep, k, len(N)))
    accept_array = np.zeros((n_rep, k, len(N)))

    for j, n in enumerate(N):
        for i in range(n_rep):

            # Set time and sample
            if D == 2:
                t0 = time.time()
                samples, accept_prob = naive_simple_accept_reject_multivariate(n, M, D, target, proposal_simple)
                t1 = time.time()
                t = t1 - t0
                time_array[i, 0, j] = t
                accept_array[i, 0, j] = accept_prob
                t0 = time.time()
                samples, ind_MH_accept_prob = naive_independent_metropolis_hastings_multivariate(x0, n, D, burn_in, target, proposal_ind_MH)
                t1 = time.time()
                t = t1 - t0
                time_array[i, 1, j] = t
                accept_array[i, 1, j] = ind_MH_accept_prob
                t0 = time.time()
                samples, rw_MH_accept_prob = naive_rw_metropolis_hastings_multivariate(x0, n, D, burn_in, target, proposal_rw_MH)
                t1 = time.time()
                t = t1 - t0
                time_array[i, 2, j] = t
                accept_array[i, 2, j] = rw_MH_accept_prob
            else:
                t0 = time.time()
                samples, ind_MH_accept_prob = naive_independent_metropolis_hastings_multivariate(x0, n, D, burn_in, target, proposal_ind_MH)
                t1 = time.time()
                t = t1 - t0
                time_array[i, 0, j] = t
                accept_array[i, 0, j] = ind_MH_accept_prob
                t0 = time.time()
                samples, rw_MH_accept_prob = naive_rw_metropolis_hastings_multivariate(x0, n, D, burn_in, target, proposal_rw_MH)
                t1 = time.time()
                t = t1 - t0
                time_array[i, 1, j] = t
                accept_array[i, 1, j] = rw_MH_accept_prob
    # Create Pandas Dataframe
    if D == 2:
        algorithm = ['sar', 'ind_mh', 'rw_mh']
    else:
        algorithm = ['ind_mh', 'rw_mh']
    dfs = []
    for j, n in enumerate(N):
        for i, alg in enumerate(algorithm):
            df = pd.DataFrame({'Time': time_array[:, i, j], 'Accept': accept_array[:, i, j]})
            df['algorithm'] = alg
            df['N'] = n
            df['N'] = df['N'].astype('category')
            dfs.append(df)

    final_df = pd.concat(dfs)

    ax = sns.catplot(x="algorithm", y="Time", hue="N", data=final_df, size=3)
    ax.set(xlabel='Algorithm', ylabel='Time in seconds');
    ax.fig.suptitle(f'{target_name}, {n_rep} repetitions')
    ax.tight_layout()

    mean_df = final_df.groupby(['algorithm', 'N']).mean()
    mean_df = mean_df.rename(columns={'Time': 'Avg. Time'})
    mean_df = mean_df.transpose()
    mean_df = mean_df[algorithm]

    if latex_table == True:
        print(mean_df.to_latex(index=True))


    return ax, mean_df
