import os
from numpy.core import numeric
from numpy.core.fromnumeric import clip
import pandas as pd
import numpy as np
import os.path as osp
import seaborn as sns; sns.set()

import matplotlib.pyplot as plt

def mkdir(path):
    from pathlib import Path
    Path(path).mkdir(parents=True, exist_ok=True)

def plot_ratios_range(res_df, clip_range=0.1, actor_lr=0.0005):
    res_df = res_df[res_df['learning_rate'] == actor_lr]
    res_df = res_df[res_df['clip_range'] == clip_range]

    ## log scale
    # result_df['ratios_max'] = result_df['ratios_max'].apply(np.log)
    # result_df['ratios_mean'] = result_df['ratios_mean'].apply(np.log)
    # result_df['ratios_min'] = result_df['ratios_min'].apply(np.log)

    # num_epochs = res_df['num_epochs']
    # ratios_min = res_df['ratios_min']
    # ratios_mean = res_df['ratios_mean']
    # ratios_max = res_df['ratios_max']
    
    # num_epochs = pd.concat([num_epochs, num_epochs, num_epochs])
    # ratios = pd.concat([ratios_min, ratios_mean, ratios_max])
    # df_ratios = pd.DataFrame({'num_epochs': num_epochs, 'ratios': ratios})

    # sns_plot = sns.catplot(x="num_epochs", y="ratios", jitter=False, data=df_ratios)
    # sns_plot.savefig('figures/ratios_range_with_epochs.pdf')

    plt.errorbar(res_df['num_epochs'], res_df['ratios_mean'], 
                 [res_df['ratios_mean']-res_df['ratios_min'], res_df['ratios_max']-res_df['ratios_mean']], 
                 fmt='.', lw=1)
    plt.xlabel('Number of optimization epochs')
    plt.ylabel('Range of ratios')
    plt.savefig('figures/ratios_range_with_epochs.pdf')

def plot_ratios_clip_range(res_df, actor_lr=0.0005, num_epochs=10):
    res_df = res_df[res_df['learning_rate'] == actor_lr]
    res_df = res_df[res_df['num_epochs'] == num_epochs]

    all_items = []
    clip_range = res_df['clip_range']
    ratios = res_df['ratios']
    for i in range(len(clip_range)):
        ratio_list = ratios.iloc[i].split(' ')
        ratio_list = [float(s) for s in ratio_list]
        # randomly sample 1000 points
        ratio_list = np.random.choice(ratio_list, 1000, replace=False)
        for j in ratio_list:
            all_items.append({'clip_range': clip_range.iloc[i], 'ratio': j})
    
    df_ratios = pd.DataFrame(all_items)

    sns_plot = sns.catplot(x="clip_range", y="ratio", jitter=True, s=2, data=df_ratios)
    # sns_plot = sns.catplot(x="clip_range", y="ratio", kind="violin", 
    #                        bw=.2, jitter=True, data=df_ratios)
    # sns.swarmplot(x="clip_range", y="ratio", color="k", size=3, data=df_ratios, ax=sns_plot.ax)

    sns_plot.set(xlabel='Clipping range', ylabel='Ratio distribution')
    sns_plot.savefig('figures/ratios_with_clip_range_epoch_%d.pdf'%num_epochs)

def plot_ratio_range_trust_region(res_df, actor_lr=0.0005, num_epochs=10):
    res_df = res_df[res_df['learning_rate'] == actor_lr]
    # res_df = res_df[res_df['num_epochs'] == num_epochs]
    res_df = res_df[res_df['clip_range'] == 0.1]
    epsilon = res_df['epsilon_max']
    approx_TV = res_df['independent_approx_TV']
    sns_plot = sns.regplot(x=epsilon, y=approx_TV, marker="+")
    sns_plot = sns_plot.get_figure()
    ax = sns_plot.axes[0]
    y = epsilon
    ax.plot(epsilon, y, linestyle=':', color='gray')

    sns_plot.savefig('figures/ratio_range_trust_region.pdf')

def plot_ratio_range_joint_trust_region(res_df, actor_lr=0.0005, num_epochs=10):
    res_df = res_df[res_df['learning_rate'] == actor_lr]
    # res_df = res_df[res_df['num_epochs'] == num_epochs]
    res_df = res_df[res_df['clip_range'] == 0.1]
    epsilon_sum = res_df['epsilon_sum']
    approx_TV = res_df['joint_approx_TV']
    sns_plot = sns.regplot(x=epsilon_sum, y=approx_TV, marker="+")
    sns_plot = sns_plot.get_figure()
    ax = sns_plot.axes[0]
    y = epsilon_sum
    ax.plot(epsilon_sum, y, linestyle=':', color='gray')

    sns_plot.savefig('figures/ratio_range_trust_region.pdf')

def main():
    dir_path = '2s3z'
    frames_dict = {}
    frames_list = []
    rootdirs = [osp.expanduser(dir_path)]
    epoch_limit = 50
    for rootdir in rootdirs:
        for dirname, _, files in os.walk(rootdir):
            if 'progress.csv' not in files:
                continue
            progcsv = osp.join(dirname, 'progress.csv')
            tmp_df = pd.read_csv(progcsv)
            tmp_df = tmp_df[tmp_df['num_epochs'] <= epoch_limit]
            frames_dict[dirname] = tmp_df
            frames_list.append(tmp_df)

    ## create dataframe
    result_df = pd.concat(frames_list)

    mkdir('figures')

    ### plot ratios range
    # plot_ratios_range(result_df)

    ### plot ratios vs clip range
    # plot_ratios_clip_range(result_df)

    ### plot trust region vs ratios
    plot_ratio_range_trust_region(result_df)
    
if __name__ == '__main__':
    main()
