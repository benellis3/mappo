import os
import pandas as pd
import numpy as np
import os.path as osp
from pandas.core.frame import DataFrame
import seaborn as sns; sns.set()
sns.set_context("paper")

import matplotlib.pyplot as plt

def mkdir(path):
    from pathlib import Path
    Path(path).mkdir(parents=True, exist_ok=True)

def plot_ratios_range(res_df, clip_range=0.1, actor_lr=0.0005):
    res_df = res_df[res_df['learning_rate'] == actor_lr]

    plt.style.use('seaborn-darkgrid')

    plot_list = [10.0, 0.5, 0.4, 0.3, 0.2, 0.1]
    # plot_list = [0.1, 0.2, 0.3, 0.4, 0.5, 10.0]
    legends = ['No clipping', 'Clipping 0.5', 'Clipping 0.4', 
               'Clipping 0.3', 'Clipping 0.2', 'Clipping 0.1']
    for i, r in enumerate(plot_list):
        res_df_tmp = res_df[res_df['clip_range'] == r]
        plt.errorbar(res_df_tmp['num_epochs'], res_df_tmp['ratios_mean'], 
                    [res_df_tmp['ratios_mean']-res_df_tmp['ratios_min'], res_df_tmp['ratios_max']-res_df_tmp['ratios_mean']], 
                    lw=4, label=legends[i], alpha=1.0)

    plt.xlabel('Number of optimization epochs')
    plt.ylabel('Range of ratios')
    plt.legend(loc='upper left')
    plt.savefig('figures/ratios_range_with_epochs.pdf')

def plot_ratios_clip_range(res_df, actor_lr=0.0005, num_epochs=20, num_samples=100):
    res_df = res_df[res_df['learning_rate'] == actor_lr]
    res_df = res_df[res_df['num_epochs'] == num_epochs]

    all_items = []
    clip_range = res_df['clip_range']
    ratios = res_df['ratios']
    for i in range(len(clip_range)):
        ratio_list = ratios.iloc[i].split(' ')
        ratio_list = [float(s) for s in ratio_list]
        # randomly sample 100 points
        ratio_list = np.random.choice(ratio_list, num_samples, replace=False)
        for j in ratio_list:
            r = clip_range.iloc[i]
            all_items.append({'clip_range': r, 'ratio': j})
    
    df_ratios = pd.DataFrame(all_items)
    sns_plot = sns.catplot(x="clip_range", y="ratio", jitter=True, s=2, data=df_ratios)

    sns_plot.set(xlim=(0, 1.0), xlabel='Clipping range', ylabel='Ratio distribution')
    sns_plot.set_xticklabels(['0.1', '0.2', '0.3', '0.4', '0.5', 'No clipping'])
    sns_plot.savefig('figures/ratios_with_clip_range_epoch_%d.pdf'%num_epochs)

def plot_ratio_range_trust_region(res_df, actor_lr=0.0005):
    res_df = res_df[res_df['learning_rate'] == actor_lr]
    res_df['clip_range'] = res_df['clip_range'].apply(lambda r: str(r) if r != 10.0 else 'No clipping')
    hue_order = ['0.1', '0.2', '0.3', '0.4', '0.5', 'No clipping']
    sns_plot = sns.displot(res_df, x="independent_approx_TV", hue="clip_range", 
                           hue_order=hue_order, kind="ecdf", legend=False)

    hue_order = ['No clipping', '0.5', '0.4', '0.3', '0.2', '0.1']
    plt.legend(labels = hue_order, loc='lower right', title='Cliping range') # plt legend is ordered reversely
    sns_plot.set(xlim=(0, 1.0), xlabel=r'$D_{\mathrm{TV}}^{\mathrm{max}}(\pi_k, \tilde{\pi}_k)$', ylabel='Cumulative percentage')
    sns_plot.savefig('figures/trust_region_clipping.pdf')

def plot_ratio_range_joint_trust_region(res_df, actor_lr=0.0005):
    res_df = res_df[res_df['learning_rate'] == actor_lr]
    res_df = res_df[res_df['clip_range'] == 0.1]

    res_df['num_agents'] = res_df['num_agents'].apply(str)
    hue_order = ['2', '3', '5', '6', '8', '9', '10']

    sns_plot = sns.displot(res_df, x="joint_approx_TV", hue="num_agents",
                           hue_order=hue_order, kind="ecdf", legend=False)

    hue_order = ['10', '9', '8', '6', '5', '3', '2']
    plt.legend(labels = hue_order, loc='lower right', title='Number of agents')
    sns_plot.set(xlim=(0, 1.0), xlabel=r'$D_{\mathrm{TV}}^{\mathrm{max}}(\mathbf{\pi}, \tilde{\mathbf{\pi}})$', ylabel='Cumulative percentage')
    sns_plot.savefig('figures/trust_region_joint_num_agents.pdf')

def plot_ratio_range_epochs_trust_region(res_df, actor_lr=0.0005):
    res_df = res_df[res_df['learning_rate'] == actor_lr]
    res_df = res_df[res_df['clip_range'] == 0.1]

    t1 = res_df['num_epochs'] == 10
    checkpoint_epoch = [15, 20, 25, 30, 40]
    for epoch in checkpoint_epoch:
        t1 = t1 | (res_df['num_epochs'] == epoch)

    res_df = res_df[t1]
    res_df['num_epochs'] = res_df['num_epochs'].apply(str)

    hue_order = ['10', '15', '20', '25', '30', '40']
    sns_plot = sns.displot(res_df, x="independent_approx_TV", hue="num_epochs",
                           hue_order=hue_order, kind="ecdf", legend=False)

    hue_order = ['40', '30', '25', '20', '15', '10']
    plt.legend(labels = hue_order, loc='lower right', title='Number of epochs')

    sns_plot.set(xlim=(0, 1.0), xlabel=r'$D_{\mathrm{TV}}^{\mathrm{max}}(\pi_k, \tilde{\pi}_k)$', ylabel='Cumulative percentage')
    sns_plot.savefig('figures/trust_region_epoch.pdf')

def plot_joint_ratio_range_epochs_trust_region(res_df, actor_lr=0.0005):
    res_df = res_df[res_df['learning_rate'] == actor_lr]
    res_df = res_df[res_df['clip_range'] == 0.1]

    t1 = res_df['num_epochs'] == 10
    checkpoint_epoch = [15, 20, 25, 30, 40]
    for epoch in checkpoint_epoch:
        t1 = t1 | (res_df['num_epochs'] == epoch)

    res_df = res_df[t1]
    res_df['num_epochs'] = res_df['num_epochs'].apply(str)

    hue_order = ['10', '15', '20', '25', '30', '40']
    sns_plot = sns.displot(res_df, x="joint_approx_TV", hue="num_epochs",
                           hue_order=hue_order, kind="ecdf", legend=False)

    hue_order = ['40', '30', '25', '20', '15', '10']
    plt.legend(labels = hue_order, loc='lower right', title='Number of epochs')

    sns_plot.set(xlim=(0, 1.0), xlabel=r'$D_{\mathrm{TV}}^{\mathrm{max}}(\mathbf{\pi}, \tilde{\mathbf{\pi}})$', ylabel='Cumulative percentage')
    sns_plot.savefig('figures/trust_region_joint_epoch.pdf')

def main():
    dir_path = 'mixed_agents'
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
    # plot_ratio_range_trust_region(result_df)

    ### plot joint trust region vs ratios
    # plot_ratio_range_joint_trust_region(result_df)

    ### plot joint trust region vs epochs
    plot_ratio_range_epochs_trust_region(result_df)
    # plot_joint_ratio_range_epochs_trust_region(result_df)
    
if __name__ == '__main__':
    main()
