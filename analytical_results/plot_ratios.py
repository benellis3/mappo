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

    clip_range = res_df['clip_range']
    ratios_min = res_df['ratios_min']
    ratios_mean = res_df['ratios_mean']
    ratios_max = res_df['ratios_max']
    
    clip_range = pd.concat([clip_range, clip_range, clip_range])
    ratios = pd.concat([ratios_min, ratios_mean, ratios_max])

    # clip_range = pd.concat([clip_range, clip_range])
    # ratios = pd.concat([ratios_min, ratios_max])

    df_ratios = pd.DataFrame({'clip_range': clip_range, 'ratios': ratios})

    # sns_plot = sns.catplot(x="clip_range", y="ratios", jitter=True, data=df_ratios)
    sns_plot = sns.catplot(x="clip_range", y="ratios", kind="violin", bw=.3, jitter=True, data=df_ratios)
    sns.swarmplot(x="clip_range", y="ratios", color="k", size=3, data=df_ratios, ax=sns_plot.ax)

    sns_plot.set(xlabel='Clipping range', ylabel='Ratios')
    sns_plot.savefig('figures/ratios_with_clip_range.pdf')

def plot_tmp(res_df, actor_lr=0.0005, num_epochs=8):
    res_df = res_df[res_df['learning_rate'] == actor_lr]
    res_df = res_df[res_df['num_epochs'] == num_epochs]

    sns_plot = sns.relplot(x="clip_range", y="ratios_max", kind="line", 
                hue="num_epochs", col="learning_rate", data=res_df, palette="tab10",)

    sns_plot.set(xlabel='Clipping range', ylabel='Ratios')
    sns_plot.savefig('figures/ratios_with_clip_range.pdf')

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
    plot_ratios_clip_range(result_df)
    
if __name__ == '__main__':
    main()
