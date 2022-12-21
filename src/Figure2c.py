import matplotlib.pyplot as plt
import numpy as np
import sys
import os
import pandas as pd
import warnings
import pickle
import seaborn as sns
from matplotlib.offsetbox import AnchoredOffsetbox
sns.set_style("ticks")
warnings.filterwarnings('ignore')


import seaborn as sns
sns.set_style("ticks")

dataset_decomposition = pd.read_hdf('../data/dataset_decomposition.h5')

#for dataset in ['super_quadruped', 'super_topview']:

rename_dict = {'swimming_ole': 'Kiehn_Lab_Swimming',
               'openfield_ole': 'Kiehn_Lab_Openfield',
               'treadmill_ole': 'Kiehn_Lab_Treadmill',
               'MackenzieMausHaus': 'MausHaus',
               'daniel3mouse': 'TriMice',
               'dlc-openfield': 'DLC_Openfield',
               'TwoWhiteMice_GoldenLab': 'WhiteMice',
               'ChanLab': 'BlackMice'
              }
               

dataset_decomposition.rename(index = rename_dict, inplace = True)


def plot_horse_data_efficiency():

    dataset_size = 1469
    temp = pd.read_hdf('../data/horse_ratios.h5')
    rename_dict = {'baseline': 'ImageNet transfer learning',
                   'zeroshot': 'SA + Zeroshot',
                   'super_remove_head': 'SA + Randomly Initialized Decoder',          
                   'unbalanced_memory_replay_threshold_0.8_700000': 'SA + Memory Replay',
                   'unbalanced_naive_finetune_700000': 'SA + Naive Fine-tuning'}
    temp.rename(index=rename_dict, inplace=True)
    unbalanced_zeroshot = temp.loc['unbalanced_zeroshot', '1000']#.mean(axis=0)

    unbalanced_zeroshot = unbalanced_zeroshot.loc[['shuffle1_best', 'shuffle2_best', 'shuffle3_best']]

    print (unbalanced_zeroshot)



    df = temp.reset_index().loc(axis=1)[['level_0', 'level_1', 'level_2', 'NE_iid', 'NE_ood']]
    df_masked = df[(df['level_0'] != 'unbalanced_zeroshot') & (df['level_0'] != 'balanced_zeroshot')]

    df_masked['level_1'] = (pd.to_numeric(df_masked['level_1']) * dataset_size).astype(int)



    fig, (ax1, ax2) = plt.subplots(
        ncols=2,
        tight_layout=True,
        figsize=(7, 4),
        sharex=True,
        sharey=True,
    )

    print (unbalanced_zeroshot['NE_iid'])

    ax1.axhline(unbalanced_zeroshot['NE_iid'].mean(), ls=':', lw=2)

    NE_iid_min = np.min(unbalanced_zeroshot['NE_iid'])
    NE_iid_max = np.max(unbalanced_zeroshot['NE_iid'])

    ax1.axhspan(NE_iid_min, NE_iid_max, facecolor='grey', alpha=0.5)

    NE_ood_min = np.min(unbalanced_zeroshot['NE_ood'])
    NE_ood_max = np.max(unbalanced_zeroshot['NE_ood'])

    ax2.axhspan(NE_ood_min, NE_ood_max, facecolor='grey', alpha=0.5)

    ax2.axhline(unbalanced_zeroshot['NE_ood'].mean(), ls=':', lw=2)
    pal = 'magma_r'



    sns.pointplot(data=df_masked, x="level_1", y="NE_iid", ax=ax1, hue='level_0', palette=pal, ci = None,)
    sns.stripplot(data=df_masked, x="level_1", y="NE_iid", ax=ax1, hue='level_0', palette=pal, alpha = .5)
    sns.pointplot(data=df_masked, x="level_1", y="NE_ood", ax=ax2, hue='level_0', palette=pal, ci = None)
    sns.stripplot(data=df_masked, x="level_1", y="NE_ood", ax=ax2, hue='level_0', palette=pal, alpha = .5)


    ax1.legend().remove()
    ax2.legend().remove()
    handles, labels = ax1.get_legend_handles_labels()

    fig.legend(
        handles[:4],
        labels[:4],
        frameon=False,
        ncol=1,
        loc='upper right',
        fontsize='medium',
        bbox_to_anchor=(1.5, 0.99),
    )
    ax1.set_xlabel('')
    ax1.set_ylabel('Normalized error IID')
    ax1.set_ylim(0, 1.8)
    ax2.set_xlabel('')
    ax2.set_ylabel('Normalized error OOD')
    sns.despine(top=True, right=True, ax=ax1)
    sns.despine(top=True, right=True, ax=ax2)
    fig.supxlabel('Number of fine-tuning images', y=0.05, x=0.5125)
    fig.savefig('horses.png', dpi=800, bbox_inches='tight', pad_inches=0.05)



def plot_rodent_data_efficiency():

    temp = pd.read_hdf('../data/rodent_ratios.h5')
    #print (temp.to_string())
    dataset_size = dataset_decomposition.loc['super_quadruped'].loc['iRodents'].loc['num_images']
    unbalanced_zeroshot = temp.loc['unbalanced_zeroshot', '700000']#.mean(axis=0)
    drop_list = ['unbalanced_zeroshot',
                 #'balanced_zeroshot'
                ]
    temp.drop(drop_list, inplace=True)

    rename_dict = {'baseline': 'ImageNet transfer learning',
                   'zeroshot': 'SA + Zeroshot',
                   'super_remove_head': 'SA + Randomly Initialized Decoder',          
                   'unbalanced_memory_replay_threshold_0.8_700000': 'SA + Memory Replay',
                   'unbalanced_naive_finetune_700000': 'SA + Naive Fine-tuning'}
    temp.rename(index=rename_dict, inplace=True)
    df = temp.reset_index().loc(axis=1)[['level_0', 'level_1', 'level_2', 'NE']]
    df_masked = df[df['level_0'] != 'unbalanced_zeroshot']
    df_masked['level_1'] = (pd.to_numeric(df_masked['level_1']) * dataset_size * 0.8).astype(int)
    fig, ax1 = plt.subplots(
        ncols=1,
        tight_layout=True,
        figsize=(3.5, 4),
        sharex=True,
        sharey=True,
    )

    zeroshot_NE_min = np.min(unbalanced_zeroshot['NE'])
    zeroshot_NE_max = np.max(unbalanced_zeroshot['NE'])

    ax1.axhline(unbalanced_zeroshot['NE'].mean(), ls=':', lw=2)

    ax1.axhspan(zeroshot_NE_min, zeroshot_NE_max, facecolor='grey', alpha=0.5)

    pal = 'magma_r'
    #print (df_masked)
    sns.pointplot(data=df_masked, x="level_1", y="NE", ax=ax1, hue='level_0', palette=pal, hue_order=['ImageNet transfer learning', 'SA + Randomly Initialized Decoder', 'SA + Memory Replay', 'SA + Naive Fine-tuning'], ci=None)
    sns.stripplot(data=df_masked, x="level_1", y="NE", ax=ax1, hue='level_0', palette=pal, alpha=.5, hue_order=['ImageNet transfer learning', 'SA + Randomly Initialized Decoder', 'SA + Memory Replay', 'SA + Naive Fine-tuning'])
    ax1.legend().remove()
    ax1.set_xlabel('')
    ax1.set_ylabel('Normalized error')
    ax1.set_ylim(0, 6)
    sns.despine(top=True, right=True, ax=ax1)
    fig.supxlabel('Number of fine-tuning images', y=0.05, x=0.5125)
    fig.savefig('rodents.png', dpi=600, bbox_inches='tight', pad_inches=0.05)    
    

def plot_openfield_data_efficiency():
    temp = pd.read_hdf('../data/openfield_ratios.h5')

    dataset_size = dataset_decomposition.loc['super_topview'].loc['DLC_Openfield'].loc['num_images']
    unbalanced_zeroshot = temp.loc['unbalanced_zeroshot', '600000']#.mean(axis=0)

    print (unbalanced_zeroshot.mean())

    drop_list = ['balanced_memory_replay_threshold_0.0_750000',
                 'unbalanced_zeroshot',
                 'balanced_zeroshot',
                 'balanced_super_remove_head_750000',
                 'balanced_memory_replay_threshold_0.8_snapshot_700000',
                 'balanced_memory_replay_threshold_0.8_750000',
                 'unbalanced_memory_replay_750000']
    temp.drop(drop_list, inplace=True)

    print (temp.groupby(level = (0,1)).mean())
    #print (temp.to_string())
    rename_dict = {'baseline': 'ImageNet transfer learning',
                   'zeroshot': 'SA + Zeroshot',
                   'super_remove_head': 'SA + Randomly Initialized Decoder',          
                   'unbalanced_memory_replay_threshold_0.8_700000': 'SA + Memory Replay',
                   'unbalanced_naive_finetune_700000': 'SA + Naive Fine-tuning'}
    temp.rename(index=rename_dict, inplace=True)
    df = temp.reset_index().loc(axis=1)[['level_0', 'level_1', 'level_2', 'RMSE']]
    df_masked = df[df['level_0'] != 'unbalanced_zeroshot']
    df_masked['level_1'] = (pd.to_numeric(df_masked['level_1']) * dataset_size * 0.95).astype(int)
    fig, ax1 = plt.subplots(
        ncols=1,
        tight_layout=True,
        figsize=(3.5, 4),
        sharex=True,
        sharey=True,
    )


    RMSE_min = np.min(unbalanced_zeroshot['RMSE'])
    RMSE_max = np.max(unbalanced_zeroshot['RMSE'])

    ax1.axhline(unbalanced_zeroshot['RMSE'].mean(), ls=':', lw=2)

    ax1.axhspan(RMSE_min, RMSE_max, facecolor='grey', alpha=0.5)

    pal = 'magma_r'
    sns.pointplot(data=df_masked, x="level_1", y="RMSE", ax=ax1, hue='level_0', palette=pal, errorbar=None)
    sns.stripplot(data=df_masked, x="level_1", y="RMSE", ax=ax1, hue='level_0', palette=pal, alpha=.5)
    ax1.legend().remove()
    ax1.set_xlabel('')
    ax1.set_ylabel('RMSE')
    ax1.set_ylim(0, 25)
    sns.despine(top=True, right=True, ax=ax1)
    fig.supxlabel('Number of fine-tuning images', y=0.05, x=0.5125)
    fig.savefig('openfield.png', dpi=600, bbox_inches='tight', pad_inches=0.05)    



plot_horse_data_efficiency()
plot_rodent_data_efficiency()
plot_openfield_data_efficiency()
