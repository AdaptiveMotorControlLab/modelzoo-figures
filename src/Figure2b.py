import matplotlib.pyplot as plt
import numpy as np
import sys
import os
import pandas as pd
import warnings
import pickle
import seaborn as sns
from matplotlib.offsetbox import AnchoredOffsetbox

import seaborn as sns
sns.set_style("ticks")

openfield_df = pd.read_hdf('../data/openfield_ratios.h5')


openfield_unbalanced_zeroshot = openfield_df.loc['unbalanced_zeroshot', '600000'].mean(axis=0)

drop_list = ['balanced_memory_replay_threshold_0.0_750000',
             'unbalanced_zeroshot',
             'balanced_zeroshot',
             'balanced_super_remove_head_750000',
             'balanced_memory_replay_threshold_0.8_snapshot_700000',
             'balanced_memory_replay_threshold_0.8_750000',
             'unbalanced_memory_replay_750000']

openfield_df.drop(drop_list, inplace=True)


rename_dict = {'baseline': 'ImageNet transfer learning',
               'zeroshot': 'SA + Zeroshot',
               'unbalanced_super_remove_head_750000': 'SA + Randomly Initialized Decoder',          
               'unbalanced_memory_replay_threshold_0.8_750000': 'SA + Memory Replay',
               'unbalanced_naive_finetune_750000': 'SA + Naive finetuning'}
openfield_df.rename(index=rename_dict, inplace=True)



openfield_RMSE = openfield_df.groupby(level = (0,1)).mean()['RMSE']


rodent_df = pd.read_hdf('../data/rodent_ratios.h5')

rodent_unbalanced_zeroshot = rodent_df.loc['unbalanced_zeroshot', '700000'].mean(axis=0)
drop_list = ['unbalanced_zeroshot',
             #'balanced_zeroshot'
            ]
rodent_df.drop(drop_list, inplace=True)

rename_dict = {'baseline': 'ImageNet transfer learning',
               'zeroshot': 'SA + Zeroshot',
               'unbalanced_super_remove_head_700000': 'SA + Randomly Initialized Decoder',          
               'unbalanced_memory_replay_threshold_0.8_700000': 'SA + Memory Replay',
               'unbalanced_naive_finetune_700000': 'SA + Naive Fine-tuning'}


rodent_df.rename(index = rename_dict, inplace = True)

rodent_RMSE = rodent_df.groupby(level = (0,1)).mean()['RMSE']


#print (rodent_RMSE)

horse_df = pd.read_hdf('../data/horse_ratios.h5')
horse_unbalanced_zeroshot = horse_df.loc['unbalanced_zeroshot', '700000'].mean(axis=0)
drop_list = ['unbalanced_zeroshot']

horse_df.drop(drop_list, inplace = True)

rename_dict = {'baseline': 'ImageNet transfer learning',
               'zeroshot': 'SA + Zeroshot',
               'super_remove_head': 'SA + Randomly Initialized Decoder',          
               'unbalanced_memory_replay_threshold_0.8_700000': 'SA + Memory Replay',
               'unbalanced_naive_finetune_700000': 'SA + Naive Fine-tuning'}
horse_df.rename(index=rename_dict, inplace=True)


horse_RMSE = horse_df['RMSE_iid'].groupby(level = [0,1]).mean()

horse_RMSE_iid = horse_RMSE

horse_RMSE_ood = horse_df['RMSE_ood'].groupby(level = [0,1]).mean()

print (horse_unbalanced_zeroshot)

#names = ['openfield', 'rodent', 'horse']

names = ['DLC-Openfield', 'iRodents', 'Horse-10']


pal = 'magma_r'

fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
 # adjust space between axes

ratios = ['0.01', '0.05', '0.1', '0.5', '1.0']
dataset_colors = plt.cm.get_cmap('tab20b', 10)
zeroshots = [openfield_unbalanced_zeroshot, rodent_unbalanced_zeroshot, horse_unbalanced_zeroshot]
for idx, dataset_score in enumerate([openfield_RMSE, rodent_RMSE, horse_RMSE]):
    name = names[idx]
    #print (dataset_RMSE['ImageNet transfer learning'])
    #baseline = dataset_RMSE.loc['ImageNet transfer learning'].loc[str(ratio)]

    color = dataset_colors(idx)
    
    # rodent uses ax1
    if idx == 1:
        ax = ax1
    else:
        ax = ax2
    print (ax)
    
    zeroshot = zeroshots[idx]['RMSE'] if idx != 2 else zeroshots[idx]['RMSE_iid']
    print (zeroshot)
    zeroshot = np.tile(zeroshot, len(ratios))

    ax.plot(ratios, dataset_score['ImageNet transfer learning'], 
             label = name + ' ' + 'ImageNet transfer learning', 
             linestyle = 'dashed',
             color = color,
             marker = 'o')
    
    ax.plot(ratios, dataset_score['SA + Memory Replay'], 
             label = name + ' '+'SA + Memory replay',
             linestyle = 'solid',
             color = color,             
             marker = 'o')
    ax.plot(ratios, zeroshot,
           label = name + ' ' + 'Zeroshot',
            linestyle = 'dashed',
            linewidth = 3,
            color = color)
            
    
ax1.spines.bottom.set_visible(False)
ax1.spines.top.set_visible(False)
ax2.spines.top.set_visible(False)

#ax1.xaxis.tick_top()
fig.subplots_adjust(hspace=0.05)     
#ax1.legend().remove


ax1.tick_params(labeltop=False)  # don't put tick labels at the top
ax2.xaxis.tick_bottom()
ax1.tick_params(bottom = False)
d = .5  # proportion of vertical to horizontal extent of the slanted line
kwargs = dict(marker=[(-1, -d), (1, d)], markersize=12,
              linestyle="none", color='k', mec='k', mew=1, clip_on=False)
ax1.plot([0, 1], [0, 0], transform=ax1.transAxes, **kwargs)
ax2.plot([0, 1], [1, 1], transform=ax2.transAxes, **kwargs)

ax1.set_xlabel('')
fig.text(0.04, 0.5, 'RMSE', va='center', rotation='vertical')
fig.text(0.5, 0.04, 'Training data fraction', ha='center')
ax1.set_ylim(20, 550)
ax2.set_ylim(0, 30)
fig.legend(bbox_to_anchor=(1.5, 1.0))

plt.savefig('summary_plot.png', dpi=800,  bbox_inches='tight')
