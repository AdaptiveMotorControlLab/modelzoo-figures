import matplotlib.pyplot as plt
import numpy as np
import sys
import os
import pandas as pd
import warnings
import pickle
import seaborn as sns
from matplotlib.offsetbox import AnchoredOffsetbox


df = pd.read_hdf('../data/video_adaptation_scores.h5')

video_names = ['m3v1mp4', 
               'maushaus_short', 
               'smear_mouse', 
               'golden_mouse' 
]

metrics = ['area_score']
fig, axs = plt.subplots(4, figsize=(9, 8), dpi=600)
snap_iters = range(0, 11000, 1000)
snapshot_list = [1000, 1000, 1000, 10000]
for i, video_name in enumerate(video_names):    

    areas_pre = df.loc[video_name].loc['before_adapt'].loc[metrics[0]].loc['200000']
    areas_post = df.loc[video_name].loc['after_adapt'][metrics[0]].loc[f'{snapshot_list[i]}']  

    axs[i].plot(areas_pre, c='dimgray', alpha=.5, label='w/o adaptation')
    axs[i].plot(areas_post, c='lightcoral', label='w/ adaptation')
    #scalebar = AnchoredScaleBar(
    #    axs[i].transData,
    #    sizey=400,
    #    labely='400 px$^2$',
    #    bbox_to_anchor=(825, 675),
    #    barcolor='k'
    #)
    
    axs[i].set_xticklabels([])
    axs[i].xaxis.set_tick_params(length=0)
    axs[i].set_yticklabels([])
    axs[i].yaxis.set_tick_params(length=0)
    #axs[i].add_artist(scalebar)
    sns.despine(ax=axs[i], top=True, right=True, left=True, bottom=True)
    axs[0].legend(frameon=False, loc='lower right')
fig.savefig('video_adaptation_area_plot.png', dpi=800, bbox_inches='tight', pad_inches=0.05)
