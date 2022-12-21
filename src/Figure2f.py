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
sns.set_style('ticks')
# %%
with open('../data/ood_mice_zeroshot.pickle', 'rb') as f:
    pickle_obj = pickle.load(f)
# %%
proj_roots = list(pickle_obj.keys())
proj_nicknames = ['smear mouse', 'ood maushaus', 'golden mouse']
method_colors = plt.cm.get_cmap('magma_r', 6)

fig, axes = plt.subplots(3, figsize=(4, 6))

remove_nan = lambda x: x[~np.isnan(x)]
dfs = []
for i, proj_root in enumerate(pickle_obj):
    with_spatial_pyramid = np.nanmean(pickle_obj[proj_root]['with_spatial_pyramid']['RMSE'], axis = (1,2))
    with_spatial_pyramid = remove_nan(with_spatial_pyramid)
    without_spatial_pyramid = np.nanmean(pickle_obj[proj_root]['without_spatial_pyramid']['RMSE'],axis = (1,2))
    without_spatial_pyramid = remove_nan(without_spatial_pyramid)
    with_pyramid = ["without"] * len(without_spatial_pyramid) + ["with"] * len(with_spatial_pyramid)
    df_ = pd.DataFrame(
        np.array([
            np.r_[without_spatial_pyramid, with_spatial_pyramid],
            with_pyramid,
        ]).T,
    columns=['RMSE', 'cond'])
    df_['dataset'] = proj_nicknames[i]
    df_['RMSE'] = df_['RMSE'].astype("float64")
    df_['cond'] = df_['cond'].astype("category")
    df_['dataset'] = df_['dataset'].astype("category")
    dfs.append(df_)
    vp = sns.violinplot(
        df_, y='dataset', x='RMSE', hue='cond', split=True, hue_order=['without', 'with'],
        inner="quart", linewidth=1, palette={"without": ".85", "with": "#8AB5E7"},
        ax=axes[i], bw='scott',
    )
    sns.despine(ax=axes[i], left=True, top=True, right=True)
    axes[i].set_yticks([])
    axes[i].set_ylabel("")
    # axes[i].set_xlim(left=0)
for i in (0, 1):
    axes[i].legend().remove()
handles, labels = axes[2].get_legend_handles_labels()
fig.legend(
    handles,
    ['Without spatial pyramid', 'With spatial pyramid'],
    frameon=False,
    ncol=1,
    # loc='lower right',
    fontsize='small',
    bbox_to_anchor=(0.9, 0.2),
)
axes[2].legend().remove()
fig.savefig('spatial_pyramid_metrics.png', dpi=600, bbox_inches='tight', pad_inches=0.05)
