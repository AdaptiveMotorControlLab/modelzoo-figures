import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import json
import sys
import copy
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
sns.set_style("ticks")

dataset_size = 1469

rename_dict = {'from_scratch': 'ImageNet transfer learning',
               'zeroshot': 'SA + Zeroshot',
               'rand_head_transfer': 'SA + Randomly Initialized Decoder',          
               'memory_replay': 'SA + Memory Replay',
               'naive_finetune': 'SA + Naive Fine-tuning'}


df = pd.read_hdf('../data/horse_topdown.h5')

custom_order = ["from_scratch",
                "rand_head_transfer",
                "memory_replay",
                "naive_finetune",
                "zeroshot"]

df = df.loc[custom_order]


zeroshot = df.loc['zeroshot']
df.rename(index=rename_dict, inplace=True)


df = df.reset_index().loc(axis=1)[['level_0', 'level_1', 'level_2', 'NE_iid', 'NE_ood']]

df_masked = df[(df['level_0']) !='SA + Zeroshot']

df.rename(index=rename_dict, inplace=True)

print (df_masked)

df_masked['level_1'] = (pd.to_numeric(df_masked['level_1']) * dataset_size).astype(int)

fig, (ax1, ax2) = plt.subplots(
    ncols=2,
    tight_layout=True,
    figsize=(7, 4),
    sharex=True,
    sharey=True,
)

ax1.axhline(zeroshot['NE_iid'].mean(), ls=':', lw=2)
NE_iid_min = np.min(zeroshot['NE_iid'])
NE_iid_max = np.max(zeroshot['NE_iid'])
ax1.axhspan(NE_iid_min, NE_iid_max, facecolor='grey', alpha=0.5)
NE_ood_min = np.min(zeroshot['NE_ood'])
NE_ood_max = np.max(zeroshot['NE_ood'])

ax2.axhspan(NE_ood_min, NE_ood_max, facecolor='grey', alpha=0.5)

ax2.axhline(zeroshot['NE_ood'].mean(), ls=':', lw=2)

pal = 'magma_r'
sns.pointplot(data=df_masked, x="level_1", y="NE_iid", ax=ax1, hue='level_0', palette=pal, errorbar=None)
sns.stripplot(data=df_masked, x="level_1", y="NE_iid", ax=ax1, hue='level_0', palette=pal, alpha = .5)

sns.pointplot(data=df_masked, x="level_1", y="NE_ood", ax=ax2, hue='level_0', palette=pal, errorbar = None)
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
ax1.set_ylim(0, 3.0)
ax2.set_xlabel('')
ax2.set_ylabel('Normalized error OOD')
sns.despine(top=True, right=True, ax=ax1)
sns.despine(top=True, right=True, ax=ax2)
fig.supxlabel('Number of fine-tuning images', y=0.05, x=0.5125)

fig.savefig('Figure1f_topdown_horse.png', dpi=800, bbox_inches='tight', pad_inches=0.05, transparent = True)
