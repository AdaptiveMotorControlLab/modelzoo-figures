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

dataset_size = 354




rename_dict = {'from_scratch': 'ImageNet transfer learning',
               'rand_head_transfer': 'SA + Randomly Initialized Decoder',          
               'memory_replay': 'SA + Memory Replay',
               'naive_finetune': 'SA + Naive Fine-tuning',
               'zeroshot': 'SA + Zeroshot',
               'ap10k_zeroshot': 'AP10K + Zeroshot',
               'ap10k_finetune': 'AP10K + Fine-tuning'}


df = pd.read_hdf('../data/rodent_topdown.h5')

custom_order = ["from_scratch",
                "rand_head_transfer",
                "memory_replay",
                "naive_finetune",
                "zeroshot",
                'ap10k_zeroshot',
                'ap10k_finetune']

df = df.loc[custom_order]

zeroshot = df.loc['zeroshot']
ap10k_zeroshot = df.loc['ap10k_zeroshot']

df.rename(index=rename_dict, inplace=True)

df = df.reset_index().loc(axis=1)[['level_0', 'level_1', 'level_2', 'mAP']]

df_masked = df[(df['level_0'] !='SA + Zeroshot') & (df['level_0'] != 'AP10K + Zeroshot')]


print (df_masked)

df_masked['level_1'] = (pd.to_numeric(df_masked['level_1']) * dataset_size).astype(int)

fig, (ax1) = plt.subplots(
    ncols=1,
    tight_layout=True,
    figsize=(7, 4),
    sharex=True,
    sharey=True,
)
print (df_masked.index)

ax1.axhline(zeroshot['mAP'].mean(), ls=':', lw=2, color = 'blue')

ax1.axhline(ap10k_zeroshot['mAP'].mean(), ls=':', lw=2, color = 'green')

superanimal_mAP_min = np.min(zeroshot['mAP'])
superanimal_mAP_max = np.max(zeroshot['mAP'])

ap10k_mAP_min = np.min(ap10k_zeroshot['mAP'])
ap10k_mAP_max = np.max(ap10k_zeroshot['mAP'])

#ax1.axhspan(superanimal_mAP_min, superanimal_mAP_max, facecolor='grey', alpha=0.5)
#ax1.axhspan(ap10k_mAP_min, ap10k_mAP_max, facecolor='green', alpha=0.5)

pal = 'magma_r'

sns.pointplot(data=df_masked, x="level_1", y="mAP", ax=ax1, hue='level_0', palette=pal, errorbar=None)
sns.stripplot(data=df_masked, x="level_1", y="mAP", ax=ax1, hue='level_0', palette=pal, alpha = .5)

ax1.legend().remove()

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
ax1.set_ylabel('mAP')
ax1.set_ylim(0, 1.0)

sns.despine(top=True, right=True, ax=ax1)
fig.supxlabel('Number of fine-tuning images', y=0.05, x=0.5125)

fig.savefig('Figure1f_topdown_rodent.png', dpi=800, bbox_inches='tight', pad_inches=0.05, transparent = True)

plt.show()
