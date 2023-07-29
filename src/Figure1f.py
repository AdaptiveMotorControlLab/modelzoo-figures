import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


sns.set_style("ticks")

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
fig.savefig('Figure1f.png', dpi=800, bbox_inches='tight', pad_inches=0.05)
