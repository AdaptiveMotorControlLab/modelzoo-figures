import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import pickle
import seaborn as sns
import seaborn as sns
import sys
import warnings
from matplotlib.offsetbox import AnchoredOffsetbox

sns.set_style("ticks")


def plot_figure2b():
    openfield_df = pd.read_hdf('../data/Figure2/openfield_ratios.h5')

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
    openfield_RMSE = openfield_df.groupby(level=(0, 1)).mean()['RMSE']

    rodent_df = pd.read_hdf('../data/Figure2/rodent_ratios.h5')

    rodent_unbalanced_zeroshot = rodent_df.loc['unbalanced_zeroshot', '700000'].mean(axis=0)
    drop_list = ['unbalanced_zeroshot',
                 # 'balanced_zeroshot'
                 ]
    rodent_df.drop(drop_list, inplace=True)

    rename_dict = {'baseline': 'ImageNet transfer learning',
                   'zeroshot': 'SA + Zeroshot',
                   'unbalanced_super_remove_head_700000': 'SA + Randomly Initialized Decoder',
                   'unbalanced_memory_replay_threshold_0.8_700000': 'SA + Memory Replay',
                   'unbalanced_naive_finetune_700000': 'SA + Naive Fine-tuning'}

    rodent_df.rename(index=rename_dict, inplace=True)

    rodent_RMSE = rodent_df.groupby(level=(0, 1)).mean()['RMSE']

    # print (rodent_RMSE)

    horse_df = pd.read_hdf('../data/Figure2/horse_ratios.h5')
    horse_unbalanced_zeroshot = horse_df.loc['unbalanced_zeroshot', '1000'].mean(axis=0)
    drop_list = ['unbalanced_zeroshot']
    horse_df.drop(drop_list, inplace=True)
    rename_dict = {'baseline': 'ImageNet transfer learning',
                   'zeroshot': 'SA + Zeroshot',
                   'super_remove_head': 'SA + Randomly Initialized Decoder',
                   'unbalanced_memory_replay_threshold_0.8_700000': 'SA + Memory Replay',
                   'unbalanced_naive_finetune_700000': 'SA + Naive Fine-tuning'}
    horse_df.rename(index=rename_dict, inplace=True)

    horse_RMSE = horse_df['RMSE_iid'].groupby(level=[0, 1]).mean()
    horse_RMSE_iid = horse_RMSE
    horse_RMSE_ood = horse_df['RMSE_ood'].groupby(level=[0, 1]).mean()

    print(horse_unbalanced_zeroshot)

    names = ['openfield', 'rodent', 'horse']
    pal = 'magma_r'

    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(6, 4), dpi=600)
    # adjust space between axes

    ratios = ['0.01', '0.05', '0.1', '0.5', '1.0']
    # ratios = list(map(float, ratios))
    colors = ["3d5a80", "98c1d9", "e7d7c1"]
    colors = [f"#{c}" for c in colors]
    lw = 3
    zeroshots = [openfield_unbalanced_zeroshot, rodent_unbalanced_zeroshot, horse_unbalanced_zeroshot]
    for idx, dataset_score in enumerate([openfield_RMSE, rodent_RMSE, horse_RMSE]):
        name = names[idx]
        color = colors[idx]

        if idx == 1:
            ax = ax1
        else:
            ax = ax2

        zeroshot = zeroshots[idx]['RMSE'] if idx != 2 else zeroshots[idx]['RMSE_iid']
        zeroshot = np.tile(zeroshot, len(ratios))

        ax.plot(ratios, dataset_score['ImageNet transfer learning'],
                label=name + ' ' + 'ImageNet transfer learning',
                linestyle='dashed',
                color=color,
                marker='o',
                lw=lw)

        ax.plot(ratios, dataset_score['SA + Memory Replay'],
                label=name + ' ' + 'SA + Memory replay',
                linestyle='solid',
                color=color,
                marker='o',
                #  alpha=0.6,
                lw=lw)
        ax.plot(ratios, zeroshot,
                label=name + ' ' + 'Zeroshot',
                linestyle='dotted',
                lw=lw,
                # alpha=0.3,
                color=color)

    ax1.spines.bottom.set_visible(False)
    ax1.spines.top.set_visible(False)
    ax2.spines.top.set_visible(False)

    # ax1.xaxis.tick_top()
    fig.subplots_adjust(hspace=0.05)
    # ax1.legend().remove

    ax1.tick_params(labeltop=False)  # don't put tick labels at the top
    ax2.xaxis.tick_bottom()
    ax1.tick_params(bottom=False)
    d = .5  # proportion of vertical to horizontal extent of the slanted line
    kwargs = dict(marker=[(-1, -d), (1, d)], markersize=12,
                  linestyle="none", color='k', mec='k', mew=1, clip_on=False)
    ax1.plot([0, 1], [0, 0], transform=ax1.transAxes, **kwargs)
    ax2.plot([0, 1], [1, 1], transform=ax2.transAxes, **kwargs)

    ax1.set_xlabel('')
    ax2.set_xlabel('Training data fraction')
    # ax2.set_ylabel('RMSE')
    fig.text(0.04, 0.5, 'RMSE', va='center', rotation='vertical')
    # fig.text(0.5, 0.04, 'Training data fraction', ha='center')
    ax1.set_ylim(20, 550)
    ax2.set_ylim(0, 30)

    mew = 0
    mec = 'dimgray'
    patches = [
        plt.Line2D([0], [0], color='none', marker='o', label="DLC-Openfield", markersize=8, mfc=colors[0], mec=mec,
                   mew=mew),
        plt.Line2D([0], [0], color='none', marker='o', label="iRodent", markersize=8, mfc=colors[1], mec=mec, mew=mew),
        plt.Line2D([0], [0], color='none', marker='o', label="Horse-10", markersize=8, mfc=colors[2], mec=mec, mew=mew),
    ]
    fc = 'lightgray'
    leg = fig.legend()
    patches_ = leg.legendHandles.copy()
    for p in patches_:
        p.set_markerfacecolor(fc)
        p.set_color(fc)
    leg.remove()
    patches.extend([
        patches_[0],
        patches_[4],
        patches_[8],
    ])
    patches[-3].set_label('ImageNet transfer learning')
    patches[-2].set_label('Memory replay')
    patches[-1].set_label('Zero-shot')
    fig.legend(
        handles=patches,
        frameon=False,
        borderaxespad=0.0,
        bbox_to_anchor=(0.88, 0.85),
        ncol=2,
        columnspacing=1,
        labelspacing=0.5,
        fontsize='medium',
        scatteryoffsets=[0.5],
    )
    fig.savefig('Figure2b.png', dpi=600, bbox_inches='tight', pad_inches=0.05)


def plot_figure2c():
    dataset_decomposition = pd.read_hdf('../data/Figure2/dataset_decomposition.h5')

    # for dataset in ['super_quadruped', 'super_topview']:

    rename_dict = {'swimming_ole': 'Kiehn_Lab_Swimming',
                   'openfield_ole': 'Kiehn_Lab_Openfield',
                   'treadmill_ole': 'Kiehn_Lab_Treadmill',
                   'MackenzieMausHaus': 'MausHaus',
                   'daniel3mouse': 'TriMice',
                   'dlc-openfield': 'DLC_Openfield',
                   'TwoWhiteMice_GoldenLab': 'WhiteMice',
                   'ChanLab': 'BlackMice'
                   }

    dataset_decomposition.rename(index=rename_dict, inplace=True)

    def plot_horse_data_efficiency():
        dataset_size = 1469
        temp = pd.read_hdf('../data/Figure2/horse_ratios.h5')
        rename_dict = {'baseline': 'ImageNet transfer learning',
                       'zeroshot': 'SA + Zeroshot',
                       'super_remove_head': 'SA + Randomly Initialized Decoder',
                       'unbalanced_memory_replay_threshold_0.8_700000': 'SA + Memory Replay',
                       'unbalanced_naive_finetune_700000': 'SA + Naive Fine-tuning'}
        temp.rename(index=rename_dict, inplace=True)
        unbalanced_zeroshot = temp.loc['unbalanced_zeroshot', '1000']  # .mean(axis=0)

        unbalanced_zeroshot = unbalanced_zeroshot.loc[['shuffle1_best', 'shuffle2_best', 'shuffle3_best']]

        print(unbalanced_zeroshot)

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

        print(unbalanced_zeroshot['NE_iid'])

        ax1.axhline(unbalanced_zeroshot['NE_iid'].mean(), ls=':', lw=2)

        NE_iid_min = np.min(unbalanced_zeroshot['NE_iid'])
        NE_iid_max = np.max(unbalanced_zeroshot['NE_iid'])

        ax1.axhspan(NE_iid_min, NE_iid_max, facecolor='grey', alpha=0.5)

        NE_ood_min = np.min(unbalanced_zeroshot['NE_ood'])
        NE_ood_max = np.max(unbalanced_zeroshot['NE_ood'])

        ax2.axhspan(NE_ood_min, NE_ood_max, facecolor='grey', alpha=0.5)

        ax2.axhline(unbalanced_zeroshot['NE_ood'].mean(), ls=':', lw=2)
        pal = 'magma_r'

        sns.pointplot(data=df_masked, x="level_1", y="NE_iid", ax=ax1, hue='level_0', palette=pal, ci=None, )
        sns.stripplot(data=df_masked, x="level_1", y="NE_iid", ax=ax1, hue='level_0', palette=pal, alpha=.5)
        sns.pointplot(data=df_masked, x="level_1", y="NE_ood", ax=ax2, hue='level_0', palette=pal, ci=None)
        sns.stripplot(data=df_masked, x="level_1", y="NE_ood", ax=ax2, hue='level_0', palette=pal, alpha=.5)

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
        fig.savefig('Figure2c_horse-10.png', dpi=800, bbox_inches='tight', pad_inches=0.05)

    def plot_rodent_data_efficiency():
        temp = pd.read_hdf('../data/Figure2/rodent_ratios.h5')
        # print (temp.to_string())
        dataset_size = dataset_decomposition.loc['super_quadruped'].loc['iRodents'].loc['num_images']
        unbalanced_zeroshot = temp.loc['unbalanced_zeroshot', '700000']  # .mean(axis=0)
        drop_list = ['unbalanced_zeroshot',
                     # 'balanced_zeroshot'
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
        # print (df_masked)
        sns.pointplot(data=df_masked, x="level_1", y="NE", ax=ax1, hue='level_0', palette=pal,
                      hue_order=['ImageNet transfer learning', 'SA + Randomly Initialized Decoder',
                                 'SA + Memory Replay', 'SA + Naive Fine-tuning'], ci=None)
        sns.stripplot(data=df_masked, x="level_1", y="NE", ax=ax1, hue='level_0', palette=pal, alpha=.5,
                      hue_order=['ImageNet transfer learning', 'SA + Randomly Initialized Decoder',
                                 'SA + Memory Replay', 'SA + Naive Fine-tuning'])
        ax1.legend().remove()
        ax1.set_xlabel('')
        ax1.set_ylabel('Normalized error')
        ax1.set_ylim(0, 6)
        sns.despine(top=True, right=True, ax=ax1)
        fig.supxlabel('Number of fine-tuning images', y=0.05, x=0.5125)
        fig.savefig('Figure2c-iRodent.png', dpi=600, bbox_inches='tight', pad_inches=0.05)

    def plot_openfield_data_efficiency():
        temp = pd.read_hdf('../data/Figure2/openfield_ratios.h5')

        dataset_size = dataset_decomposition.loc['super_topview'].loc['DLC_Openfield'].loc['num_images']
        unbalanced_zeroshot = temp.loc['unbalanced_zeroshot', '600000']  # .mean(axis=0)
        print ('openfield data efficiency')
        print(unbalanced_zeroshot.mean())

        drop_list = ['balanced_memory_replay_threshold_0.0_750000',
                     'unbalanced_zeroshot',
                     'balanced_zeroshot',
                     'balanced_super_remove_head_750000',
                     'balanced_memory_replay_threshold_0.8_snapshot_700000',
                     'balanced_memory_replay_threshold_0.8_750000',
                     'unbalanced_memory_replay_750000']
        temp.drop(drop_list, inplace=True)

        print(temp.groupby(level=(0, 1)).mean())
        # print (temp.to_string())
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
        fig.savefig('Figure2c-DLC-openfield.png', dpi=600, bbox_inches='tight', pad_inches=0.05)

    plot_horse_data_efficiency()
    plot_rodent_data_efficiency()
    plot_openfield_data_efficiency()


def plot_figure2i():
    df = pd.read_hdf('../data/Figure2/video_adaptation_scores.h5')

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
        # scalebar = AnchoredScaleBar(
        #    axs[i].transData,
        #    sizey=400,
        #    labely='400 px$^2$',
        #    bbox_to_anchor=(825, 675),
        #    barcolor='k'
        # )

        axs[i].set_xticklabels([])
        axs[i].xaxis.set_tick_params(length=0)
        axs[i].set_yticklabels([])
        axs[i].yaxis.set_tick_params(length=0)
        # axs[i].add_artist(scalebar)
        sns.despine(ax=axs[i], top=True, right=True, left=True, bottom=True)
        axs[0].legend(frameon=False, loc='lower right')
    fig.savefig('Figure2i.png', dpi=800, bbox_inches='tight', pad_inches=0.05)


def plot_figure2f():
    # %%
    with open('../data/Figure2/ood_mice_zeroshot.pickle', 'rb') as f:
        pickle_obj = pickle.load(f)
    # %%
    proj_roots = list(pickle_obj.keys())
    proj_nicknames = ['smear mouse', 'ood maushaus', 'golden mouse']
    method_colors = plt.cm.get_cmap('magma_r', 6)

    fig, axes = plt.subplots(3, figsize=(4, 6))

    remove_nan = lambda x: x[~np.isnan(x)]
    dfs = []
    for i, proj_root in enumerate(pickle_obj):
        with_spatial_pyramid = np.nanmean(pickle_obj[proj_root]['with_spatial_pyramid']['RMSE'], axis=(1, 2))
        with_spatial_pyramid = remove_nan(with_spatial_pyramid)
        without_spatial_pyramid = np.nanmean(pickle_obj[proj_root]['without_spatial_pyramid']['RMSE'], axis=(1, 2))
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
    fig.savefig('Figure2f.png', dpi=600, bbox_inches='tight', pad_inches=0.05)


plot_figure2b()
plot_figure2c()
plot_figure2i()
plot_figure2f()
