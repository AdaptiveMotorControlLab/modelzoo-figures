# %%
import pandas as pd
import pingouin as pg

df = pd.read_hdf('hrnet_w32_adaptation_results_quadruped80k.h5').reset_index()
df.columns = ['video', 'method', 'cond', 'mAP']
df = df[(df.method != 'self_pacing_fix_BN') & (df.method != 'video_adaptation_relax_BN')]
df['mAP'] *= 100
df['method'] = df['method'].map(
    {
        'kalman_filter': 'Kalman filter',
        'self_pacing_relax_BN': 'Self-pacing',
        'video_adaptation_fix_BN': 'Video adaptation',
    },
)
# %%
df_ = df[(df.cond == 'adaptation_gain')]
aov = pg.rm_anova(df_, dv='mAP', within='method', subject='video')
print(aov)
post_hocs = pg.pairwise_tests(df_, dv='mAP', within='method', subject='video', effsize='cohen')
print(post_hocs.round(3))
# %%
df_ = df[(df.cond == 'robustness_gain')]
tt = pg.ttest(df_[df_.method == 'Self-pacing'].mAP, df_[df_.method == 'Video adaptation'].mAP, paired=True)
print(tt)