library(arrow)

df <- arrow::read_feather('data/rmse_dists.feather')
df$dataset <- factor(df$dataset, ordered = FALSE)

for (dataset_ in unique(df$dataset)) {
  df_ <- filter(df, dataset == dataset_)
  print(ks.test(df_[df_$cond == 'without',]$RMSE, df_[df_$cond == 'with',]$RMSE, alternative = "l"))
}
