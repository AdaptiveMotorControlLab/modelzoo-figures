library(lme4)
library(emmeans)
library(dplyr)
library(flextable)
library(xtable)

# Read in the data
df <- arrow::read_feather('data/Figure1/stats_topview.feather')
df$cond <- factor(df$cond, ordered = FALSE)
df$dataset <- factor(df$dataset, ordered = FALSE)
df$shuffle <- factor(df$shuffle, ordered = FALSE)
df$frac <- factor(df$frac, ordered = TRUE)

for (dataset_ in c("trimouse", "dlc_openfield")) {
  df2 <- filter(df, dataset == dataset_)

  model = lmer(rmse ~ cond * frac + (1 | shuffle), data = df2)
  anova_table <- anova(model, ddf="Kenward-Roger")
  print(anova_table)
  emm <- emmeans(model, pairwise ~ cond | frac)

  adjusted_results <- summary(emm$contrasts, adjust = "tukey")
  effect_sizes <- eff_size(emm, sigma = sigma(model), edf = df.residual(model), type='d')
  effect_sizes_df <- as.data.frame(effect_sizes)

  adjusted_results_df <- as.data.frame(adjusted_results)
  adjusted_results_df$eff.size <- effect_sizes_df$effect.size

  latex_table <- xtable(adjusted_results_df)
  print(latex_table, type = "latex")
}
