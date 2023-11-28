library(lme4)
library(dplyr)
library(lmerTest)
library(emmeans)
library(flextable)
library(xtable)

# Read in the data
df <- read.csv("data/Extended_Figure7/TD_dlc_openfield_memory_replay_ablation_study.csv")
df$method <- factor(df$method, ordered = FALSE)
df$pretrain_model <- factor(df$pretrain_model, ordered = FALSE)
df$shuffle <- factor(df$shuffle, ordered = FALSE)
df$train_ratio <- factor(df$train_ratio, ordered = TRUE)

model = lmer(keypoint_dropping ~ method * train_ratio + (1 | shuffle), data = df)
anova_table <- anova(model, ddf="Kenward-Roger")
anova_latex_table <- xtable(anova_table)
print(anova_latex_table)

emm <- emmeans(model, pairwise ~ method | train_ratio)
adjusted_results <- summary(emm$contrasts, adjust = "tukey")
effect_sizes <- eff_size(emm, sigma = sigma(model), edf = df.residual(model), type='d')
effect_sizes_df <- as.data.frame(effect_sizes)

adjusted_results_df <- as.data.frame(adjusted_results)
adjusted_results_df$eff.size <- effect_sizes_df$effect.size

latex_table <- xtable(adjusted_results_df)
print(latex_table, type = "latex")
