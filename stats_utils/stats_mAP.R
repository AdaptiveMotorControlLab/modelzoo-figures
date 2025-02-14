library(arrow)
library(lme4)
library(lmerTest)
library(ggplot2)
library(dplyr)
library(emmeans)
library(xtable)

df <- arrow::read_feather('data/Figure2/results/stats_quad.feather')
df$cond <- factor(df$cond, ordered = FALSE)
df$dataset <- factor(df$dataset, ordered = FALSE)
df$shuffle <- factor(df$shuffle, ordered = FALSE)
df$frac <- factor(df$frac, ordered = TRUE)

df %>%
  ggplot(aes(x=frac, y=map, group=interaction(dataset, cond), color=dataset, linetype=cond)) +
  geom_line() +
  geom_point() +
  theme_classic()

for (dataset_ in c("horse10", "rodent")) {
  print(dataset_)
  df2 <- filter(df, dataset == dataset_)

  model = lmer(map ~ cond * frac + (1 | shuffle), data = df2)
  anova_table <- anova(model, ddf="Kenward-Roger")
  anova_latex_table <- xtable(anova_table)
  print(anova_latex_table)
  emm <- emmeans(model, pairwise ~ cond | frac)

  adjusted_results <- summary(emm$contrasts, adjust = "tukey")
  effect_sizes <- eff_size(emm, sigma = sigma(model), edf = df.residual(model), type='d')
  effect_sizes_df <- as.data.frame(effect_sizes)

  adjusted_results_df <- as.data.frame(adjusted_results)
  adjusted_results_df$eff.size <- effect_sizes_df$effect.size

  latex_table <- xtable(adjusted_results_df)
  print(latex_table, type = "latex")
}
