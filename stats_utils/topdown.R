library(lme4)
library(emmeans)
library(dplyr)
library(flextable)
library(xtable)

# Read in the data
df <- arrow::read_feather('data/Figure1/stats_topview.feather')
df$method <- factor(df$cond, ordered = FALSE)
df <- filter(df, method %in% c('ImageNet transfer-learning', 'SuperAnimal memory-replay', 'SuperAnimal zeroshot'))
df$dataset <- factor(df$dataset, ordered = FALSE)
df$shuffle <- factor(df$shuffle, ordered = FALSE)
df$frac <- factor(df$frac, ordered = TRUE)

mod1 <- lmer(rmse ~ (1|dataset), data=df, REML=F)  # Null model
mod2 <- lmer(rmse ~ method + (1|dataset), data=df, REML=F)  # Add random intercept
mod3 <- lmer(rmse ~ method * frac + (1|dataset), data=df, REML=F)  # Add interaction term
mod4 <- lmer(rmse ~ method * frac + (1 + frac|dataset), data=df, REML=F)  # Add random slope
mod5 <- lmer(rmse ~ method * frac + (1 + frac|dataset) + (1 | shuffle), data=df, REML=F)  # Add shuffle's crossed random effect

anova(mod1, mod2, mod3, mod4, mod5)
emm <- emmeans(mod4, pairwise ~ method | frac)
emm$contrasts %>%
  summary(infer = TRUE)
eff_size(emm, sigma = sigma(mod4), edf = df.residual(mod4))

for (dataset_ in c("trimouse", "dlc_openfield")) {
  df2 <- filter(df, dataset == dataset_)

  model = lmer(rmse ~ method * frac + (1 | shuffle), data = df2)
  anova_table <- anova(model, ddf="Kenward-Roger")
  print(anova_table)
  emm <- emmeans(model, pairwise ~ method | frac)

  adjusted_results <- summary(emm$contrasts, adjust = "tukey")
  effect_sizes <- eff_size(emm, sigma = sigma(model), edf = df.residual(model), type='d')
  effect_sizes_df <- as.data.frame(effect_sizes)

  adjusted_results_df <- as.data.frame(adjusted_results)
  adjusted_results_df$eff.size <- effect_sizes_df$effect.size

  latex_table <- xtable(adjusted_results_df)
  print(latex_table, type = "latex")
}
