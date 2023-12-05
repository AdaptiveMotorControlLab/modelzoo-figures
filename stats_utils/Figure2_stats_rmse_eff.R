# # Statistics to support Figure 2 - Ye et al. 2023

library(arrow)
library(lme4)
library(lmerTest)
library(ggplot2)
library(dplyr)
library(emmeans)
library(xtable)

df <- arrow::read_feather('data/rmse_eff.feather')
df$method <- factor(df$method, ordered = FALSE)
df$dataset <- factor(df$dataset, ordered = FALSE)
df$shuffle <- factor(df$shuffle, ordered = FALSE)
df$frac <- factor(df$frac, ordered = TRUE)

df %>%
  ggplot(aes(x=frac, y=rmse, group=interaction(dataset, method), color=dataset, linetype=method)) +
  geom_line() +
  geom_point() +
  theme_classic()

mod1 <- lmer(rmse ~ (1|dataset), data=df, REML=F)  # Null model
mod2 <- lmer(rmse ~ method + (1|dataset), data=df, REML=F)  # Add random intercept
mod3 <- lmer(rmse ~ method * frac + (1|dataset), data=df, REML=F)  # Add interaction term
mod4 <- lmer(rmse ~ method * frac + (1 + frac|dataset), data=df, REML=F)  # Add random slope
mod5 <- lmer(rmse ~ method * frac + (1 + frac|dataset) + (1 | shuffle), data=df, REML=F)  # Add shuffle's crossed random effect

anova(mod1, mod2, mod3, mod4, mod5)

# df %>%
#   mutate(pred = fitted(mod4)) %>%
#   ggplot(aes(x=frac, y=pred, group=interaction(dataset, method), color=dataset, linetype=method)) +
#   geom_line(linewidth=1) +
#   theme_classic()

emm <- emmeans(mod4, pairwise ~ method | frac)
emm$contrasts %>%
  summary(infer = TRUE)
eff_size(emm, sigma = sigma(mod4), edf = df.residual(mod4))

df_openfield <- filter(df, dataset == 'openfield')
df_openfield <- df_openfield %>%
  mutate(method = recode(
    method,
    "zeroshot" = "zero-shot",
    "SA + Memory Replay" = "SuperAnimal memory-replay",
    "SA + Naive Fine-tuning" = "SuperAnimal finetune",
    "SA + Randomly Initialized Decoder" = "SuperAnimal transfer learning",
  ))

model = lmer(rmse ~ method * frac + (1 | shuffle), data = df_openfield)
anova_table <- anova(model, ddf="Kenward-Roger")
anova_latex_table <- xtable(anova_table)
print(anova_latex_table)

emm <- emmeans(model, pairwise ~ method | frac)
adjusted_results <- summary(emm$contrasts, adjust = "tukey")
effect_sizes <- eff_size(emm, sigma = sigma(model), edf = df.residual(model), type='d')
effect_sizes_df <- as.data.frame(effect_sizes)

adjusted_results_df <- as.data.frame(adjusted_results)
adjusted_results_df$eff.size <- effect_sizes_df$effect.size

latex_table <- xtable(adjusted_results_df)
print(latex_table, type = "latex")
