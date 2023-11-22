library(lme4)
library(emmeans)
library(dplyr)
library(flextable)
library(xtable)

# Read in the data
data <- read.csv("data/Extended_Figure4/horsejitter.csv")

null_model <- lmer(jitter ~ 1 + (1 | video), data = data)
fixed_effects_model <- lmer(jitter ~ cond + method + (1 | video), data = data)
interaction_model <- lmer(jitter ~ cond * method + (1 | video), data = data)
random_slope_model <- lmer(jitter ~ cond + method + (1 + cond | video), data = data)
full_model <- lmer(jitter ~ cond * method + (1 + cond | video), data = data)
anova(null_model, fixed_effects_model, interaction_model, full_model)

aov <- anova(interaction_model)
print(aov)
emm <- emmeans(interaction_model, pairwise ~ cond | method)

adjusted_results <- summary(emm$contrasts, adjust = "tukey")
df <- as_tibble(data)
effect_sizes <- eff_size(emm, sigma = sigma(interaction_model), edf = df.residual(interaction_model), type='d')
effect_sizes_df <- as.data.frame(effect_sizes)

adjusted_results_df <- as.data.frame(adjusted_results)
adjusted_results_df$eff.size <- effect_sizes_df$effect.size

latex_table <- xtable(adjusted_results_df)
print(latex_table, type = "latex")


ft <- flextable(adjusted_results_df)
ft <- set_formatter(
  ft,
  p.value = function(x) format(x, digits = 2, nsmall = 2),
  t.ratio = function(x) format(x, digits = 2, nsmall = 2),
  eff.size = function(x) format(x, digits = 2, nsmall = 2),
  estimate = function(x) format(x, digits = 2, nsmall = 2),
  SE = function(x) format(x, digits = 2, nsmall = 2)
)
ft <- autofit(ft)
ft
