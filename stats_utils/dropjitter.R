library(lme4)
library(emmeans)
library(flextable)
library(xtable)

# Read in the data
data <- read.csv("data/dropjitter.csv")

null_model <- lmer(val ~ 1 + (1 | video), data = data)
fixed_effects_model <- lmer(val ~ cond + metric + (1 | video), data = data)
interaction_model <- lmer(val ~ cond * metric + (1 | video), data = data)
random_slope_model <- lmer(val ~ cond + metric + (1 + cond | video), data = data)
full_model <- lmer(val ~ cond * metric + (1 + cond | video), data = data)
full_model2 <- lmer(val ~ video * cond * metric + (1 | video), data = data)
full_model3 <- lmer(val ~ video * cond * metric + (1 + cond | video), data = data)
anova(null_model, fixed_effects_model, interaction_model, random_slope_model, full_model, full_model2, full_model3)

aov <- anova(full_model2)
emm <- emmeans(full_model2, pairwise ~ cond | video * metric)

adjusted_results <- summary(emm$contrasts, adjust = "tukey")
df <- as_tibble(data)
effect_sizes <- eff_size(emm, sigma = sigma(full_model2), edf = df.residual(full_model2), type='d')
effect_sizes_df <- as.data.frame(effect_sizes)

adjusted_results_df <- as.data.frame(adjusted_results)
adjusted_results_df$eff.size <- effect_sizes_df$effect.size

new_names <- c(m3v1mp4 = "DLC-Openfield", maushaus_short = "MausHaus", smear_mouse = "Smear Lab", golden_mouse = "Golden Lab")
adjusted_results_df$video <- new_names[adjusted_results_df$video]

latex_table <- xtable(adjusted_results_df)
print(latex_table, type = "latex")


ft <- flextable(adjusted_results_df)
ft <- set_formatter(
  ft,
  p.value = function(x) format(x, digits = 2, nsmall = 2),
  z.ratio = function(x) format(x, digits = 2, nsmall = 2),
  eff.size = function(x) format(x, digits = 2, nsmall = 2),
  estimate = function(x) format(x, digits = 2, nsmall = 2),
  SE = function(x) format(x, digits = 2, nsmall = 2)
)
ft <- autofit(ft)
