library(lme4)
library(dplyr)
library(lmerTest)
library(emmeans)
library(flextable)
library(xtable)

# Read in the data
data <- read.csv("data/dropjitter.csv")
data_jitter <- filter(data, metric == 'jittering')
data_drop <- filter(data, metric == 'keypoint_dropping')

full_model_jitter <- lmer(val ~ video * cond + (1 | video), data = data_jitter)
aov <- anova(full_model_jitter)
print(xtable(aov))

emm <- emmeans(full_model_jitter, pairwise ~ cond | video)
effect_sizes <- eff_size(emm, sigma = sigma(full_model_jitter), edf = df.residual(full_model_jitter))
effect_sizes_df <- as.data.frame(effect_sizes)

adjusted_results <- summary(emm$contrasts, adjust = "tukey")
adjusted_results_df <- as.data.frame(adjusted_results)
adjusted_results_df$eff.size <- effect_sizes_df$effect.size

new_names <- c(m3v1mp4 = "DLC-Openfield", maushaus_short = "MausHaus", smear_mouse = "Smear Lab", golden_mouse = "Golden Lab", black_dog = "Dog", elf = "Elk", horse = "Horse")
adjusted_results_df$video <- new_names[adjusted_results_df$video]

latex_table <- xtable(adjusted_results_df)
print(latex_table, type = "latex")


full_model_drop <- lmer(val ~ video * cond + (1 | video), data = data_drop)
aov <- anova(full_model_drop)
print(xtable(aov))

emm <- emmeans(full_model_drop, pairwise ~ cond | video)
effect_sizes <- eff_size(emm, sigma = sigma(full_model_drop), edf = df.residual(full_model_drop))
effect_sizes_df <- as.data.frame(effect_sizes)

adjusted_results <- summary(emm$contrasts, adjust = "tukey")
adjusted_results_df <- as.data.frame(adjusted_results)
adjusted_results_df$eff.size <- effect_sizes_df$effect.size

new_names <- c(m3v1mp4 = "DLC-Openfield", maushaus_short = "MausHaus", smear_mouse = "Smear Lab", golden_mouse = "Golden Lab", black_dog = "Dog", elf = "Elk", horse = "Horse")
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
