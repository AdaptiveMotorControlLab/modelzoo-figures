library(arrow)
library(lme4)
library(ggplot2)
library(dplyr)
library(emmeans)

df <- arrow::read_feather('data/rmse_eff.feather')
df$method <- factor(df$method, ordered = FALSE)
df <- filter(df, method %in% c('ImageNet transfer learning', 'SA + Memory Replay', 'zeroshot'))
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

emm <- emmeans(mod4, pairwise ~ frac * method)
contrast(emm, interaction = "pairwise")
