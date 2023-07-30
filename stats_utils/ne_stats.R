library(arrow)
library(lme4)
library(ggplot2)
library(dplyr)
library(emmeans)

df <- arrow::read_feather('data/metrics.feather')
df$method <- factor(df$method, ordered = FALSE)
df <- filter(df, method != 'zeroshot')
df$dataset <- factor(df$dataset, ordered = FALSE)
df$shuffle <- factor(df$shuffle, ordered = FALSE)
df$n <- factor(df$n, ordered = TRUE)

df %>%
  ggplot(aes(x=n, y=metric, group=interaction(dataset, method), color=dataset, linetype=method)) +
  geom_line() +
  geom_point() +
  theme_classic()

lmems <- df %>% group_by(dataset) %>% do(model = lmer(metric ~ method * n + (1 | shuffle), data = .))

for (i in 1:dim(lmems)[1]) {
  temp <- lmems[i,]
  print('----------')
  print(temp$dataset)
  lmem <- temp$model[[1]]
  print(anova(lmem, ddf="Kenward-Roger"))
  emm <- emmeans(lmem, pairwise ~ method | n)
  print(emm$contrasts %>%
    summary(infer = TRUE))
  print(eff_size(emm, sigma = sigma(lmem), edf = df.residual(lmem)))
}
