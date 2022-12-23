import pandas as pd

def plot_extended_figure4e():
    results = pd.read_csv('../data/Extended_Figure4/results.csv', index_col=0)
    print(results)

def plot_extended_figure4f():
    complexity = pd.read_csv('../data/Extended_Figure4/complexity.csv', index_col=0)
    print(complexity)

plot_extended_figure4e()
# %%
plot_extended_figure4f()
