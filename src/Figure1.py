import matplotlib.font_manager as font_manager
import matplotlib.pyplot as plt
import pandas as pd


def plot_figure1e():
    dataset_decomposition = pd.read_hdf('../data/dataset_decomposition.h5')

    print (dataset_decomposition)
    rename_dict = {'swimming_ole': 'Kiehn Lab Swimming',
                   'openfield_ole': 'Kiehn Lab Openfield',
                   'treadmill_ole': 'Kiehn Lab Treadmill',
                   'MackenzieMausHaus': 'MausHaus',
                   'daniel3mouse': 'TriMouse',
                   'dlc-openfield': 'DLC Openfield',
                   'TwoWhiteMice_GoldenLab': 'WhiteMice',
                   'ChanLab': 'BlackMice',
                   'animalpose': 'AnimalPose',
                   'cheetah': 'AcinoSet',
                   'horse-10': 'Horse-10',
                   'iRodents': 'iRodent',
                   'stanforddogs': 'StanfordDogs',
                  }

    dataset_decomposition.rename(index=rename_dict, inplace=True)
    df = dataset_decomposition.loc['super_topview']
    x = df['num_images']
    labels = df.index.to_list()
    colors = ["033270","1368aa","4091c9","9dcee2","fedfd4","f29479","f26a4f","ef3c2d","cb1b16","65010c"]
    # colors = ["54478c","2c699a","048ba8","0db39e","16db93","83e377","b9e769","efea5a","f1c453","f29e4c", "f26a4f","ef3c2d","cb1b16"]
    colors = ["355070","6d597a","b56576","e56b6f","e88c7d","ffcdb2"]
    colors = ["1368aa","4091c9","9dcee2"][::-1] + colors + ["F2CEBA", "fee1dd", "e9c2c5", "9db0ce"]
    colors = [f'#{c}' for c in colors]
    fig, ax = plt.subplots(figsize=(4, 5))
    ax.pie(x, labels=labels, autopct=lambda n: '{:.0f}'.format(n*x.sum()/100),
           colors=colors, labeldistance=None,
           wedgeprops={'linewidth': 1, "edgecolor":"w"},
           textprops={'fontsize': 'medium', 'color': 'w', 'font': 'Arial'},
    )
    font = font_manager.FontProperties(family='Arial',
                                       style='normal', size=10)
    ax.legend(loc='center left', bbox_to_anchor=(1., 0.5), frameon=False, prop=font)
    fig.savefig('pie_topview.png', dpi=600, bbox_inches='tight', pad_inches=0.05)
    # %%
    df = dataset_decomposition.loc['super_quadruped']
    x = df['num_images']
    labels = df.index.to_list()
    colors = ["355070","6d597a","b56576","e56b6f","e88c7d","F2CEBA"]
    colors = [f'#{c}' for c in colors]
    fig, ax = plt.subplots(figsize=(4, 5))
    ax.pie(x, labels=labels, autopct=lambda n: '{:.0f}'.format(n*x.sum()/100),
           colors=colors, labeldistance=None,
           wedgeprops={'linewidth': 1, "edgecolor":"w"},
           textprops={'fontsize': 'medium', 'color': 'w', 'font': 'Arial'},
    )
    font = font_manager.FontProperties(family='Arial',
                                       style='normal', size=10)
    ax.legend(loc='center left', bbox_to_anchor=(1., 0.5), frameon=False, prop=font)
    fig.savefig('pie_quadruped.png', dpi=600, bbox_inches='tight', pad_inches=0.05)
# %%
plot_figure1e()
