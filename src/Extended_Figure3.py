# # Extended Data Figure 3 - Ye et al. 2023

# - import dependencies and load the data

import matplotlib.pyplot as plt
import matplotlib
plt.rcParams["figure.figsize"] = (13,10)
import matplotlib.pyplot as mpl
import json
mpl.rcParams['font.size'] = 25
plt.rcParams["figure.autolayout"] = True
plt.rcParams["figure.autolayout"] = True


def plot_extended_data_figure3b():

    zeroshot_bottomup_quadruped_awaood = ''
    zeroshot_topdown_quadruped_awaood = ''
    zeroshot_bottomup_quadruped_irodent = ''
    zeroshot_topdown_quadruped_irodent = ''


    with open('../data/Extended_Figure3/HRNet_summary_plot.json', 'r') as f:
        results = json.load(f)


    zeroshot =  results['zeroshot']
    pseudo = results['pseudo']
    data_1percent = results['data_1percent']

    datasets = ['AwA_ood', 'iRodent', 'Openfield', 'TriMouse']
    weight_types = ['ImageNet', 'SuperAnimal']
    baselines = ['top-down']
    print (datasets)


    # In[5]:


    xticks = [0, 1, 2]
    xtick_labels = ['zeroshot', 'psuedo', '1%']

    SA_awa_ood = [zeroshot['SuperAnimal']['AwA_ood']['top-down'],
                  pseudo['SuperAnimal']['AwA_ood']['top-down'],
                  data_1percent['SuperAnimal']['AwA_ood']['top-down']
                 ]

    SA_rodent = [zeroshot['SuperAnimal']['iRodent']['top-down'],
                  pseudo['SuperAnimal']['iRodent']['top-down'],
                  data_1percent['SuperAnimal']['iRodent']['top-down']
                 ]

    SA_openfield = [zeroshot['SuperAnimal']['Openfield']['top-down'],
                  pseudo['SuperAnimal']['Openfield']['top-down'],
                  data_1percent['SuperAnimal']['Openfield']['top-down']
                 ]

    SA_trimouse = [zeroshot['SuperAnimal']['TriMouse']['top-down'],
                  pseudo['SuperAnimal']['TriMouse']['top-down'],
                  data_1percent['SuperAnimal']['TriMouse']['top-down']
                 ]




    IM_awa_ood = [zeroshot['ImageNet']['AwA_ood']['top-down'],
                  pseudo['ImageNet']['AwA_ood']['top-down'],
                  data_1percent['ImageNet']['AwA_ood']['top-down']
                 ]

    IM_rodent = [zeroshot['ImageNet']['iRodent']['top-down'],
                  pseudo['ImageNet']['iRodent']['top-down'],
                  data_1percent['ImageNet']['iRodent']['top-down']
                 ]

    IM_openfield = [zeroshot['ImageNet']['Openfield']['top-down'],
                  pseudo['ImageNet']['Openfield']['top-down'],
                  data_1percent['ImageNet']['Openfield']['top-down']
                 ]

    IM_trimouse = [zeroshot['ImageNet']['TriMouse']['top-down'],
                  pseudo['ImageNet']['TriMouse']['top-down'],
                  data_1percent['ImageNet']['TriMouse']['top-down']
                 ]



    fig, ax1 = plt.subplots()

    bottomup_linestyle = 'solid'
    topdown_linestyle = 'dashed'
    SA_marker = 'o'
    SA_alpha = 1
    IM_alpha = 0.3
    IM_marker = 'x'
    datasets = ['AwA_ood', 'iRodent', 'Openfield', 'TriMouse']
    dataset_colors = ['blueviolet', 'red', 'blue', 'cyan']
    markersize=15
    linewidth=7

    ax1.plot(range(3), SA_awa_ood, label ='SuperAnimal AwA-OOD', linewidth = linewidth,marker = SA_marker,
             linestyle = topdown_linestyle, alpha = SA_alpha, markersize = markersize+3, color = 'blueviolet')

    ax1.plot(range(3), IM_awa_ood, label ='ImageNet AwA-OOD', linewidth = linewidth, marker = IM_marker,
             linestyle = topdown_linestyle, alpha = IM_alpha, markersize = markersize+3, color = 'blueviolet')

    ax1.plot(range(3), SA_rodent, label ='SuperAnimal iRodent', linewidth = linewidth,marker = SA_marker,
             linestyle = topdown_linestyle, alpha = SA_alpha, markersize = markersize+3, color = 'tomato')

    ax1.plot(range(3), IM_rodent, label ='ImageNet iRodent', linewidth = linewidth, marker = IM_marker,
             linestyle = topdown_linestyle, alpha = IM_alpha, markersize = markersize+3, color = 'tomato')

    ax1.plot(range(3), SA_openfield, label = 'SuperAnimal Openfield',  linewidth = linewidth, color = 'blue', marker = SA_marker,
             linestyle = topdown_linestyle, alpha = SA_alpha, markersize = markersize+3)

    ax1.plot(range(3), IM_openfield, label = 'ImageNet Openfield', linewidth = linewidth, color = 'blue', marker = IM_marker,
             linestyle = topdown_linestyle, alpha = IM_alpha, markersize = markersize+3)

    ax1.plot(range(3), SA_trimouse, label = 'SuperAnimal TriMouse',  linewidth = linewidth, color = 'darkturquoise', marker = SA_marker,
             linestyle = topdown_linestyle, alpha = SA_alpha, markersize = markersize+3)

    ax1.plot(range(3), IM_trimouse, label = 'ImageNet TriMouse', linewidth = linewidth, color = 'darkturquoise', marker = IM_marker,
             linestyle = topdown_linestyle, alpha = IM_alpha, markersize = markersize+3)



    ax1.set_xticks([0,1,2])
    ax1.set_xticklabels(['zeroshot',
                         'pseudo labeled',
                         '1% data'
                         ])
    ax1.legend(bbox_to_anchor=(3, 0.8))
    ax1.set_ylabel('mAP')

    ax1.spines['right'].set_visible(False)
    ax1.spines['top'].set_visible(False)
    
    
    fig.savefig('Extended_Data_Figure3b-summary.png', dpi=600, bbox_inches='tight', pad_inches=0.5)


plot_extended_data_figure3b()    


