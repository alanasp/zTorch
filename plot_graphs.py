import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from mpl_toolkits.mplot3d import Axes3D

plt_color_codes = 'bgrcmykw'

results_folder = 'results/'

# simulation identifiers for which we generate graphs
simulations = [(10, 750), (10, 1000), (10, 1250), (6, 1000), (8, 1000), (12, 1000)]

data_names = ['num_aff_groups', 'mon_indices', 'surv_epoch_lengths']

results = dict()

for data_name in data_names:
    results[data_name] = dict()
    for i in range(len(simulations)):
        sim = simulations[i]
        with open(results_folder + '{}_{}_{}'.format(data_name, sim[0], sim[1])) as data_file:
            n = int(data_file.readline())
            x = list()
            y = list()
            rolling_avg = 0.0
            m = 0
            for line in data_file:
                nums = line.split(' ')
                val = int(nums[1])
                rolling_avg = (rolling_avg*m + val)/(m+1)
                m += 1
                x.append(int(nums[0]))
                y.append(rolling_avg)
            if data_name == 'surv_epoch_lengths':
                y = np.array(y)
                y = (100 * y / y[0]) - 100
            x = np.array(x)
            x = 100 * x / x[-1]
            results[data_name][sim] = (x, y)

plot_settings_by_data = {
    'num_aff_groups': {
        'xlabel': 'Simulation Time',
        'ylabel': 'Number of VNF Affinity Groups (N)'
    },
    'mon_indices': {
        'xlabel': 'Simulation Time',
        'ylabel': 'Monitoring Frequency Index (δ)',
        'ylim': (1, 5)
    },
    'surv_epoch_lengths': {
        'xlabel': 'Simulation Time',
        'ylabel': 'Surveillance Epoch Increase (ω) [%]'
    }
}

plot_settings_by_sims = {
    0: {
        'sims': [(10, 750), (10, 1000), (10, 1250)],
        'labels': ['VNFs (I) = 750', 'VNFs (I) = 1000', 'VNFs (I) = 1250']
    },
    1: {
        'sims': [(6, 1000), (10, 1000), (8, 1000), (12, 1000)],
        'labels': ['σ = 0.06', 'σ = 0.08', 'σ = 0.1', 'σ = 0.12']
    }
}

for group_id in plot_settings_by_sims:
    for i in range(len(data_names)):
        plt.subplot(len(plot_settings_by_sims), len(data_names), group_id*len(data_names) + i+1)
        data_name = data_names[i]
        lines = list()
        for i in range(len(plot_settings_by_sims[group_id]['sims'])):
            plot_data = list()
            sim = plot_settings_by_sims[group_id]['sims'][i]
            plot_data.append(results[data_name][sim][0])
            plot_data.append(results[data_name][sim][1])
            plot_data.append(plt_color_codes[i])

            line, = plt.plot(*plot_data, label=plot_settings_by_sims[group_id]['labels'][i])
            lines.append(line)
        settings = plot_settings_by_data[data_name]
        axes = plt.gca()
        if 'xlabel' in settings:
            plt.xlabel(settings['xlabel'])
        if 'ylabel' in settings:
            plt.ylabel(settings['ylabel'])
        if 'ylim' in settings:
            axes.set_ylim(settings['ylim'])
        axes.yaxis.set_major_locator(MaxNLocator(integer=True))
        plt.legend(handles=lines)


plt.show()
