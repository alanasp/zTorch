import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from mpl_toolkits.mplot3d import Axes3D

plt_color_codes = 'bgrcmykw'

results_folder = 'results/'

# simulation identifiers for which we generate graphs
simulations = [(50, 100), (50, 500), (50, 1000), (1, 500), (200, 500), (500, 500)]

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
            avgs = list()
            rolling_avg = 0.0
            m = 0
            for line in data_file:
                nums = line.split(' ')
                val = int(nums[1])

                if m < 20:
                    rolling_avg = (rolling_avg * m + val) / (m + 1)
                    m += 1
                else:
                    rolling_avg += (1/m) * (val - y[-m])
                x.append(int(nums[0]))
                y.append(val)
                avgs.append(rolling_avg)
            if data_name == 'surv_epoch_lengths':
                y = np.array(y)
                y = (100 * y / y[0]) - 100
            x = np.array(x)
            x = 100 * x / x[-1]
            results[data_name][sim] = (x, avgs)

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
        'sim_ids': [0, 1, 2],
        'get_label': lambda sim: 'VNFs (I) = {}'.format(sim[1])
    },
    1: {
        'sim_ids': [3, 4, 1, 5],
        'get_label': lambda sim: 'σ = {}'.format(sim[0]/100)
    }
}

for group_id in plot_settings_by_sims:
    for i in range(len(data_names)):
        plt.subplot(len(plot_settings_by_sims), len(data_names), group_id*len(data_names) + i+1)
        data_name = data_names[i]
        lines = list()
        for i in plot_settings_by_sims[group_id]['sim_ids']:
            plot_data = list()
            sim_set = plot_settings_by_sims[group_id]
            sim = simulations[i]
            plot_data.append(results[data_name][sim][0])
            plot_data.append(results[data_name][sim][1])
            plot_data.append(plt_color_codes[i])

            line, = plt.plot(*plot_data, label=plot_settings_by_sims[group_id]['get_label'](sim))
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
