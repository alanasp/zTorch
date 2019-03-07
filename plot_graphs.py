import numpy as np
import matplotlib.pyplot as plt
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
            results[data_name][sim] = (x, y)

plot_descriptors = [[(10, 750), (10, 1000), (10, 1250)],
                    [(6, 1000), (10, 1000), (8, 1000), (12, 1000)]]

for row_id in range(len(plot_descriptors)):
    for i in range(len(data_names)):
        plt.subplot(len(plot_descriptors), len(data_names), row_id*len(data_names) + i+1)
        data_name = data_names[i]
        plot_data = list()
        for i in range(len(plot_descriptors[row_id])):
            sim = plot_descriptors[row_id][i]
            plot_data.append(results[data_name][sim][0])
            plot_data.append(results[data_name][sim][1])
            plot_data.append(plt_color_codes[i])
        plt.plot(*plot_data)

plt.show()
