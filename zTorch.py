import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import data_gen


base_vnf_profiles = {
     'low':  {'MME':  [17.7, 15.9, 5.8],
              'SGW':  [0.7, 0.3, 0.14],
              'HSS':  [0.9, 1.1, 0.7],
              'PCRF': [1.2, 0.6, 0.5],
              'PGW':  [1.7, 2.1, 0.8]},

     'high': {'MME':  [2.9, 3.8, 1.9],
              'SGW':  [79.1, 3.3, 91.2],
              'HSS':  [2.9, 4.5, 1.3],
              'PCRF': [1.9, 3.9, 0.9],
              'PGW':  [53.1, 37.2, 92]}
}

plt_color_codes = 'bgrcmykw'


num_vnf_profiles = 1000
num_vnf_kpis = 3
num_base_profiles = len(base_vnf_profiles['high'])
surveilance_epoch = 500
beta = 0.5
phi = 0.5
psi = 0.9
vnf_profile_std = 0.1
mon_interval = [2, 5, 10, 20, 50]
sim_time = 10**7

k_means_granularity = 0.001



#print(base_vnf_profiles['high'])
#init_profiles = data_gen.gen_init_profiles(base_vnf_profiles['high'], 1)
#print(init_profiles)

sim008 = data_gen.Simulation(std=0.08)
sim010 = data_gen.Simulation(std=0.1)
sim012 = data_gen.Simulation(std=0.12)
#steps, centres, aff_groups, points = sim.run_ekm()
num_aff_groups = [sim008.run_sim(), sim010.run_sim(), sim012.run_sim()]

#vnf_groups = data_gen.group_points(points, aff_groups)

plot_ekm_results = False
plot_sim_results = True

if plot_ekm_results:
    fig = plt.figure()
    ax = fig.gca(projection='3d')

    avail_colors = set(plt_color_codes)

    for gid in vnf_groups:
        group = vnf_groups[gid]
        if len(avail_colors) > 0:
            color = avail_colors.pop()
            ax.scatter(*group.T, c=color, alpha=0.3)

    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.set_zlim(0, 100)
    plt.show()

if plot_sim_results:
    plt.plot(num_aff_groups[0])
    plt.plot(num_aff_groups[1])
    plt.plot(num_aff_groups[2])
    plt.show()