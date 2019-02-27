import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import ztorch_simulation

plt_color_codes = 'bgrcmykw'


num_vnf_profiles = 1000
num_vnf_kpis = 3
num_base_profiles = len(ztorch_simulation.base_vnf_profiles['high'])
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

sim008 = ztorch_simulation.Simulation(std=0.08, output_files_prefix=True)
sim008 = ztorch_simulation.Simulation(init_profiles=1000, std=0.08, input_files_prefix=True)

exit(0)

sim010 = ztorch_simulation.Simulation(std=0.1)
sim012 = ztorch_simulation.Simulation(std=0.12)
sims = []

num_aff_groups = []
for sim in sims:
    num_aff_groups.append(sim.run_sim())


steps, centres, aff_groups, points = sim010.run_ekm()
vnf_groups = ztorch_simulation.group_points(points, aff_groups)

plot_ekm_results = True
plot_sim_results = False

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
    for aff in num_aff_groups:
        plt.plot(aff)
    plt.show()