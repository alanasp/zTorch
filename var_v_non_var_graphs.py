import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from mpl_toolkits.mplot3d import Axes3D

plt_color_codes = 'bgrcmykw'

std = [0.01, 0.1, 0.3]
num_prof = 1000

filename = 'saved_exp_data/varied_v_non_varied_{}_{}_{}_{}'.format(int(100*std[0]), int(100*std[1]),
                                                                   int(100*std[2]), num_prof)


with open(filename, 'r') as file:
    file.readline()
    file.readline()
    time = np.array(list(map(float, file.readline().split(' '))))
    time = time/time[-1]*100
    aff_groups_non_var = np.array(list(map(float, file.readline().split(' '))))
    aff_groups_var = np.array(list(map(float, file.readline().split(' '))))

#fit line on aff_groups_non_var vs aff_groups_var
fit_params = np.polyfit(aff_groups_non_var, aff_groups_var, 1, full=True)
coeffs = fit_params[0]
residuals = fit_params[1]
fit_line = np.poly1d(coeffs)
print(residuals)

#perfect correspondence would have slope of 1
mid_point = (aff_groups_non_var[0] + aff_groups_non_var[-1])/2.0
perfect_coeffs = [1, fit_line(mid_point)-mid_point]
perfect_line = np.poly1d(perfect_coeffs)
print(fit_line(mid_point), perfect_line(mid_point), mid_point)
var_vf_hat = np.var(perfect_line(aff_groups_non_var))
var_vf = np.var(aff_groups_var)
r2 = 1 - var_vf_hat/var_vf
print(var_vf_hat, var_vf, r2)


plt_non_var_line, = plt.plot(time, aff_groups_non_var, label='Non-varied monitoring frequency, std={}'.format(std))
plt_var_line, = plt.plot(time, aff_groups_var, label='Varied monitoring frequency, std={}'.format(std))
plt.legend(handles=[plt_non_var_line, plt_var_line])
plt.xlabel('Simulation Time')
plt.ylabel('Number of VNF Affinity Groups (N)')
#plt.show()

plt_fit_line, = plt.plot(aff_groups_non_var, fit_line(aff_groups_non_var), color='g',
                         label='Fitted line, slope={:.2f}'.format(coeffs[0]))
plt_perfect_line, = plt.plot(aff_groups_non_var, perfect_line(aff_groups_non_var), color='r', linestyle='dashed',
                             label='Perfect correspondence line, slope=1.00')
plt_points = plt.scatter(aff_groups_non_var, aff_groups_var, color='b', label='Observed values'.format(coeffs[0]))
plt.legend(handles=[plt_fit_line, plt_perfect_line, plt_points])
plt.xlabel('Non-varied frequency affinity groups')
plt.ylabel('Varied frequency affinity groups')
#plt.show()
