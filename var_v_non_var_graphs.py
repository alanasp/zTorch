import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from mpl_toolkits.mplot3d import Axes3D

plt_color_codes = 'bgrcmykw'

std1 = [0.01, 0.1, 0.3]
num_prof1 = 1000

filename = 'results/varied_v_non_varied_{}_{}_{}_{}'.format(int(100*std1[0]), int(100*std1[1]),
                                                                   int(100*std1[2]), num_prof1)


with open(filename, 'r') as file:
    file.readline()
    file.readline()
    time = np.array(list(map(float, file.readline().split(' '))))
    time = time/time[-1]*100
    aff_groups_non_var1 = np.array(list(map(float, file.readline().split(' '))))
    aff_groups_var1 = np.array(list(map(float, file.readline().split(' '))))


std2 = [0.01]*80 + [0.1]*80 + [0.3]*80
num_prof2 = 1000

filename = 'results/varied_v_non_varied_{}_{}_{}_{}'.format(int(100*std2[0]), int(100*std2[1]),
                                                              int(100*std2[2]), num_prof2)


with open(filename, 'r') as file:
    file.readline()
    file.readline()
    time = np.array(list(map(float, file.readline().split(' '))))
    time = time/time[-1]*100
    aff_groups_non_var2 = np.array(list(map(float, file.readline().split(' '))))
    aff_groups_var2 = np.array(list(map(float, file.readline().split(' '))))

# 3-feature VNFs
# fit line on aff_groups_non_var vs aff_groups_var
fit_params = np.polyfit(aff_groups_non_var1, aff_groups_var1, 1, full=True)
coeffs1 = fit_params[0]
residuals = fit_params[1]
fit_line1 = np.poly1d(coeffs1)
print(residuals)

# perfect correspondence would have slope of 1
mid_point = (aff_groups_non_var1[0] + aff_groups_non_var1[-1])/2.0
perfect_coeffs1 = [1, fit_line1(mid_point)-mid_point]
perfect_line1 = np.poly1d(perfect_coeffs1)
print(fit_line1(mid_point), perfect_line1(mid_point), mid_point)
var_e = np.var(aff_groups_var1 - perfect_line1(aff_groups_non_var1))
print()
var_vf = np.var(aff_groups_var1)
r2 = 1 - var_e/var_vf
print(var_e, var_vf, r2)


#1200-feature VNFs
# fit line on aff_groups_non_var vs aff_groups_var
fit_params = np.polyfit(aff_groups_non_var2, aff_groups_var2, 1, full=True)
coeffs2 = fit_params[0]
residuals = fit_params[1]
fit_line2 = np.poly1d(coeffs2)
print(residuals)

# perfect correspondence would have slope of 1
mid_point = (aff_groups_non_var2[0] + aff_groups_non_var2[-1])/2.0
perfect_coeffs2 = [1, fit_line2(mid_point)-mid_point]
perfect_line2 = np.poly1d(perfect_coeffs2)
print(fit_line1(mid_point), perfect_line2(mid_point), mid_point)
var_e = np.var(aff_groups_var2 - perfect_line2(aff_groups_non_var2))
var_vf = np.var(aff_groups_var2)
r2 = 1 - var_e/var_vf
print(var_e, var_vf, r2)


# 3-feature VNFs
plt.subplot(211)
plt_non_var_line, = plt.plot(time, aff_groups_non_var1, label='Non-varied monitoring frequency, std={}'.format(std1))
plt_var_line, = plt.plot(time, aff_groups_var1, label='Varied monitoring frequency, std={}'.format(std1))
plt.legend(handles=[plt_non_var_line, plt_var_line])
plt.xlabel('Simulation Time')
plt.ylabel('Number of VNF Affinity Groups (N)')

# 1200-feature VNFs
plt.subplot(212)
plt_non_var_line, = plt.plot(time, aff_groups_non_var2, label='Non-varied monitoring frequency, std={}*400'.format(std1))
plt_var_line, = plt.plot(time, aff_groups_var2, label='Varied monitoring frequency, std={}*400'.format(std1))
plt.legend(handles=[plt_non_var_line, plt_var_line])
plt.xlabel('Simulation Time')
plt.ylabel('Number of VNF Affinity Groups (N)')
#plt.show()


# 3-feature VNFs
plt.subplot(211)
plt_fit_line, = plt.plot(aff_groups_non_var1, fit_line1(aff_groups_non_var1), color='g',
                         label='Fitted line, slope={:.2f}'.format(coeffs1[0]))
plt_perfect_line, = plt.plot(aff_groups_non_var1, perfect_line1(aff_groups_non_var1), color='r', linestyle='dashed',
                             label='Perfect correspondence line, slope=1.00')
plt_points = plt.scatter(aff_groups_non_var1, aff_groups_var1, color='b', label='Observed values'.format(coeffs1[0]))
plt.legend(handles=[plt_fit_line, plt_perfect_line, plt_points])
plt.xlabel('Non-varied frequency affinity groups')
plt.ylabel('Varied frequency affinity groups')


# 1200-feature VNFs
plt.subplot(212)
plt_fit_line, = plt.plot(aff_groups_non_var2, fit_line2(aff_groups_non_var2), color='g',
                         label='Fitted line, slope={:.2f}'.format(coeffs2[0]))
plt_perfect_line, = plt.plot(aff_groups_non_var2, perfect_line2(aff_groups_non_var2), color='r', linestyle='dashed',
                             label='Perfect correspondence line, slope=1.00')
plt_points = plt.scatter(aff_groups_non_var2, aff_groups_var2, color='b', label='Observed values'.format(coeffs2[0]))
plt.legend(handles=[plt_fit_line, plt_perfect_line, plt_points])
plt.xlabel('Non-varied frequency affinity groups')
plt.ylabel('Varied frequency affinity groups')
#plt.show()
