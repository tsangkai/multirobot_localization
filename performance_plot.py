

import numpy as np
import matplotlib.pyplot as plt

import sim_env



color_red = '#DC4C46'
color_blue = '#2F4A92'


T = sim_env.total_T

time_arr = np.linspace(0, sim_env.total_T*sim_env.dt, num=sim_env.total_T)




### GS CI
gs_ci_tr = np.empty(T)
gs_ci_upper_tr = np.empty(T)
gs_ci_error = np.empty(T)

gs_ci_file = open('gs_ci_output.txt', 'r')

line_count = 0
for line in gs_ci_file:
	data = line.split(", ")

	gs_ci_tr[line_count] = float(data[0])
	gs_ci_upper_tr[line_count] = float(data[1])
	gs_ci_error[line_count] = float(data[2])

	line_count += 1
gs_ci_file.close()



### Performance Plot

plt.figure(1)
plt.subplot(211)


plt.plot(time_arr, gs_ci_tr, label = 'GS CI', linewidth = 1.6, color = color_blue)
plt.plot(time_arr, gs_ci_upper_tr, '--', label = 'GS CI upper bound', linewidth = 1.6, color = color_blue)




# plotting setting

plt.ylabel('RMTE [m]')






plt.subplot(212)


plt.plot(time_arr, gs_ci_error, label = 'GS CI', linewidth = 1.6, color = color_blue)

plt.xlabel('time [s]')
plt.ylabel('RMSE [m]')


plt.show()

