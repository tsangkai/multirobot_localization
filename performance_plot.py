

import numpy as np
import matplotlib.pyplot as plt

import sim_env



color_red = '#DC4C46'
color_blue = '#2F4A92'


sim_color = {
	'gs_ci': color_blue,
	'gs_sci': color_red
}

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



### GS SCI

gs_sci_tr = np.empty(T)
gs_sci_error = np.empty(T)

gs_sci_file = open('gs_sci_output.txt', 'r')

line_count = 0
for line in gs_sci_file:
	data = line.split(", ")

	gs_sci_tr[line_count] = float(data[0])
	gs_sci_error[line_count] = float(data[1])

	line_count += 1
gs_sci_file.close()



### Performance Plot

plt.figure(1)
plt.subplot(211)


plt.plot(time_arr, gs_ci_tr, label = 'GS CI', linewidth = 1.6, color = sim_color['gs_ci'])
plt.plot(time_arr, gs_ci_upper_tr, '--', label = 'GS CI upper bound', linewidth = 1.6, color = sim_color['gs_ci'])

plt.plot(time_arr, gs_sci_tr, label = 'GS SCI', linewidth = 1.6, color = sim_color['gs_sci'])



# plotting setting

plt.ylabel('RMTE [m]')
plt.ylim([0, 0.3])





plt.subplot(212)


plt.plot(time_arr, gs_ci_error, label = 'GS CI', linewidth = 1.6, color = sim_color['gs_ci'])
plt.plot(time_arr, gs_sci_error, label = 'GS SCI', linewidth = 1.6, color = sim_color['gs_sci'])

plt.xlabel('time [s]')
plt.ylabel('RMSE [m]')
plt.ylim([0, 0.3])



plt.show()

