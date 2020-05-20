

import numpy as np
import matplotlib.pyplot as plt

import sim_env



color_red = '#DC4C46'
color_blue = '#2F4A92'
color_green = '#47785E'
color_mustard = '#E3AE27'
color_purple = '#C796C7'

sim_color = {
	'ls_cen': color_green,
	'ls_ci': color_purple,
	'ls_bda': color_red,
	'gs_ci': color_blue,
	'gs_sci': color_mustard
}

T = sim_env.total_T

time_arr = np.linspace(0, sim_env.total_T*sim_env.dt, num=sim_env.total_T)




### LS Cen
ls_cen_tr = np.empty(T)
ls_cen_error = np.empty(T)

ls_cen_file = open('ls_cen_output.txt', 'r')

line_count = 0
for line in ls_cen_file:
	data = line.split(", ")

	ls_cen_tr[line_count] = float(data[0])
	ls_cen_error[line_count] = float(data[1])

	line_count += 1
ls_cen_file.close()


### LS CI
ls_ci_tr = np.empty(T)
ls_ci_error = np.empty(T)

ls_ci_file = open('ls_ci_output.txt', 'r')

line_count = 0
for line in ls_ci_file:
	data = line.split(", ")

	ls_ci_tr[line_count] = float(data[0])
	ls_ci_error[line_count] = float(data[1])

	line_count += 1
ls_ci_file.close()


### LS BDA
ls_bda_tr = np.empty(T)
ls_bda_error = np.empty(T)

ls_bda_file = open('ls_bda_output.txt', 'r')

line_count = 0
for line in ls_bda_file:
	data = line.split(", ")

	ls_bda_tr[line_count] = float(data[0])
	ls_bda_error[line_count] = float(data[1])

	line_count += 1
ls_bda_file.close()



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

y_lim_upper = 0.3 #1.1 

plt.figure(1)
plt.subplot(211)

plt.plot(time_arr, ls_cen_error, label = 'LS Cen', linewidth = 1.6, color = sim_color['ls_cen'])
plt.plot(time_arr, ls_ci_error, label = 'LS CI', linewidth = 1.6, color = sim_color['ls_ci'])
plt.plot(time_arr, ls_bda_error, label = 'LS BDA', linewidth = 1.6, color = sim_color['ls_bda'])

plt.plot(time_arr, gs_ci_error, label = 'GS CI', linewidth = 1.6, color = sim_color['gs_ci'])
plt.plot(time_arr, gs_sci_error, label = 'GS SCI', linewidth = 1.6, color = sim_color['gs_sci'])

plt.legend()
plt.ylabel('RMSE [m]')
plt.xlim([0, sim_env.total_T*sim_env.dt])
plt.ylim([0, y_lim_upper])






plt.subplot(212)

plt.plot(time_arr, ls_cen_tr, linewidth = 1.6, color = sim_color['ls_cen'])
plt.plot(time_arr, ls_ci_tr, linewidth = 1.6, color = sim_color['ls_ci'])
plt.plot(time_arr, ls_bda_tr, linewidth = 1.6, color = sim_color['ls_bda'])

plt.plot(time_arr, gs_ci_tr, linewidth = 1.6, color = sim_color['gs_ci'])
plt.plot(time_arr, gs_ci_upper_tr, '--', label = 'GS CI upper bound', linewidth = 1.6, color = sim_color['gs_ci'])

plt.plot(time_arr, gs_sci_tr, linewidth = 1.6, color = sim_color['gs_sci'])



plt.legend()
plt.xlabel('time [s]')
plt.ylabel('RMTE [m]')
plt.xlim([0, sim_env.total_T*sim_env.dt])
plt.ylim([0, y_lim_upper])


# plt.savefig('plot/performance_dr.png')

plt.show()

