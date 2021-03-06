

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
	'ls_sci': color_mustard
}

T = sim_env.total_T

time_arr = np.linspace(0, sim_env.total_T*sim_env.dt, num=sim_env.total_T)


result_dir = 'boundedness_result/'

### LS Cen
ls_cen_tr = np.empty(T)
ls_cen_error = np.empty(T)

ls_cen_file = open(result_dir + 'ls_cen_output.txt', 'r')

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

ls_ci_file = open(result_dir + 'ls_ci_output.txt', 'r')

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

ls_bda_file = open(result_dir + 'ls_bda_output.txt', 'r')

line_count = 0
for line in ls_bda_file:
	data = line.split(", ")

	ls_bda_tr[line_count] = float(data[0])
	ls_bda_error[line_count] = float(data[1])

	line_count += 1
ls_bda_file.close()


### LS SCI
ls_sci_tr = np.empty(T)
ls_sci_error = np.empty(T)

ls_sci_file = open(result_dir + 'ls_sci_output.txt', 'r')

line_count = 0
for line in ls_sci_file:
	data = line.split(", ")

	ls_sci_tr[line_count] = float(data[0])
	ls_sci_error[line_count] = float(data[1])

	line_count += 1
ls_sci_file.close()


### GS CI
gs_ci_tr = np.empty(T)
gs_ci_error = np.empty(T)

gs_ci_file = open(result_dir + 'gs_ci_output.txt', 'r')

line_count = 0
for line in gs_ci_file:
	data = line.split(", ")

	gs_ci_tr[line_count] = float(data[0])
	gs_ci_error[line_count] = float(data[1])

	line_count += 1
gs_ci_file.close()


### GS CI / ONE
gs_ci_1_tr = np.empty(T)
gs_ci_1_upper_tr = np.empty(T)
gs_ci_1_error = np.empty(T)

gs_ci_1_file = open(result_dir + 'gs_ci_1_output.txt', 'r')

line_count = 0
for line in gs_ci_1_file:
	data = line.split(", ")

	gs_ci_1_tr[line_count] = float(data[0])
	gs_ci_1_upper_tr[line_count] = float(data[1])
	gs_ci_1_error[line_count] = float(data[2])

	line_count += 1
gs_ci_1_file.close()





### GS SCI
'''
gs_sci_tr = np.empty(T)
gs_sci_error = np.empty(T)

gs_sci_file = open(result_dir + 'gs_sci_output.txt', 'r')

line_count = 0
for line in gs_sci_file:
	data = line.split(", ")

	gs_sci_tr[line_count] = float(data[0])
	gs_sci_error[line_count] = float(data[1])

	line_count += 1
gs_sci_file.close()
'''

### Performance Plot

y_lim = [0, 0.34]
x_lim = [0, sim_env.total_T*sim_env.dt]

x_lim_boarder = (x_lim[1]-x_lim[0])*0.01
x_lim_extra = [x_lim[0]-x_lim_boarder, x_lim[1]+x_lim_boarder]

y_lim_boarder = (y_lim[1]-y_lim[0])*0.01
y_lim_extra = [y_lim[0]-y_lim_boarder, y_lim[1]+y_lim_boarder]

plt.figure(1)
plt.subplot(211)

ls_cen_plt, = plt.plot(time_arr, ls_cen_error, label = 'LS-Cen', linewidth = 1.6, color = sim_color['ls_cen'])
ls_ci_plt, = plt.plot(time_arr, ls_ci_error, label = 'LS-CI', linewidth = 1.6, color = sim_color['ls_ci'])
ls_bda_plt, = plt.plot(time_arr, ls_bda_error, label = 'LS-BDA', linewidth = 1.6, color = sim_color['ls_bda'])

ls_sci_plt, = plt.plot(time_arr, ls_sci_error, label = 'LS-SCI', linewidth = 1.6, color = sim_color['ls_sci'])


#gs_sci_plt, = plt.plot(time_arr, gs_sci_error, label = 'GS-SCI', linewidth = 1.6, color = sim_color['gs_sci'])
gs_ci_plt, = plt.plot(time_arr, gs_ci_error, label = 'GS-CI', linewidth = 1.6, color = sim_color['gs_ci'])

gs_ci_1_plt, = plt.plot(time_arr, gs_ci_1_error, '-.', label = 'GS-CI', linewidth = 1.6, color = sim_color['gs_ci'])

plt.legend([ls_cen_plt, ls_ci_plt, ls_bda_plt, ls_sci_plt, gs_ci_plt, gs_ci_1_plt], ['LS-Cen', 'LS-CI', 'LS-BDA', 'LS-SCI', 'GS-CI', 'R1 of GS-CI'], loc ='upper right')
plt.ylabel('RMSE [m]')
plt.xlim(x_lim_extra)
plt.ylim(y_lim_extra)


plt.subplot(212)

plt.plot(time_arr, ls_cen_tr, linewidth = 1.6, color = sim_color['ls_cen'])
plt.plot(time_arr, ls_ci_tr, linewidth = 1.6, color = sim_color['ls_ci'])
plt.plot(time_arr, ls_bda_tr, linewidth = 1.6, color = sim_color['ls_bda'])

plt.plot(time_arr, ls_sci_tr, linewidth = 1.6, color = sim_color['ls_sci'])

plt.plot(time_arr, gs_ci_tr, linewidth = 1.6, color = sim_color['gs_ci'])


plt.plot(time_arr, gs_ci_1_tr, '-.', linewidth = 1.6, color = sim_color['gs_ci'])
plt.plot(time_arr, gs_ci_1_upper_tr, '--', label = ' R1 of GS CI upper bound', linewidth = 1.6, color = sim_color['gs_ci'])

# plt.plot(time_arr, gs_sci_tr, linewidth = 1.6, color = sim_color['gs_sci'])



plt.legend()
plt.xlabel('time [s]')
plt.ylabel('RMTE [m]')
plt.xlim(x_lim_extra)
plt.ylim(y_lim_extra)

plt.savefig(result_dir + 'performance.png')

plt.show()

