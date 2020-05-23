

import numpy as np
import matplotlib.pyplot as plt
import csv

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

### Data import

comm_prob = []
ls_cen_rmse = []
ls_ci_rmse = []
ls_bda_rmse = []
gs_ci_rmse = []
gs_sci_rmse = []

with open('topology/rmse.txt') as rmse_file:
	csvReader = csv.reader(rmse_file)
	for data_row in csvReader:
		comm_prob.append(float(data_row[0]))
		ls_cen_rmse.append(float(data_row[1]))
		ls_ci_rmse.append(float(data_row[2]))
		ls_bda_rmse.append(float(data_row[3]))
		gs_ci_rmse.append(float(data_row[4]))
		gs_sci_rmse.append(float(data_row[5]))


ls_cen_rmte = []
ls_ci_rmte = []
ls_bda_rmte = []
gs_ci_rmte = []
gs_sci_rmte = []

with open('topology/rmte.txt') as rmte_file:
	csvReader = csv.reader(rmte_file)
	for data_row in csvReader:
		ls_cen_rmte.append(float(data_row[1]))
		ls_ci_rmte.append(float(data_row[2]))
		ls_bda_rmte.append(float(data_row[3]))
		gs_ci_rmte.append(float(data_row[4]))
		gs_sci_rmte.append(float(data_row[5]))

### Data visualization

y_lim = [0, 0.2] 
x_lim = [0, 1]

x_lim_boarder = (x_lim[1]-x_lim[0])*0.01
x_lim_extra = [x_lim[0]-x_lim_boarder, x_lim[1]+x_lim_boarder]

y_lim_boarder = (y_lim[1]-y_lim[0])*0.01
y_lim_extra = [y_lim[0]-y_lim_boarder, y_lim[1]+y_lim_boarder]

plt.figure(1)
plt.subplot(211)

ls_cen_plt, = plt.plot(comm_prob, ls_cen_rmse, '-x', label = 'LS Cen', linewidth = 1.6, color = sim_color['ls_cen'])
ls_ci_plt, = plt.plot(comm_prob, ls_ci_rmse, '-x', label = 'LS CI', linewidth = 1.6, color = sim_color['ls_ci'])
ls_bda_plt, = plt.plot(comm_prob, ls_bda_rmse, '-x', label = 'LS BDA', linewidth = 1.6, color = sim_color['ls_bda'])

gs_sci_plt, = plt.plot(comm_prob, gs_sci_rmse, '-x', label = 'GS SCI', linewidth = 1.6, color = sim_color['gs_sci'])
gs_ci_plt, = plt.plot(comm_prob, gs_ci_rmse, '-x', label = 'GS CI', linewidth = 1.6, color = sim_color['gs_ci'])

plt.legend([ls_cen_plt, ls_ci_plt, ls_bda_plt, gs_ci_plt, gs_sci_plt], ['LS-Cen', 'LS-CI', 'LS-BDA', 'GS-CI', 'GS-SCI'])
plt.ylabel('RMSE [m]')

plt.xlim(x_lim_extra)
plt.ylim(y_lim_extra)

plt.subplot(212)

ls_cen_plt, = plt.plot(comm_prob, ls_cen_rmte, '-x', label = 'LS Cen', linewidth = 1.6, color = sim_color['ls_cen'])
ls_ci_plt, = plt.plot(comm_prob, ls_ci_rmte, '-x', label = 'LS CI', linewidth = 1.6, color = sim_color['ls_ci'])
ls_bda_plt, = plt.plot(comm_prob, ls_bda_rmte, '-x', label = 'LS BDA', linewidth = 1.6, color = sim_color['ls_bda'])

gs_sci_plt, = plt.plot(comm_prob, gs_sci_rmte, '-x', label = 'GS SCI', linewidth = 1.6, color = sim_color['gs_sci'])
gs_ci_plt, = plt.plot(comm_prob, gs_ci_rmte, '-x', label = 'GS CI', linewidth = 1.6, color = sim_color['gs_ci'])

# plt.legend([ls_cen_plt, ls_ci_plt, ls_bda_plt, gs_ci_plt, gs_sci_plt], ['LS-Cen', 'LS-CI', 'LS-BDA', 'GS-CI', 'GS-SCI'])
plt.ylabel('RMTE [m]')

plt.xlabel('communication link prob.')

plt.xlim(x_lim_extra)
plt.ylim(y_lim_extra)
plt.savefig('topology_result/topology.png')

plt.show()




