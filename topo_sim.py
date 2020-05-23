
import sim_env

import topo_gen

import ls_sim_topo
import gs_ci_sim_topo
import gs_sci_sim_topo


rmse_arr = [0] * 5
rmte_arr = [0] * 5

num_of_topology = 20

for i in range(num_of_topology):

	print(i)

	topo_gen.generate_topology()

	[euclidean_error, trace_error] = ls_sim_topo.simulation('cen')
	rmse_arr[0] += euclidean_error / float(num_of_topology)
	rmte_arr[0] += trace_error / float(num_of_topology)

	[euclidean_error, trace_error] = ls_sim_topo.simulation('ci')
	rmse_arr[1] += euclidean_error / float(num_of_topology)
	rmte_arr[1] += trace_error / float(num_of_topology)

	[euclidean_error, trace_error] = ls_sim_topo.simulation('bda')
	rmse_arr[2] += euclidean_error / float(num_of_topology)
	rmte_arr[2] += trace_error / float(num_of_topology)	

	[euclidean_error, trace_error] = gs_ci_sim_topo.simulation()
	rmse_arr[3] += euclidean_error / float(num_of_topology)
	rmte_arr[3] += trace_error / float(num_of_topology)

	[euclidean_error, trace_error] = gs_sci_sim_topo.simulation()
	rmse_arr[4] += euclidean_error / float(num_of_topology)
	rmte_arr[4] += trace_error / float(num_of_topology)



rmse_file = open('topology_result/rmse.txt', 'a')

rmse_file.write(str(sim_env.comm_prob) + ', ' + str(rmse_arr[0]) + ', ' + str(rmse_arr[1]) + ', '+str(rmse_arr[2]) + ', '+str(rmse_arr[3]) + ', '+str(rmse_arr[4]) +'\n')

rmse_file.close()



rmte_file = open('topology_result/rmte.txt', 'a')

rmte_file.write(str(sim_env.comm_prob) + ', ' + str(rmte_arr[0]) + ', ' + str(rmte_arr[1]) + ', '+str(rmte_arr[2]) + ', '+str(rmte_arr[3]) + ', '+str(rmte_arr[4]) +'\n')

rmte_file.close()