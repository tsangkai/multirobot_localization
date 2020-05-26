import sys
sys.path.append("algorithm/")

import numpy as np
from math import cos, sin, atan2, sqrt

import sim_env

import topology

from robot import Robot

from gs_ci import GS_CI
from gs_sci import GS_SCI

from ls_cen import LS_Cen
from ls_ci import LS_CI
from ls_sci import LS_SCI
from ls_bda import LS_BDA


num_of_trial = sim_env.num_of_trial
total_T = sim_env.total_T

num_of_topology = 25

N = sim_env.N     # number of robot
M = sim_env.M     # number of landmark

dt = sim_env.dt


rmse_arr = {
	'LS-Cen': 0.0,
	'LS-CI': 0.0,
	'LS-SCI': 0.0,
	'LS-BDA': 0.0,
	'GS-CI': 0.0,
}

rmte_arr = {
	'LS-Cen': 0.0,
	'LS-CI': 0.0,
	'LS-SCI': 0.0,
	'LS-BDA': 0.0,
	'GS-CI': 0.0,
}

for iter_of_topology in range(num_of_topology):

	print(iter_of_topology)

	initial = sim_env.initial_position


	# Robot Initialization
	robots = [None] * N

	ls_cen_team = LS_Cen(initial.copy())
	ls_ci_team = LS_CI(initial.copy())
	ls_sci_team = LS_SCI(initial.copy())
	ls_bda_team = LS_BDA(initial.copy())

	gs_ci_robots = [None] * N
	# gs_sci_robots = [None] * N

	for n in range(N):
		robots[n] = Robot(_position=initial[2*n:2*n+2])
		gs_ci_robots[n] = GS_CI(n, initial.copy())
		# gs_sci_robots[n] = GS_SCI(n, initial.copy())


	landmarks = [None] * M
	for m in range(M):
		landmarks[m] = sim_env.Landmark(m, np.matrix(sim_env.landmark_position, dtype=float).getT())

	# N = 5
	topology.generate_topology(N, sim_env.observ_prob, sim_env.comm_prob)




	### Network Topology
	topo_file = open('topology/output.txt', 'r')

	observ_topology = topology.Topology(N)
	comm_topology = topology.Topology(N)

	edge_num_str = topo_file.readline()

	for i in range(int(edge_num_str)):
		line = topo_file.readline()
		edge = line.split(", ")

		observ_topology.add_edge(int(edge[0]), int(edge[1]))

	edge_num_str = topo_file.readline()
	for i in range(int(edge_num_str)):
		line = topo_file.readline()
		edge = line.split(", ")

		comm_topology.add_edge(int(edge[0]), int(edge[1]))

	topo_file.close()


	# LS-Cen topology
	ls_cen_observ_topology = topology.Topology(N)
	all_to_all_comm = True
	for i in range(N):
		for j in range(i+1, N):
			all_to_all_comm = all_to_all_comm and ([i,j] in comm_topology.edges) and ([j,i] in comm_topology.edges) 

	if all_to_all_comm:
		ls_cen_observ_topology.edges = observ_topology.edges
	else: # only absolute observation is allowed
		for edge in observ_topology.edges:
			if edge[1] == N:
				ls_cen_observ_topology.add_edge(edge[0], edge[1])

	# LS-CI and LS-SCI topology
	ls_ci_observ_topology = topology.Topology(N)
	for edge in observ_topology.edges:
		if edge[1] == sim_env.N: # absolute observation
			ls_ci_observ_topology.add_edge(edge[0], edge[1])

		elif edge in comm_topology.edges:  # relative obseravion
			ls_ci_observ_topology.add_edge(edge[0], edge[1])

	# LS-BDA topology
	ls_bda_observ_topology = topology.Topology(N)
	for edge in observ_topology.edges:
		if edge[1] == sim_env.N: # absolute observation
			ls_bda_observ_topology.add_edge(edge[0], edge[1])

		elif (edge in comm_topology.edges) and ([edge[1], edge[0]] in comm_topology.edges):  # relative obseravion
			ls_bda_observ_topology.add_edge(edge[0], edge[1])


	### Simulation
	for t in range(total_T):
 
		# reset theta
		for n in range(N):
			ls_cen_team.theta[n] = robots[n].theta
			ls_ci_team.theta[n] = robots[n].theta
			ls_sci_team.theta[n] = robots[n].theta
			ls_bda_team.theta[n] = robots[n].theta

			gs_ci_robots[n].theta = robots[n].theta
			# gs_sci_robots[n].theta = robots[n].theta


		# motion propagation 
		odometry_input = [0] * N
		odometry_star_input = [0] * N

		for n in range(N):
			[v, omega] = [0,0]
			v_star = 0
			pre_update_position = [100, 100]

			while(not sim_env.inRange(pre_update_position, sim_env.origin)):
				[v, omega] = [sim_env.max_v*np.random.uniform(-1,1), sim_env.max_omega*np.random.uniform(-1,1)]
				v_star = v + np.random.normal(0, sqrt(sim_env.var_u_v))
				pre_update_position = [robots[n].position[0] + cos(robots[n].theta)*v_star*dt, robots[n].position[1] + sin(robots[n].theta)*v_star*dt]

			odometry_input[n] = [v, omega]
			odometry_star_input[n] = [v_star, omega]

		ls_cen_team.motion_propagation_update(odometry_input, dt)
		ls_ci_team.motion_propagation_update(odometry_input, dt)
		ls_sci_team.motion_propagation_update(odometry_input, dt)
		ls_bda_team.motion_propagation_update(odometry_input, dt)

		for n in range(N):
			robots[n].motion_propagation(odometry_star_input[n], dt)

			gs_ci_robots[n].motion_propagation_update(odometry_input[n], dt)
			# gs_sci_robots[n].motion_propagation_update(odometry_input[n], dt)


		# observation update
		for edge in observ_topology.edges:
			[observer_idx, observed_idx] = edge

			# absoluate observation
			if observed_idx == N:
				[dis, phi] = sim_env.relative_measurement(robots[observer_idx].position, robots[observer_idx].theta, landmarks[0].position)
				
				gs_ci_robots[observer_idx].ablt_obsv_update([dis, phi], landmarks[0])				
				# gs_sci_robots[observer_idx].ablt_obsv_update([dis, phi], landmarks[0])				

				if edge in ls_cen_observ_topology.edges:
					ls_cen_team.ablt_obsv_update(observer_idx, [dis, phi], landmarks[0])			

				if edge in ls_ci_observ_topology.edges:
					ls_ci_team.ablt_obsv_update(observer_idx, [dis, phi], landmarks[0])				
					ls_sci_team.ablt_obsv_update(observer_idx, [dis, phi], landmarks[0])				

				if edge in ls_bda_observ_topology.edges:
					ls_bda_team.ablt_obsv_update(observer_idx, [dis, phi], landmarks[0])				

			# relative observation
			else:
				[dis, phi] = sim_env.relative_measurement(robots[observer_idx].position, robots[observer_idx].theta, robots[observed_idx].position)
				
				gs_ci_robots[observer_idx].rela_obsv_update(observed_idx, [dis, phi])
				# gs_sci_robots[observer_idx].rela_obsv_update(observed_idx, [dis, phi])

				if edge in ls_cen_observ_topology.edges:
					ls_cen_team.rela_obsv_update(observer_idx, observed_idx, [dis, phi])		

				if edge in ls_ci_observ_topology.edges:
					ls_ci_team.rela_obsv_update(observer_idx, observed_idx, [dis, phi])		
					ls_sci_team.rela_obsv_update(observer_idx, observed_idx, [dis, phi])		

				if edge in ls_bda_observ_topology.edges:
					ls_bda_team.rela_obsv_update(observer_idx, observed_idx, [dis, phi])				


		# communication update
		for edge in comm_topology.edges:
			[sender_idx, receiver_idx] = edge

			gs_ci_robots[receiver_idx].comm_update(gs_ci_robots[sender_idx].s, gs_ci_robots[sender_idx].sigma, gs_ci_robots[sender_idx].th_sigma)
			# gs_sci_robots[receiver_idx].comm_update(gs_sci_robots[sender_idx].s, gs_sci_robots[sender_idx].sigma_i, gs_sci_robots[sender_idx].sigma_d)


	# Error Calculation

	gs_ci_se = 0
	gs_ci_te = 0

	ls_cen_se = 0
	ls_cen_te = 0
	
	ls_ci_se = 0
	ls_ci_te = 0	

	ls_sci_se = 0
	ls_sci_te = 0

	ls_bda_se = 0
	ls_bda_te = 0	



	for j in range(N):
		gs_ci_se += ((gs_ci_robots[j].s[2*j,0] - robots[j].position[0]) ** 2 + (gs_ci_robots[j].s[2*j+1,0] - robots[j].position[1]) ** 2)
		gs_ci_te += gs_ci_robots[j].sigma[2*j:2*j+2,2*j:2*j+2].trace()[0,0]

		ls_cen_se += (ls_cen_team.s[2*j,0] - robots[j].position[0]) ** 2 + (ls_cen_team.s[2*j+1,0] - robots[j].position[1]) ** 2
		ls_cen_te += ls_cen_team.sigma[2*j:2*j+2,2*j:2*j+2].trace()[0,0]

		ls_ci_se += (ls_ci_team.s[2*j,0] - robots[j].position[0]) ** 2 + (ls_ci_team.s[2*j+1,0] - robots[j].position[1]) ** 2
		ls_ci_te += ls_ci_team.sigma[2*j:2*j+2,2*j:2*j+2].trace()[0,0]

		ls_sci_se += (ls_sci_team.s[2*j,0] - robots[j].position[0]) ** 2 + (ls_sci_team.s[2*j+1,0] - robots[j].position[1]) ** 2
		ls_sci_te += ls_sci_team.getSigma()[2*j:2*j+2,2*j:2*j+2].trace()[0,0]

		ls_bda_se += (ls_bda_team.s[2*j,0] - robots[j].position[0]) ** 2 + (ls_bda_team.s[2*j+1,0] - robots[j].position[1]) ** 2
		ls_bda_te += ls_bda_team.sigma[2*j:2*j+2,2*j:2*j+2].trace()[0,0]


	rmse_arr['LS-Cen'] += sqrt(ls_cen_se/float(N)) / float(num_of_topology)
	rmte_arr['LS-Cen'] += sqrt(ls_cen_te/float(N))  / float(num_of_topology)

	rmse_arr['LS-CI'] += sqrt(ls_ci_se/float(N)) / float(num_of_topology)
	rmte_arr['LS-CI'] += sqrt(ls_ci_te/float(N))  / float(num_of_topology)

	rmse_arr['LS-SCI'] += sqrt(ls_sci_se/float(N)) / float(num_of_topology)
	rmte_arr['LS-SCI'] += sqrt(ls_sci_te/float(N))  / float(num_of_topology)

	rmse_arr['LS-BDA'] += sqrt(ls_bda_se/float(N)) / float(num_of_topology)
	rmte_arr['LS-BDA'] += sqrt(ls_bda_te/float(N))  / float(num_of_topology)

	rmse_arr['GS-CI'] += sqrt(gs_ci_se/float(N)) / float(num_of_topology)
	rmte_arr['GS-CI'] += sqrt(gs_ci_te/float(N))  / float(num_of_topology)



rmse_file = open('topology_result/rmse.txt', 'a')
rmse_file.write(str(sim_env.comm_prob) + ', ' + str(rmse_arr['LS-Cen']) + ', ' + str(rmse_arr['LS-CI']) + ', ' + str(rmse_arr['LS-SCI']) + ', ' +  str(rmse_arr['LS-BDA']) + ', ' + str(rmse_arr['GS-CI']) +'\n')
rmse_file.close()


rmte_file = open('topology_result/rmte.txt', 'a')
rmte_file.write(str(sim_env.comm_prob) + ', ' + str(rmte_arr['LS-Cen']) + ', ' + str(rmte_arr['LS-CI']) + ', ' + str(rmse_arr['LS-SCI']) + ', ' +  str(rmte_arr['LS-BDA']) + ', ' + str(rmte_arr['GS-CI']) +'\n')
rmte_file.close()