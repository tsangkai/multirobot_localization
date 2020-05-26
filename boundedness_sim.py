import sys
sys.path.append("algorithm/")

import numpy as np
from math import cos, sin, atan2, sqrt

import sim_env
from topology import Topology

from robot import Robot

from gs_ci import GS_CI
from gs_sci import GS_SCI

from ls_cen import LS_Cen
from ls_ci import LS_CI
from ls_bda import LS_BDA
from ls_sci import LS_SCI


num_of_trial = sim_env.num_of_trial
total_T = sim_env.total_T


N = sim_env.N     # number of robot
M = sim_env.M     # number of landmark

dt = sim_env.dt

### Network Topology
topo_file = open('topology/default.txt', 'r')

observ_topology = Topology(sim_env.N)
comm_topology = Topology(sim_env.N)

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



gs_ci_rmse_arr = [0] * total_T
gs_ci_rmte_arr = [0] * total_T
gs_ci_th_rmte_tr_arr = [0] * total_T

gs_sci_rmse_arr = [0] * total_T
gs_sci_rmte_arr = [0] * total_T

ls_cen_rmse_arr = [0] * total_T
ls_cen_rmte_arr = [0] * total_T

ls_ci_rmse_arr = [0] * total_T
ls_ci_rmte_arr = [0] * total_T

ls_bda_rmse_arr = [0] * total_T
ls_bda_rmte_arr = [0] * total_T

ls_sci_rmse_arr = [0] * total_T
ls_sci_rmte_arr = [0] * total_T

for iter_of_trial in range(num_of_trial):

	print(iter_of_trial)

	initial = sim_env.initial_position

	# initialization
	robots = [None] * N

	gs_ci_robots = [None] * N
	gs_sci_robots = [None] * N

	for n in range(N):
		robots[n] = Robot(_position=initial[2*n:2*n+2])
		gs_ci_robots[n] = GS_CI(n, initial.copy())
		gs_sci_robots[n] = GS_SCI(n, initial.copy())

	ls_cen_team = LS_Cen(initial.copy())
	ls_ci_team = LS_CI(initial.copy())
	ls_bda_team = LS_BDA(initial.copy())
	ls_sci_team = LS_SCI(initial.copy())


	landmarks = [None] * M
	for m in range(M):
		landmarks[m] = sim_env.Landmark(m, np.matrix(sim_env.landmark_position, dtype=float).getT())

	# simulation body
	for t in range(total_T):
 
		# reset theta
		for n in range(N):
			gs_ci_robots[n].theta = robots[n].theta
			gs_sci_robots[n].theta = robots[n].theta

			ls_cen_team.theta[n] = robots[n].theta
			ls_ci_team.theta[n] = robots[n].theta
			ls_bda_team.theta[n] = robots[n].theta
			ls_sci_team.theta[n] = robots[n].theta


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


		for n in range(N):
			robots[n].motion_propagation(odometry_star_input[n], dt)

			gs_ci_robots[n].motion_propagation_update(odometry_input[n], dt)
			gs_sci_robots[n].motion_propagation_update(odometry_input[n], dt)

		ls_cen_team.motion_propagation_update(odometry_input, dt)
		ls_ci_team.motion_propagation_update(odometry_input, dt)
		ls_bda_team.motion_propagation_update(odometry_input, dt)
		ls_sci_team.motion_propagation_update(odometry_input, dt)


		
		# observation update
		for edge in observ_topology.edges:
			[observer_idx, observed_idx] = edge

			# absoluate observation
			if observed_idx == N:
				[dis, phi] = sim_env.relative_measurement(robots[observer_idx].position, robots[observer_idx].theta, landmarks[0].position)
				
				gs_ci_robots[observer_idx].ablt_obsv_update([dis, phi], landmarks[0])				
				gs_sci_robots[observer_idx].ablt_obsv_update([dis, phi], landmarks[0])				

				ls_cen_team.ablt_obsv_update(observer_idx, [dis, phi], landmarks[0])				
				ls_ci_team.ablt_obsv_update(observer_idx, [dis, phi], landmarks[0])				
				ls_bda_team.ablt_obsv_update(observer_idx, [dis, phi], landmarks[0])				
				ls_sci_team.ablt_obsv_update(observer_idx, [dis, phi], landmarks[0])				

			# relative observation
			else:
				[dis, phi] = sim_env.relative_measurement(robots[observer_idx].position, robots[observer_idx].theta, robots[observed_idx].position)
				
				gs_ci_robots[observer_idx].rela_obsv_update(observed_idx, [dis, phi])
				gs_sci_robots[observer_idx].rela_obsv_update(observed_idx, [dis, phi])

				ls_cen_team.rela_obsv_update(observer_idx, observed_idx, [dis, phi])				
				ls_ci_team.rela_obsv_update(observer_idx, observed_idx, [dis, phi])				
				ls_bda_team.rela_obsv_update(observer_idx, observed_idx, [dis, phi])				
				ls_sci_team.rela_obsv_update(observer_idx, observed_idx, [dis, phi])				

		'''
		# communication update
		for edge in comm_topology.edges:
			[sender_idx, receiver_idx] = edge

			gs_ci_robots[receiver_idx].comm_update(gs_ci_robots[sender_idx].s, gs_ci_robots[sender_idx].sigma, gs_ci_robots[sender_idx].th_sigma)
			gs_sci_robots[receiver_idx].comm_update(gs_sci_robots[sender_idx].s, gs_sci_robots[sender_idx].sigma_i, gs_sci_robots[sender_idx].sigma_d)
		
		'''
		

		# error calculation
		# real error
		focus_idx = 0

		gs_ci_se = 0
		gs_sci_se = 0

		ls_cen_se = 0
		ls_ci_se = 0
		ls_bda_se = 0
		ls_sci_se = 0

		for j in range(N):
			# gs_ci_se += ((gs_ci_robots[focus_idx].s[2*j,0] - robots[j].position[0]) ** 2 + (gs_ci_robots[focus_idx].s[2*j+1,0] - robots[j].position[1]) ** 2)
			# gs_sci_se += ((gs_sci_robots[focus_idx].s[2*j,0] - robots[j].position[0]) ** 2 + (gs_sci_robots[focus_idx].s[2*j+1,0] - robots[j].position[1]) ** 2)
			gs_ci_se += ((gs_ci_robots[j].s[2*j,0] - robots[j].position[0]) ** 2 + (gs_ci_robots[j].s[2*j+1,0] - robots[j].position[1]) ** 2)
			gs_sci_se += ((gs_sci_robots[j].s[2*j,0] - robots[j].position[0]) ** 2 + (gs_sci_robots[j].s[2*j+1,0] - robots[j].position[1]) ** 2)

			ls_cen_se += ((ls_cen_team.s[2*j,0] - robots[j].position[0])**2 + (ls_cen_team.s[2*j+1,0] - robots[j].position[1])**2)
			ls_ci_se += ((ls_ci_team.s[2*j,0] - robots[j].position[0])**2 + (ls_ci_team.s[2*j+1,0] - robots[j].position[1])**2)
			ls_bda_se += ((ls_bda_team.s[2*j,0] - robots[j].position[0])**2 + (ls_bda_team.s[2*j+1,0] - robots[j].position[1])**2)
			ls_sci_se += ((ls_sci_team.s[2*j,0] - robots[j].position[0])**2 + (ls_sci_team.s[2*j+1,0] - robots[j].position[1])**2)



		gs_ci_rmse_arr[t] += sqrt(gs_ci_se/float(N)) / float(num_of_trial)
		gs_sci_rmse_arr[t] += sqrt(gs_sci_se/float(N)) / float(num_of_trial)

		ls_cen_rmse_arr[t] += sqrt(ls_cen_se/float(N)) / float(num_of_trial)
		ls_ci_rmse_arr[t] += sqrt(ls_ci_se/float(N)) / float(num_of_trial)
		ls_bda_rmse_arr[t] += sqrt(ls_bda_se/float(N)) / float(num_of_trial)
		ls_sci_rmse_arr[t] += sqrt(ls_sci_se/float(N)) / float(num_of_trial)


		# covariance error
		gs_ci_rmte_arr[t] += sqrt(gs_ci_robots[focus_idx].sigma.trace()[0,0]/float(N)) / float(num_of_trial)
		gs_ci_th_rmte_tr_arr[t] += sqrt(gs_ci_robots[focus_idx].th_sigma.trace()[0,0]/float(N)) / float(num_of_trial)

		total_sigma = gs_sci_robots[focus_idx].getSigma()
		gs_sci_rmte_arr[t] += sqrt(total_sigma.trace()[0,0]/float(N)) / float(num_of_trial)

		ls_cen_rmte_arr[t] += sqrt(ls_cen_team.sigma.trace()[0,0]/float(N)) / float(num_of_trial)
		ls_ci_rmte_arr[t] += sqrt(ls_ci_team.sigma.trace()[0,0]/float(N)) / float(num_of_trial)
		ls_bda_rmte_arr[t] += sqrt(ls_bda_team.sigma.trace()[0,0]/float(N)) / float(num_of_trial)

		ls_sci_rmte_arr[t] += sqrt(ls_sci_team.getSigma().trace()[0,0]/float(N)) / float(num_of_trial)



# output
gs_ci_file = open('boundedness_result/gs_ci_output.txt', 'w')
gs_sci_file = open('boundedness_result/gs_sci_output.txt', 'w')

ls_cen_file = open('boundedness_result/ls_cen_output.txt', 'w')
ls_ci_file = open('boundedness_result/ls_ci_output.txt', 'w')
ls_bda_file = open('boundedness_result/ls_bda_output.txt', 'w')
ls_sci_file = open('boundedness_result/ls_sci_output.txt', 'w')


for t in range(total_T):
	gs_ci_file.write( str(gs_ci_rmte_arr[t]) + ', ' + str(gs_ci_th_rmte_tr_arr[t]) + ', ' + str(gs_ci_rmse_arr[t]) + '\n')
	gs_sci_file.write( str(gs_sci_rmte_arr[t]) + ', ' + str(gs_sci_rmse_arr[t]) + '\n')

	ls_cen_file.write(str(ls_cen_rmte_arr[t]) + ', ' + str(ls_cen_rmse_arr[t]) + '\n')
	ls_ci_file.write(str(ls_ci_rmte_arr[t]) + ', ' + str(ls_ci_rmse_arr[t]) + '\n')
	ls_bda_file.write(str(ls_bda_rmte_arr[t]) + ', ' + str(ls_bda_rmse_arr[t]) + '\n')
	ls_sci_file.write(str(ls_sci_rmte_arr[t]) + ', ' + str(ls_sci_rmse_arr[t]) + '\n')

gs_ci_file.close()
gs_sci_file.close()

ls_cen_file.close()
ls_ci_file.close()
ls_bda_file.close()
ls_sci_file.close()

