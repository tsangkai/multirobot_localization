import sys
sys.path.append("algorithm/")

import numpy as np
import math

import sim_env
from topology import Topology

from gs_ci_robot import GS_CI_Robot


num_of_trial = sim_env.num_of_trial
total_T = sim_env.total_T

# number of robot
N = sim_env.N 

# number of landmark
M = sim_env.M 


def simulation():

	### Network Topology
	topo_file = open('topology/output.txt', 'r')

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




	sigma_tr_arr = 0
	error_arr = 0

	for i in range(num_of_trial):

		initial = sim_env.initial_position

		# initialization
		robots = [None] * N
		for n in range(N):
			robots[n] = GS_CI_Robot(n, initial.copy())

		landmarks = [None] * M
		for m in range(M):
			landmarks[m] = sim_env.Landmark(m, np.matrix(sim_env.landmark_position, dtype=float).getT())

		# simulation body
		for t in range(total_T):
	 
			# motion propagation 
			for i in range(N):
				robots[i].prop_update()

			
			# observation update
			for edge in observ_topology.edges:
				[observer_idx, observed_idx] = edge

				# absoluate observation
				if observed_idx == N:
					[dis, phi] = sim_env.relative_measurement(robots[observer_idx].position, robots[observer_idx].theta, landmarks[0].position)
					robots[observer_idx].ablt_obsv([dis, phi], landmarks[0])				

				# relative observation
				else:
					[dis, phi] = sim_env.relative_measurement(robots[observer_idx].position, robots[observer_idx].theta, robots[observed_idx].position)
					robots[observer_idx].rela_obsv(observed_idx, [dis, phi])

			# communication update
			for edge in comm_topology.edges:
				[sender_idx, receiver_idx] = edge

				robots[receiver_idx].comm(robots[sender_idx].s, robots[sender_idx].sigma, robots[sender_idx].th_sigma)


			# error calculation
			# real error
		error = 0
		tr_error = 0
		for j in range(N):
			error += pow(robots[j].s[2*j,0] - robots[j].position[0],2) + pow(robots[j].s[2*j+1,0] - robots[j].position[1],2)
			tr_error += robots[j].sigma[2*j:2*j+2,2*j:2*j+2].trace()[0,0]

		error_arr += math.sqrt(error/float(N)) / float(num_of_trial)

		# covariance error
		sigma_tr_arr += math.sqrt(tr_error/float(N)) / float(num_of_trial)

	return [error_arr, sigma_tr_arr]


# output
#file = open('topology_result/gs_ci_output.txt', 'w')

# for t in range(total_T):
# 	file.write( str(sigma_tr_arr[t]) + ', ' + str(sigma_th_tr_arr[t]) + ', ' + str(error_arr[t]) + '\n')

# file.close()

