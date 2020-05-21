import sys
sys.path.append("algorithm/")

import numpy as np
import math

import sim_env
from topology import Topology

from gs_sci_robot import GS_SCI_Robot


num_of_trial = sim_env.num_of_trial
total_T = sim_env.total_T

# number of robot
N = sim_env.N 

# number of landmark
M = sim_env.M 


### Network Topology
topo_file = open('topology/default.txt', 'r')

node_num_str = topo_file.readline()
observ_topology = Topology(int(node_num_str))
comm_topology = Topology(int(node_num_str))

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



sigma_tr_arr = [0] * total_T
error_arr = [0] * total_T

for i in range(num_of_trial):

	print(i)

	initial = sim_env.initial_position

	robots = [None] * N
	for n in range(N):
		robots[n] = GS_SCI_Robot(n, initial.copy())

	landmarks = [None] * M
	for m in range(M):
		landmarks[m] = sim_env.Landmark(m, np.matrix(sim_env.landmark_position, dtype=float).getT())

	for t in range(total_T):

		# motion propagation
		robots[0].prop_update()
		robots[1].prop_update()
		robots[2].prop_update()
		robots[3].prop_update()
		robots[4].prop_update()


		# observation update
		for edge in observ_topology.edges:
			[observer_idx, observed_idx] = edge

			# absoluate observation
			if observed_idx == sim_env.N:
				[dis, phi] = sim_env.relative_measurement(robots[observer_idx].position, robots[observer_idx].theta, landmarks[0].position)
				robots[observer_idx].ablt_obsv([dis, phi], landmarks[0])				

			# relative observation
			else:
				[dis, phi] = sim_env.relative_measurement(robots[observer_idx].position, robots[observer_idx].theta, robots[observed_idx].position)
				robots[observer_idx].rela_obsv(observed_idx, [dis, phi])


		# communication update
		for edge in comm_topology.edges:

		 	[sender_idx, receiver_idx] = edge
			robots[receiver_idx].comm(robots[sender_idx].s, robots[sender_idx].sigma_i, robots[sender_idx].sigma_d)

		# real error
		s = 0
		for j in range(N):
		 	s += (robots[0].s[2*j,0] - robots[j].position[0]) ** 2 + (robots[0].s[2*j+1,0] - robots[j].position[1]) ** 2
		error_arr[t] += math.sqrt(s/float(N)) /float(num_of_trial)

		# covariance error
		total_sigma = robots[0].getSigma()
		sigma_tr_arr[t] += math.sqrt(total_sigma.trace()[0,0]/float(N)) / float(num_of_trial)

file = open('gs_sci_output.txt', 'w')

for t in range(total_T):

	file.write( str(sigma_tr_arr[t]) + ', ' + str(error_arr[t]) + '\n')

file.close()


