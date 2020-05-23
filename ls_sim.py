### This simulation takes the observation topology directly.


import sys
sys.path.append("algorithm/")


import numpy as np
import math

import sim_env
from topology import Topology


### Algorithm
algorithm = sys.argv[1]
output_file_name = 'boundedness_result/ls_' + algorithm + '_output.txt'

if algorithm == 'cen':	
	from ls_cen_team import LS_Cen_Team as LS_Team

elif algorithm == 'bda':
	from ls_bda_team import LS_BDA_Team as LS_Team

elif algorithm == 'ci':
	from ls_ci_team import LS_CI_Team as LS_Team

print("run " + algorithm + ":")


### Network Topology
# topo_file = open('topology/default.txt', 'r')
topo_file = open('topology/output.txt', 'r')

observ_topology = Topology(sim_env.N)

edge_num_str = topo_file.readline()

for i in range(int(edge_num_str)):
	line = topo_file.readline()
	edge = line.split(", ")

	observ_topology.add_edge(int(edge[0]), int(edge[1]))


### Simulation Body

num_of_trial = sim_env.num_of_trial
total_T = sim_env.total_T

# number of robot
N = sim_env.N 

# number of landmark
M = sim_env.M 

sigma_tr_arr = [0] * total_T
error_arr = [0] * total_T


for i in range(num_of_trial):

	print(i)
	initial = sim_env.initial_position

	robots = LS_Team(initial)

	landmarks = [None] * M
	for m in range(M):
		landmarks[m] = sim_env.Landmark(m, np.matrix(sim_env.landmark_position, dtype=float).getT())

	for t in range(total_T):

		# motion propagation
		robots.prop_update()

		# observation update
		for edge in observ_topology.edges:
			[observer_idx, observed_idx] = edge

			# absoluate observation
			if observed_idx == sim_env.N:
				[dis, phi] = sim_env.relative_measurement(robots.position[2*observer_idx:2*observer_idx+2], robots.theta[observer_idx], landmarks[0].position)
				robots.ablt_obsv(observer_idx, [dis, phi], landmarks[0])				

			# relative observation
			else:
				[dis, phi] = sim_env.relative_measurement(robots.position[2*observer_idx:2*observer_idx+2], robots.theta[observer_idx], robots.position[2*observed_idx:2*observed_idx+2])
				robots.rela_obsv(observer_idx, observed_idx, [dis, phi])

		# localization error
		s = 0
		for j in range(N):
			jj = 2*j
			s += ((robots.s[jj,0] - robots.position[jj,0])**2 + (robots.s[jj+1,0] - robots.position[jj+1,0])**2)
		
		error_arr[t] += math.sqrt(s/float(N)) / float(num_of_trial)

		# covariance error
		sigma_tr_arr[t] += math.sqrt(robots.sigma.trace()[0,0]/float(N)) / float(num_of_trial)


### Simulation Output

file = open(output_file_name, 'w')

for t in range(total_T):
	file.write(str(sigma_tr_arr[t]) + ', ' + str(error_arr[t]) + '\n')

file.close()
