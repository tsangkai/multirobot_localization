

from topology import Topology
import numpy as np

import sim_env
node_num = sim_env.N

observ_prob = sim_env.observ_prob
comm_prob = sim_env.comm_prob

def generate_topology():

	observ_topology = Topology(node_num)

	for i in range(node_num):
		for j in range(node_num+1):
			if (i!=j) and (np.random.rand() < observ_prob):
				observ_topology.add_edge(i, j)

	comm_topology = Topology(node_num)

	for i in range(node_num):
		for j in range(node_num):
			if (i!=j) and (np.random.rand() < comm_prob):
				comm_topology.add_edge(i, j)



	output_file = open('topology/output.txt', 'w')

	# observation topology
	output_file.write(str(len(observ_topology.edges))+'\n')

	for edge in observ_topology.edges:
		[i, j] = edge
		output_file.write(str(i) + ', ' + str(j) + '\n')

	# communication topology
	output_file.write(str(len(comm_topology.edges))+'\n')

	for edge in comm_topology.edges:
		[i, j] = edge
		output_file.write(str(i) + ', ' + str(j) + '\n')

	output_file.close()
