import numpy as np


class Topology:
	"""Class that governs observation topology and communciation topology"""
	def __init__(self, node_num):
		self.nodes = node_num
		self.edges = []

	def add_edge(self, i, j):
		self.edges.append([i,j])

	def status(self):
		print(self.nodes)

		for edge in self.edges:
			print(edge)



def generate_topology(node_num, observ_prob, comm_prob):

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