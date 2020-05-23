

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
