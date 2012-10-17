# This is an implementation of an algorithm finding 
# the Most Relevant Explanation(Yuan)
def generate_MRE(graph):
	"""Returns a pair(assign, GBF) where assign is a list that contain pairs of
	assignments and GBF is the generalized Bayes factor"""

	answer_space = {}
	max_assignment = None
	max_GBF = float('-inf')
	for i in range(len(graph.nodes)): #a MRE can take 1~N assignments
		for choice in combination(i+1, graph.variables.keys()):
			mutations = []
			for node_name in choice:
				node = graph.get_node_with_name(node_name)
				init_assignment.update([(node_name, node.cpt.myVals[0])])
				for other_possible_value in node.cpt.myVals[1:]:
					mutations.append([(node_name, other_possible_value)])
			# initial assignment(every node has been assigned to default value)
			cur_GBF = calculate_GBF(graph, init_assignment)
			if cur_GBF > max_GBF:
				max_assignment = init_assignment
				max_GBF = cur_GBF
			# go through all combinations of mutations
			for j in range(len(mutations)):
				for change in combination(j+1, mutations):
					assignment = init_assignment.copy()
					assignment.update(change)
					cur_GBF = calculate_GBF(graph, assignment)
					if cur_GBF > max_GBF:
						max_assignment = assignment
						max_GBF = cur_GBF


	return max_assignment, max_GBF

def combination(n, choose_from):
	""""This is a helper function that returns a list of
	all combination of n from choose_from""""
	pass

def calculate_GBF(graph, assign):
		