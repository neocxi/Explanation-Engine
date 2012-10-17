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


def permutation(n, choose_from):
	"""Helper for combination

	>>> permutation(1, [1,2])
	[[1], [2]]
	>>> permutation(2, [1,2])
	[[1, 2], [2, 1]]"""	
	if n == 1:
		return [[item] for item in choose_from]

	pre = permutation(n-1, choose_from)
	out = []

	for n_minus_one in pre:
		for item in choose_from:
			if item not in n_minus_one:
				out_item = n_minus_one[:]
				out_item.append(item)
				out.append(out_item)

	return out

def combination(n, choose_from):
	"""This is a helper function that returns a list of
	all permutations of n from choose_from.
	Note: this needs to be refactored

	>>> combination(1, [1,2])
	[[1], [2]]
	>>> combination(2, [1,2])
	[[1, 2]]"""
	permutations = permutation(n, choose_from)
	combinations = []
	for item in permutations:
		if set(item) not in combinations:
			combinations.append(set(item))
	return [list(item) for item in combinations]

def assignment_space(graph, n):
	"""returns all possible assignments to n variables in this graph
	in a list"""
	space = []
	for choice in combination(n, graph.variables.keys()):
			mutations = []
			for node_name in choice:
				node = graph.get_node_with_name(node_name)
				init_assignment.update([(node_name, node.cpt.myVals[0])])
				for other_possible_value in node.cpt.myVals[1:]:
					mutations.append([(node_name, other_possible_value)])
			# initial assignment(every node has been assigned to default value)
			space.append(init_assignment)
			# go through all combinations of mutations
			for j in range(len(mutations)):
				for change in combination(j+1, mutations):
					assignment = init_assignment.copy()
					assignment.update(change)
					space.append(assignment)
	return space


def calculate_GBF(graph, assign):
	return None	