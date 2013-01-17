from most_probable_explanation import generate_MPE
from most_relevant_explanation import combination

def generate_MAP_Ind_Simplification(graph, exp_var, explanadum, threshold):
	"""A method that outputs MAP(Simplified MPE) using Independence-based simplification rule."""
	
	MPE = generate_MPE(graph, exp_var, explanadum)

	best_simplified_exp = []
	key_set = {}

	for (explanation, prob) in MPE:
		ori_metric = graph.prob_given(explanadum, explanation)
		lower_bound = ori_metric * (1 - threshold)
		upper_bound = ori_metric * (1 + threshold)

		simplified_space = []
		keys = explanation.keys()
		# Loop over all possible abductions of an explanation to find all simplifications
		for i in range(len(explanation)):
			for abduction in combination(i, keys):
				# retrieve assignments from the explanation
				abducted_assignment = {}
				for var in abduction:
					abducted_assignment[var] = explanation[var]
				# test equivalence
				if graph.prob_given(explanadum, abducted_assignment) >= lower_bound \
					and graph.prob_given(explanadum, abducted_assignment) <= upper_bound:
					simplified_space.append(abducted_assignment)

		# find out the best simplification and its posterior probability
		if not len(simplified_space) == 0:	
			cur_min = len(simplified_space[0])
			candidates = []
			for simplification in simplified_space:
				if len(simplification) == cur_min:
					candidates.append(simplification)
				elif len(simplification) < cur_min:
					candidates = [simplification]
					cur_min = len(simplification)


			min_imprecision = graph.prob_given(explanadum, candidates[0]) - ori_metric
			cur_best = candidates[0]
			for candidate in candidates:
				if graph.prob_given(explanadum, candidate) - ori_metric < min_imprecision:
					min_imprecision = graph.prob_given(explanadum, candidate) - ori_metric
					cur_best = candidate

			if cur_best.__str__() not in key_set:
				best_simplified_exp.append( (cur_best, graph.prob_given(cur_best, explanadum)) )
				key_set[cur_best.__str__()] = 0

	return sorted(best_simplified_exp, key = lambda x: -x[1])




