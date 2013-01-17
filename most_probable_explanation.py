# This is an implementation of the MPE algorithm
from most_relevant_explanation import assignment_space

def generate_MPE(graph, exp_var, explanadum):
	"""Returns a list of pairs (hypothesis, posterior probability) in
	sorted order."""

	out = []
	# All assignments are full assignments and hence len(exp_var)
	space = assignment_space(graph, len(exp_var), explanadum, exp_var)
	for ind_assignment in space:
		posterior = graph.prob_given(ind_assignment, explanadum)
		out.append( (ind_assignment, posterior) )

	return sorted(out, key = lambda x: -x[1])
