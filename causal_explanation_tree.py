from bayesnet import DiscreteBayesNode, DiscreteCPT, DiscreteBayesNet, cut, prob_given
from explanation_tree import ExplanationTreeNode, max_mutual_information, merge
import math

def generate_causal_explanation_tree(graph, explanatory_var, observation, explanadum, path, alpha):
    x, inf = max_causal_information(graph, explanatory_var, observation, explanadum)
    
    if inf < alpha:
       return ExplanationTreeNode()

    if len(explanatory_var) is 0:
        return ExplanationTreeNode()

    t = ExplanationTreeNode(parent = path[-1][0] if path else None, root = x) #new tree with a parent pointer to its parent
    
    for value in graph.get_node_with_name(x).cpt.values():
    	intervened_graph = graph.create_graph_with_intervention( {x:calue} )
        new_tree = generate_explanation_tree(intervened_graph, cut(explanatory_var, x), \
                            explanadum, path + [(x, value)], alpha)
        strength = math.log( intervened_graph.prob_given(explanadum, observation) / \
        					graph.prob_given(explanadum, observation))
        t.add_branch(value, new_tree, prob_given(graph, dict(path + [(x, value)]), explanadum) )

    return t


def max_causal_information(graph, explanatory_var, observation, explanadum):
	max_x, max_inf = None, float("-inf")
	for x in explanatory_var:
		cur_inf = 0
		for x_val in graph.get_node_with_name(x).cpt.values():
			intervened_graph = graph.create_graph_with_intervention( {x:x_val} )
			denominator = sum( [graph.prob_given({x:x_val_temp}, observation) \
								* intervened_graph(explanadum, observation)
								 for x_val_temp in graph.get_node_with_name(x).cpt.values()])
			log_part = math.log( intervened_graph.prob_given(explanadum, observation) / \
								 denominator)
			cur_inf += graph.prob_given({x:x_val}, observation) * \
						intervened_graph.prob_given(explanadum, observation) / \
						graph.prob_given(explanadum, observation) * \
						log_part

		if cur_inf > max_inf:
			max_x = x
			max_inf = cur_inf

	return max_x, max_inf