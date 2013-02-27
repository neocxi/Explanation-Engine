from bayesnet import DiscreteBayesNode, DiscreteCPT, DiscreteBayesNet, cut, prob_given
from explanation_tree import ExplanationTreeNode, max_mutual_information, merge
import math

def generate_causal_explanation_tree(ori_graph, graph, explanatory_var, observation, explanadum, path, alpha):
    x, inf = max_causal_information(graph, explanatory_var, observation, explanadum)
    
    if inf < alpha:
       return ExplanationTreeNode()

    if len(explanatory_var) is 0:
        return ExplanationTreeNode()

    t = ExplanationTreeNode(parent = path[-1][0] if path else None, root = x) #new tree with a parent pointer to its parent
    
    for value in graph.get_node_with_name(x).cpt.values():
        intervened_graph = graph.create_graph_with_intervention( {x:value} )
        new_tree = generate_causal_explanation_tree(ori_graph, intervened_graph, cut(explanatory_var, x), \
                            observation, explanadum, path + [(x, value)], alpha)
        strength = math.log( intervened_graph.prob_given(explanadum, observation) / \
                            ori_graph.prob_given(explanadum, observation), 2) if intervened_graph.prob_given(explanadum, observation) and ori_graph.prob_given(explanadum, observation) else float("NaN")
        t.add_branch(value, new_tree, strength)

    return t

def generate_CET_forest(ori_graph, graph, explanatory_var, observation, explanadum, path):
    if len(explanatory_var) is 0:
        return [ExplanationTreeNode()]

    forest = []
    for x in explanatory_var:

        for value in graph.get_node_with_name(x).cpt.values():
            intervened_graph = graph.create_graph_with_intervention( {x:value} )
            new_forest = generate_CET_forest(ori_graph, intervened_graph, cut(explanatory_var, x), observation, explanadum, path + [(x, value)])
            for new_tree in new_forest:
                t = ExplanationTreeNode(parent = path[-1][0] if path else None, root = x)
                strength = math.log( intervened_graph.prob_given(explanadum, observation) / \
                            ori_graph.prob_given(explanadum, observation), 2) if intervened_graph.prob_given(explanadum, observation) and ori_graph.prob_given(explanadum, observation) else float("NaN")
                t.add_branch(value, new_tree, strength)
                forest.append(t)

    return forest


def max_causal_information(graph, explanatory_var, observation, explanadum):
    max_x, max_inf = None, float("-inf")
    for x in explanatory_var:
        cur_inf = 0
        denominator = sum( [graph.prob_given({x:x_val_temp}, observation) \
                                * graph.create_graph_with_intervention({x:x_val_temp}).prob_given(explanadum, observation)
                                 for x_val_temp in graph.get_node_with_name(x).cpt.values()] )
        for x_val in graph.get_node_with_name(x).cpt.values():
            intervened_graph = graph.create_graph_with_intervention( {x:x_val} )
            log_part = math.log( intervened_graph.prob_given(explanadum, observation) / \
                                 denominator, 2) if denominator and intervened_graph.prob_given(explanadum, observation) else 0
            temp = graph.prob_given({x:x_val}, observation) * \
                        intervened_graph.prob_given(explanadum, observation) / \
                        graph.prob_given(explanadum, observation) * \
                        log_part if graph.prob_given(explanadum, observation) else 0
            cur_inf += temp 
        if cur_inf > max_inf:
            max_x = x
            max_inf = cur_inf
    return max_x, max_inf

def calculate_CET_score(ori_graph, explanation, observation, explanadum):
    intervened_graph = ori_graph.create_graph_with_intervention( explanation )
    strength = math.log( intervened_graph.prob_given(explanadum, observation) / \
                        ori_graph.prob_given(explanadum, observation), 2) if intervened_graph.prob_given(explanadum, observation) and ori_graph.prob_given(explanadum, observation) else float("NaN")
    return strength