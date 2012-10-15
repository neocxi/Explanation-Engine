from bayesnet import DiscreteBayesNode, DiscreteCPT, DiscreteBayesNet, cut
import math

class ExplanationTreeNode(object):
    """A class that represents Flores's(2005) explanation trees"""
    def __init__(self, parent = None, root = None, children = None):
        self.parent = parent
        self.root = root
        self.children = children if children else {}
        self.children_prob = {}
        
    def is_empty(self):
        return self.root == None

    def add_branch(self, assignment, new_tree, prob):
        """docstring for add_branch"""
        self.children[assignment] = new_tree
        self.children_prob[assignment] = prob

    def __str__(self):
        out = ""
        out += "This node is: %s\n" % self.root
        for branch in self.children.keys():
            out += "has branch assignment of %s with probability of %f\n" % (branch, self.children_prob[branch])
            if not self.children[branch].is_empty():
               out += "points to:"
               out += self.children[branch].__str__()

        return out
            


def generate_explanation_tree(graph, explanatory_var, explanadum, path, alpha, beta):
    x, inf = max_mutual_information(graph, explanatory_var, merge(explanadum, path))
    
    # if inf < alpha or prob_given(graph, dict(path), explanadum):
    #        return ExplanationTreeNode()
    if len(explanatory_var) is 0:
        return ExplanationTreeNode()
    t = ExplanationTreeNode(parent = path[-1][0] if path else None, root = x) #new tree with a parent pointer to its parent
    
    for value in graph.get_node_with_name(x).cpt.values():
        new_tree = generate_explanation_tree(graph, cut(explanatory_var, x), \
                            explanadum, path + [(x, value)], alpha, beta)
        t.add_branch(value, new_tree, prob_given(graph, dict(path + [(x, value)]), explanadum) )
    return t

def merge(a,b):
    """merge a with b"""
    c = a.copy()
    c.update(b)
    return c
    
def prob_given(graph, posterior, prior): 
    """calculate P(posterior|prior) on a given graph. Posterior and prior are two dicts
    specifying assignments"""
    return graph.prob(merge(prior, posterior)) / graph.prob(prior)
    
def max_mutual_information(graph, explanatory_var, condition):
    """Return (argmax, max) X in exp_var such that Sum of Y in exp_var INF(X;Y | condition)"""
    argmax = ''
    max_inf = float("-inf")
    
    for x in explanatory_var:
        cur_inf = .0
        #loop through every variable in explanatory_var
        for y in explanatory_var:
            for val_x in graph.get_node_with_name(x).cpt.values():
                x_assign = merge({x:val_x}, condition)
                temp = 0
                for val_y in graph.get_node_with_name(y).cpt.values():
                    y_assign = {y:val_y}
                    temp += prob_given(graph, y_assign, x_assign) * \
                            math.log(prob_given(graph, y_assign, x_assign) / \
                                        prob_given(graph, y_assign, condition))
                cur_inf += temp * prob_given(graph, {x:val_x}, condition)
        
        print "current arg", x, cur_inf
        if cur_inf > max_inf:
            argmax, max_inf = x, cur_inf
    return argmax, max_inf
      
