from bayesnet import DiscreteBayesNode, DiscreteCPT, DiscreteBayesNet, cut, prob_given
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
        self.children[assignment] = new_tree
        self.children_prob[assignment] = prob

    def __str__(self):
        return self.print_tree(0)

    def print_tree(self, depth):
        prefix = depth * "-"
        out = ""
        out += prefix + "This node is: %s\n" % self.root
        for branch in self.children.keys():
            out += prefix + "%s has branch assignment of %s with score of %f\n" % (self.root, branch, self.children_prob[branch])
            if not self.children[branch].is_empty():
               out += self.children[branch].print_tree(depth + 2)
        return out 
            
    def assignment_space(self):
        out = []
        for key in self.children.keys():
            out.append( ({self.root : key}, self.children_prob[key]) )
            for assignment, score in self.children[key].assignment_space():
                out.append( (dict(assignment.items() + [(self.root, key)]), score) )
        return out

    def is_leaf(self, assignment):
        # print "testing if", assignment, "on ", self.root
        if len(self.children) == 0 and len(assignment) == 0:
            # print "returning true"
            return True

        if self.root in assignment:
            assignment = assignment.copy()
            child = assignment.pop(self.root)
            return self.children[child].is_leaf(assignment)
        else:
            # print "returning false"
            return False


def generate_explanation_tree(graph, explanatory_var, explanadum, path, alpha, beta):
    x, inf = max_mutual_information(graph, explanatory_var, merge(explanadum, path))
    
    if inf < alpha or prob_given(graph, dict(path), explanadum) < beta:
        return ExplanationTreeNode()

    if len(explanatory_var) is 0:
        return ExplanationTreeNode()

    t = ExplanationTreeNode(parent = path[-1][0] if path else None, root = x) #new tree with a parent pointer to its parent
    
    for value in graph.get_node_with_name(x).cpt.values():
        new_tree = generate_explanation_tree(graph, cut(explanatory_var, x), \
                            explanadum, path + [(x, value)], alpha, beta)
        t.add_branch(value, new_tree, prob_given(graph, dict(path + [(x, value)]), explanadum) )

    return t

def generate_ET_forest(graph, explanatory_var, explanadum, path):
    if len(explanatory_var) is 0:
        return [ExplanationTreeNode()]

    forest = []
    for x in explanatory_var:

        for value in graph.get_node_with_name(x).cpt.values():
            new_forest = generate_ET_forest(graph, cut(explanatory_var, x), \
                                explanadum, path + [(x, value)])
            for new_tree in new_forest:
                t = ExplanationTreeNode(parent = path[-1][0] if path else None, root = x)
                t.add_branch(value, new_tree, prob_given(graph, dict(path + [(x, value)]), explanadum) )
                forest.append(t)

    return forest


def merge(a,b):
    """merge a with b"""
    c = a.copy()
    c.update(b)
    return c
    
    
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
                    temp += graph.prob_given(y_assign, x_assign) * \
                            math.log(graph.prob_given(y_assign, x_assign) / \
                                        graph.prob_given(y_assign, condition)) \
                            if graph.prob_given(y_assign, x_assign) and \
                               graph.prob_given(y_assign, condition) \
                            else 0
                cur_inf += temp * prob_given(graph, {x:val_x}, condition)
        
        # print "current arg", x, cur_inf
        if cur_inf > max_inf:
            argmax, max_inf = x, cur_inf
    return argmax, max_inf
      
def calculate_ET_score(graph, assignment, explanadum):
    return graph.prob_given(assignment, explanadum)
