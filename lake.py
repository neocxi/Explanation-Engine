from bayesnet import DiscreteBayesNode, DiscreteCPT, DiscreteBayesNet, cut
import math

# build the lake graph
island = DiscreteBayesNode('Island', [], \
                        DiscreteCPT(['T','F'], [0.5, 0.5]))
                        
bird = DiscreteBayesNode('Bird', ['Island'], \
                        DiscreteCPT(['T','F'],
                        {('T',):[3.0/4, .25], 
                         ('F',):[.25, 3.0/4]}))
                         
pox = DiscreteBayesNode('Pox', ['Island', 'Bird'], \
                        DiscreteCPT(['T','F'],
                        {('T','T'):[.6, .4],
                         ('T','F'):[.7, .3],
                         ('F','T'):[.2, .8],
                         ('F','F'):[.3, .7]                        
                        }))                         
lake_graph = DiscreteBayesNet([island, bird, pox])

# test the proabalistic inference engine
print "prob dist of presence of birds" , lake_graph.enumerate_ask('Bird', {})
print "prob dist of presence of birds" , lake_graph.enumerate_ask('Pox', {})
print "exact prob of pox present", lake_graph.prob({'Pox':'T'})
print "prob dist of presence of birds conditioned on having an island" ,\
                                lake_graph.enumerate_ask('Bird', {'Island':'T'})
print "exact prob of birds present given it doesnt have an island", \
                                lake_graph.prob({'Bird':'T','Island':'F' })
                                

class ExplanationTreeNode(object):
    """A class that represents Flores's(2005) explanation trees"""
    def __init__(self, parent = None, root = None, children = None):
        self.parent = parent
        self.root = root
        self.children = children if children else {}
        self.children_prob = {}
        
    def add_branch(self, assignment, new_tree, prob):
        """docstring for add_branch"""
        self.children[assignment] = new_tree
        self.children_prob[assignment] = prob


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
    max_inf = -float("-inf")
    
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
        
        if cur_inf > max_inf:
            argmax, max_inf = x, cur_inf
    return argmax, max_inf
      
test_tree = generate_explanation_tree(lake_graph, ['Bird', 'Island'], {'Pox':'T'}, [], 0.01, 0.2)  
print test_tree.children, test_tree.children_prob, test_tree.root