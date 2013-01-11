from bayesnet import DiscreteBayesNode, DiscreteCPT, DiscreteBayesNet
from explanation_tree import generate_explanation_tree
from most_relevant_explanation import generate_MRE, generate_K_MRE
from causal_explanation_tree import generate_causal_explanation_tree

asia = DiscreteBayesNode('VisitToAsia', [], \
						DiscreteCPT(['yes', 'no'], [.01, .99]))

smoking = DiscreteBayesNode('Smoking', [], \
						DiscreteCPT(['yes','no'], [.5, .5]))

tuberculosis = DiscreteBayesNode('Tuberculosis', ['VisitToAsia'], \
						DiscreteCPT(['yes', 'no'],
							{
							('yes', ) : [.05, .95],
							('no', ) : [.01, .99]
							}))

lung_cancer = DiscreteBayesNode('Lung_Cancer', ['Smoking'], \
						DiscreteCPT(['yes', 'no'],
							{
							('yes', ) : [.1, .9],
							('no', ) : [.01, .99]
							}))

bronchitis = DiscreteBayesNode('Bronchitis', ['Smoking'], \
						DiscreteCPT(['yes', 'no'], 
							{
							('yes', ) : [.6, .4],
							('no', ) : [.3, .7]
							}))

tborca = DiscreteBayesNode('TborCa', ['Tuberculosis', 'Lung_Cancer'], \
						DiscreteCPT(['yes', 'no'],
							{
							('yes', 'yes') : [1, 0],
							('yes', 'no') : [1, 0],
							('no', 'yes') : [1, 0],
							('no', 'no') : [0, 1]
							}))

x_ray = DiscreteBayesNode('X_ray', ['TborCa'], \
						DiscreteCPT(['abnormal', 'normal'],
							{
							('yes', ) : [.98, .02],
							('no', ) : [.05, .95]
							})	)

dyspnea = DiscreteBayesNode('Dyspnea', ['TborCa', 'Bronchitis'], \
						DiscreteCPT(['yes', 'no'],
							{
							('yes', 'yes') : [.9, .1],
							('yes', 'no') : [.7, .3],
							('no', 'yes') : [.8, .2],
							('no', 'no') : [.1, .9]
							}))

asia_graph = DiscreteBayesNet( [asia, tuberculosis, tborca, x_ray, dyspnea, bronchitis, smoking, lung_cancer] )
exp_var = ['Lung_Cancer', 'VisitToAsia', 'Tuberculosis', 'Smoking', 'Bronchitis']
explanadum = {'X_ray':'abnormal'}

print "Testing MRE:"
MRE = generate_MRE(asia_graph, exp_var, explanadum)
print MRE
print "========================="

print "Testing K-MRE:"
K_MRE = generate_K_MRE(MRE)
print K_MRE
print "========================="

print "Testing Explanation Tree:"
test_tree = generate_explanation_tree(asia_graph, exp_var, explanadum, [], 0.01, 0.2) 
print test_tree
print "========================="

print "Testing Causal Explanation Tree:"
test_tree = generate_causal_explanation_tree(asia_graph, asia_graph, exp_var, {},explanadum, [], 0.001) 
print test_tree
print "========================="
