from bayesnet import DiscreteBayesNode, DiscreteCPT, DiscreteBayesNet
from explanation_tree import generate_explanation_tree
from most_relevant_explanation import generate_MRE
from causal_explanation_tree import generate_causal_explanation_tree


# build the Drug graph

sex = DiscreteBayesNode('Sex', [],\
						DiscreteCPT(['f', 'm'], [.5, .5]))

drug = DiscreteBayesNode('Drug', ['Sex'], \
						DiscreteCPT(['yes', 'no'], 
							{('f', ):[.25, .75], 
							 ('m', ):[.75, .25]}))
recovery = DiscreteBayesNode('Recovery', ['Sex', 'Drug'], \
							DiscreteCPT(['recovery', 'death'],
								{
								('f', 'yes'):[.2, .8],
								('f', 'no'):[.3, .7],
								('m', 'yes'):[.6, .4],
								('m', 'no'):[.7, .3]
								}))

drug_graph = DiscreteBayesNet( [sex, drug, recovery])

print "Testing Explanation Tree:"
test_tree = generate_explanation_tree(drug_graph, ['Sex', 'Drug'], {'Recovery':'recovery'}, [], 0.01, 0.2) 
print test_tree
print "========================="

print "Testing MRE:"
MRE = generate_MRE(drug_graph, ['Sex', 'Drug'], {'Recovery':'recovery'})
print MRE
print "========================="

print "Testing Causal Explanation Tree:"
test_tree = generate_causal_explanation_tree(drug_graph, drug_graph, ['Sex', 'Drug'], {},{'Recovery':'recovery'}, [], 0.001) 
print test_tree
print "========================="