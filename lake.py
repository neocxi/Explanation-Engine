from bayesnet import DiscreteBayesNode, DiscreteCPT, DiscreteBayesNet, cut
from explanation_tree import generate_explanation_tree, generate_ET_forest, calculate_ET_score
from most_relevant_explanation import generate_MRE, calculate_GBF, generate_K_MRE
from causal_explanation_tree import generate_causal_explanation_tree, generate_CET_forest, calculate_CET_score
from most_probable_explanation import generate_MPE
from maximum_a_posteriori import generate_MAP_Ind_Simplification

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
print "prob dist of presence of pox" , lake_graph.enumerate_ask('Pox', {})
print "exact prob of pox present", lake_graph.prob({'Pox':'T'})
print "prob dist of presence of birds conditioned on having an island" ,\
                                lake_graph.enumerate_ask('Bird', {'Island':'T'})
print "exact prob of birds present given it doesnt have an island", \
                                lake_graph.prob({'Bird':'T','Island':'F' })
                                
#test running explanation tree algorithm on the above graph
print "Testing Explanation Tree:"
test_tree = generate_explanation_tree(lake_graph, ['Bird', 'Island'], {'Pox':'T'}, [], 0.0001, 0.0002) 
print test_tree
print "========================="

print "Testing Explanation Forest:"
forest = generate_ET_forest(lake_graph, ['Bird', 'Island'], {'Pox':'T'}, []) 
for tree in forest:
    print tree
print "========================="

print "Testing MRE:"
MRE = generate_MRE(lake_graph, ['Bird', 'Island'], {'Pox':'T'})
for x in MRE:
    print x
print "========================="

print "Testing K-MRE:"
K_MRE = generate_K_MRE(MRE)
for x in K_MRE:
    print x
print "========================="

print "Testing Causal Intervention:"
print "Intervene Bird to true"
intervened = lake_graph.create_graph_with_intervention({'Bird' : 'T'})
print "post intervention probability of Bird being true", intervened.prob( {'Bird' : 'T'} )
print "post intervention prob of Pox being true", intervened.prob( {'Pox':'T'} )
print "========================="

print "Testing Causal Explanation Tree:"
test_tree = generate_causal_explanation_tree(lake_graph, lake_graph, ['Bird', 'Island'], {}, {'Pox':'T'}, [], 0.0001) 
print test_tree
print "========================="


print "Testing Explanation Forest:"
forest = generate_CET_forest(lake_graph, lake_graph, ['Bird', 'Island'], {}, {'Pox':'T'}, []) 
for tree in forest:
    print tree
print "========================="


print "Testing scores calculations of different methods:"
print "BGF of [Island being true]: ", calculate_GBF( lake_graph, {"Island" : "T" }, {"Pox" : "T"} )
print "BGF of [Bird being true, Island being false]: ", calculate_GBF( lake_graph, {"Island" : "T", "Bird" : "T" }, {"Pox" : "T"} )
print "ET score of [Island being true], which is essentially posterior probability of the explanation given explanadum : ", calculate_ET_score( lake_graph, {"Island" : "T"}, {"Pox" : "T"})
print "CET score of [Island being true]", calculate_CET_score( lake_graph, {"Island" : "T"}, {}, {"Pox" : "T"}) #The empty hash is for Observation

print "========================="
print "Testing MPE:"
MRE = generate_MPE(lake_graph, ['Bird', 'Island'], {'Pox':'T'})
for x in MRE:
    print x
print "========================="

print "Testing MAP:"
MRE = generate_MAP_Ind_Simplification(lake_graph, ['Bird', 'Island'], {'Pox':'T'}, 0.1)
for x in MRE:
    print x
print "========================="
