from bayesnet import DiscreteBayesNode, DiscreteCPT, DiscreteBayesNet, cut
from explanation_tree import generate_explanation_tree, generate_ET_forest, calculate_ET_score
from most_relevant_explanation import generate_MRE, calculate_GBF, generate_K_MRE
from causal_explanation_tree import generate_causal_explanation_tree, generate_CET_forest, calculate_CET_score
from maximum_a_posteriori import generate_MAP_Ind_Simplification

# build the academe graph

theory = DiscreteBayesNode('Theory', [], \
                        DiscreteCPT(['g', 'a', 'b'], [.4, .3, .3]))

practice = DiscreteBayesNode('Practice', [],\
                        DiscreteCPT(['g', 'a', 'b'], [.6, .25, .15]))

extra = DiscreteBayesNode('Extra', [],\
                        DiscreteCPT(['y', 'n'], [.3, .7]))

others = DiscreteBayesNode('Others', [], \
                        DiscreteCPT(['+', '-'], [.8, .2]))

markTP = DiscreteBayesNode('markTP', ['Theory', 'Practice'], \
                        DiscreteCPT(['pass', 'fail'],
                                {('g', 'g'):[1, 0],
                                 ('g', 'a'):[.85, .15],
                                 ('g', 'b'):[0, 1],
                                 ('a', 'g'):[.9, .1],
                                 ('a', 'a'):[.2, .8],
                                 ('a', 'b'):[0, 1],
                                 ('b', 'g'):[0, 1],
                                 ('b', 'a'):[0, 1],
                                 ('b', 'b'):[0, 1]
                                }))                        

globalMark = DiscreteBayesNode('globalMark', ['Extra', 'markTP'], \
                        DiscreteCPT(['pass', 'fail'],
                                {('y', 'pass'):[1, 0],
                                 ('y', 'fail'):[.25, .75],
                                 ('n', 'pass'):[1, 0],
                                 ('n', 'fail'):[0, 1]
                                 }))

finalMark = DiscreteBayesNode('finalMark', ['Others', 'globalMark'], \
                        DiscreteCPT(['pass', 'fail'],
                                {('+', 'pass'):[1, 0],
                                 ('+', 'fail'):[.05, .95],
                                 ('-', 'pass'):[.7, .3],
                                 ('-', 'fail'):[0, 1]
                                 }))

academe_graph = DiscreteBayesNet([theory, practice, extra, others, markTP, others, globalMark, finalMark])

# This is designed to reproduce the results of Flores' paper
print "Testing Explanation Tree:"
test_tree = generate_explanation_tree(academe_graph, ['Theory', 'Practice', 'Extra', 'Others'], {'finalMark':'fail'}, [], 0.0001, 0.0002) 

print test_tree
print "========================="

print "Testing MRE:"
MRE = generate_MRE(academe_graph, ['Theory', 'Practice', 'Extra', 'Others'], {'finalMark':'fail'})
for n in MRE:
        print n
print "========================="

print "Testing K-MRE:"
K_MRE = generate_K_MRE(MRE)
print K_MRE

print "========================="

print "Testing MAP:"
MRE = generate_MAP_Ind_Simplification(academe_graph, ['Theory', 'Practice', 'Extra', 'Others'], {'finalMark':'fail'}, 0.1)
for x in MRE:
    print x
print "========================="