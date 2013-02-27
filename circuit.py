from bayesnet import DiscreteBayesNode, DiscreteCPT, DiscreteBayesNet, cut
from explanation_tree import generate_explanation_tree, generate_ET_forest, calculate_ET_score
from most_relevant_explanation import generate_MRE, calculate_GBF, generate_K_MRE
from causal_explanation_tree import generate_causal_explanation_tree, calculate_CET_score

noisy = False 
# build either of the Circuit graphs
if noisy:
    a = DiscreteBayesNode('A_switch', [],\
                            DiscreteCPT(['def', 'ok'], [.016, .984]))
    b = DiscreteBayesNode('B_switch', [],\
                            DiscreteCPT(['def', 'ok'], [.1, .9]))
    c = DiscreteBayesNode('C_switch', [],\
                            DiscreteCPT(['def', 'ok'], [.15, .85]))
    d = DiscreteBayesNode('D_switch', [],\
                            DiscreteCPT(['def', 'ok'], [.1, .9]))
    b_out = DiscreteBayesNode('B_output', ['Input', 'B_switch'], \
                            DiscreteCPT(['on', 'off'], 
                                {
                                ('on', 'def') : [.99, .01],
                                ('on', 'ok') : [0, 1],
                                ('off', 'def') : [0, 1],
                                ('off', 'ok') : [0, 1]
                                }) )
    a_out = DiscreteBayesNode('A_output', ['Input', 'A_switch'], \
                            DiscreteCPT(['on', 'off'], 
                                {
                                ('on', 'def') : [.999, .001],
                                ('on', 'ok') : [0, 1],
                                ('off', 'def') : [0, 1],
                                ('off', 'ok') : [0, 1]
                                }) )
    c_out = DiscreteBayesNode('C_output', ['B_output', 'C_switch'], \
                            DiscreteCPT(['on', 'off'], 
                                {
                                ('on', 'def') : [.985, .015],
                                ('on', 'ok') : [0, 1],
                                ('off', 'def') : [0, 1],
                                ('off', 'ok') : [0, 1]
                                }) )
    d_out = DiscreteBayesNode('D_output', ['B_output', 'D_switch'], \
                            DiscreteCPT(['on', 'off'], 
                                {
                                ('on', 'def') : [.995, .005],
                                ('on', 'ok') : [0, 1],
                                ('off', 'def') : [0, 1],
                                ('off', 'ok') : [0, 1]
                                }) )
    out = DiscreteBayesNode('Output', ['A_output', 'C_output', 'D_output'], \
                            DiscreteCPT(['on', 'off'],
                                {
                                ('on', 'on', 'on') : [.999995, .000005],
                                ('on', 'on', 'off') : [.999, .001],
                                ('on', 'off', 'on') : [.9995, .0005],
                                ('on', 'off', 'off') : [.9, .1],
                                ('off', 'on', 'off') : [.99, .01],
                                ('off', 'on', 'on') : [.99995, .00005],
                                ('off', 'off', 'on') : [.995, .005],
                                ('off', 'off', 'off') : [0, 1],
                                }))
    cur_input = DiscreteBayesNode('Input', [], DiscreteCPT(['on', 'off'], [1, 0]))

else:

    a = DiscreteBayesNode('A_switch', [],\
                            DiscreteCPT(['def', 'ok'], [.016, .984]))
    b = DiscreteBayesNode('B_switch', [],\
                            DiscreteCPT(['def', 'ok'], [.1, .9]))
    c = DiscreteBayesNode('C_switch', [],\
                            DiscreteCPT(['def', 'ok'], [.15, .85]))
    d = DiscreteBayesNode('D_switch', [],\
                            DiscreteCPT(['def', 'ok'], [.1, .9]))
    b_out = DiscreteBayesNode('B_output', ['Input', 'B_switch'], \
                            DiscreteCPT(['on', 'off'], 
                                {
                                ('on', 'def') : [1, .0],
                                ('on', 'ok') : [0, 1],
                                ('off', 'def') : [0, 1],
                                ('off', 'ok') : [0, 1]
                                }) )
    a_out = DiscreteBayesNode('A_output', ['Input', 'A_switch'], \
                            DiscreteCPT(['on', 'off'], 
                                {
                                ('on', 'def') : [1, .0],
                                ('on', 'ok') : [0, 1],
                                ('off', 'def') : [0, 1],
                                ('off', 'ok') : [0, 1]
                                }) )
    c_out = DiscreteBayesNode('C_output', ['B_output', 'C_switch'], \
                            DiscreteCPT(['on', 'off'], 
                                {
                                ('on', 'def') : [1, .0],
                                ('on', 'ok') : [0, 1],
                                ('off', 'def') : [0, 1],
                                ('off', 'ok') : [0, 1]
                                }) )
    d_out = DiscreteBayesNode('D_output', ['B_output', 'D_switch'], \
                            DiscreteCPT(['on', 'off'], 
                                {
                                ('on', 'ok') : [0, 1],
                                ('on', 'def') : [1, .0],
                                ('off', 'def') : [0, 1],
                                ('off', 'ok') : [0, 1]
                                }) )
    out = DiscreteBayesNode('Output', ['A_output', 'C_output', 'D_output'], \
                            DiscreteCPT(['on', 'off'],
                                {
                                    ('on', 'on', 'on') : [1, 0],
                                    ('on', 'on', 'off') : [1, 0],
                                    ('on', 'off', 'on') : [1, 0],
                                    ('on', 'off', 'off') : [1, 0],
                                    ('off', 'on', 'off') : [1, 0],
                                    ('off', 'on', 'on') : [1, 0],
                                    ('off', 'off', 'on') : [1, 0],
                                    ('off', 'off', 'off') : [0, 1],
                                }))

cur_input = DiscreteBayesNode('Input', [], DiscreteCPT(['on', 'off'], [1, 0]))

circuit_graph = DiscreteBayesNet( [a, b, c, d, a_out, b_out, c_out, d_out, out, cur_input] )


print "Testing Causal Explanation Tree:"
test_tree = generate_causal_explanation_tree(circuit_graph, circuit_graph, ['A_switch', 'B_switch', 'C_switch', 'D_switch'], {},{'Output':'on'}, [], 0.001) 
print test_tree
print "========================="

# print "Testing Explanation Tree:"
# test_tree = generate_explanation_tree(circuit_graph, ['A_switch', 'B_switch', 'C_switch', 'D_switch'], {'Output':'on'}, [], 0.01, 0.2) 
# print test_tree
# print "========================="



# print "Testing MRE:"
# MRE = generate_MRE(circuit_graph, ['A_switch', 'B_switch', 'C_switch', 'D_switch'], {'Output':'on', 'Input':'on'})
# for x in MRE:
#     print x
# #print MRE
# print "========================="





# print "Testing scores calculations of different methods:"

# print "BGF of [A def ]: ", calculate_GBF( circuit_graph, {'A_switch': 'def'},{'Output': 'on', 'Input':'on'}  )

# print "BGF of [A and D ok, B and C defective ]: ", calculate_GBF( circuit_graph, {'A_switch': 'ok', 'B_switch': 'def', 'C_switch': 'def', 'D_switch': 'ok'},{'Output': 'on', 'Input':'on'}  )

# print "BGF of [All defective ]: ", calculate_GBF( circuit_graph, {'A_switch': 'def', 'B_switch':'def', 'C_switch':'def', 'D_switch':'def'},{'Output': 'on', 'Input':'on'}  )

# print "BGF of [B and C def ]: ", calculate_GBF( circuit_graph, {'B_switch': 'def', 'C_switch': 'def'},{'Output': 'on', 'Input':'on'}  )

# print "BGF of [A def, C ok ]: ", calculate_GBF( circuit_graph, {'A_switch': 'def',  'C_switch': 'ok'},{'Output': 'on', 'Input':'on'}  )

# print "BGF of [B and D def]: ", calculate_GBF( circuit_graph, {'B_switch': 'def',  'D_switch': 'def'},{'Output': 'on', 'Input':'on'}  )

# print "BGF of [C ok ]: ", calculate_GBF( circuit_graph, {'C_switch': 'ok'},{'Output': 'on', 'Input':'on'})


# print "ET of [A def ]: ", calculate_ET_score( circuit_graph, {'A_switch': 'def'},{'Output': 'on', 'Input':'on'}  )

# print "ET of [A and D ok, B and C defective ]: ", calculate_ET_score( circuit_graph, {'A_switch': 'ok', 'B_switch': 'def', 'C_switch': 'def', 'D_switch': 'ok'},{'Output': 'on', 'Input':'on'}  )

# print "ET of [All defective ]: ", calculate_ET_score( circuit_graph, {'A_switch': 'def', 'B_switch':'def', 'C_switch':'def', 'D_switch':'def'},{'Output': 'on', 'Input':'on'}  )

# print "ET of [B and C def ]: ", calculate_ET_score( circuit_graph, {'B_switch': 'def', 'C_switch': 'def'},{'Output': 'on', 'Input':'on'}  )

# print "ET of [A def, C ok ]: ", calculate_ET_score( circuit_graph, {'A_switch': 'def',  'C_switch': 'ok'},{'Output': 'on', 'Input':'on'}  )

# print "ET of [B and D def]: ", calculate_ET_score( circuit_graph, {'B_switch': 'def',  'D_switch': 'def'},{'Output': 'on', 'Input':'on'}  )

# print "ET of [C ok ]: ", calculate_ET_score( circuit_graph, {'C_switch': 'ok'},{'Output': 'on', 'Input':'on'})



# print "CET of [A def ]: ", calculate_CET_score( circuit_graph, {'A_switch': 'def'},{},{'Output': 'on', 'Input':'on'}  )

# print "CET of [A and D ok, B and C defective ]: ", calculate_CET_score( circuit_graph, {'A_switch': 'ok', 'B_switch': 'def', 'C_switch': 'def', 'D_switch': 'ok'},{},{'Output': 'on', 'Input':'on'}  )

# print "CET of [All defective ]: ", calculate_CET_score( circuit_graph, {'A_switch': 'def', 'B_switch':'def', 'C_switch':'def', 'D_switch':'def'},{},{'Output': 'on', 'Input':'on'}  )

# print "CET of [B and C def ]: ", calculate_CET_score( circuit_graph, {'B_switch': 'def', 'C_switch': 'def'},{},{'Output': 'on', 'Input':'on'}  )

# print "CET of [A def, C ok ]: ", calculate_CET_score( circuit_graph, {'A_switch': 'def',  'C_switch': 'ok'},{},{'Output': 'on', 'Input':'on'}  )

# print "CET of [B and D def]: ", calculate_CET_score( circuit_graph, {'B_switch': 'def',  'D_switch': 'def'},{},{'Output': 'on', 'Input':'on'}  )

# print "CET of [C ok ]: ", calculate_CET_score( circuit_graph, {'C_switch': 'ok'},{},{'Output': 'on', 'Input':'on'})

