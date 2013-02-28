# from circuit import circuit_graph
from lake import lake_graph
from code_translation import *

# generate_encoding_to_csv(circuit_graph, ['A_switch', 'B_switch', 'C_switch', 'D_switch'])
# print encode_nodes_to_hash([circuit_graph.variables[i] for i in ['A_switch', 'B_switch', 'C_switch', 'D_switch']])

# fill_in_csv(circuit_graph, ['A_switch', 'B_switch', 'C_switch', 'D_switch'], {'Output':'on', 'Input':'on'}, 'circuit.csv')
fill_in_csv(lake_graph, ['Island', 'Bird'], {'Pox':'T'}, 'lake.csv', lake = 1)
