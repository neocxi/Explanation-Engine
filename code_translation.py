import csv
from most_relevant_explanation import assignment_space

def encode_a_graph(graph):
	"""Takes a graph object.
		Returns a hash that has key as an assignment 
					and value as the coded value"""
	return encode_nodes(graph.nodes)

def encode_nodes(nodes):
	out = {}
	code = '0'
	if len(nodes) == 1:
		for value in nodes[0].cpt.values():
			out[nodes[0].name + " : " + value] = code
			code = chr(ord(code) + 1)
		out[nodes[0].name + " : NA"] = '-'
	else:
		prev = encode_nodes([node for node in nodes if node != nodes[0]])
		cur = encode_nodes([nodes[0]])
		for key in cur.keys():
			for k in prev.keys():
				out[key + ", " + k] = cur[key] + prev[k]

	return out
