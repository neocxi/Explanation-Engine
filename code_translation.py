import csv
from most_relevant_explanation import *
from most_probable_explanation import generate_MPE
from maximum_a_posteriori import generate_MAP_Ind_Simplification
from explanation_tree import generate_explanation_tree, calculate_ET_score
from causal_explanation_tree import generate_causal_explanation_tree, calculate_CET_score
from string import join
import math

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

def encode_nodes_to_hash(nodes):
	out = {}
	code = '1'
	if len(nodes) == 1:
		for value in nodes[0].cpt.values():
			out[code] = {nodes[0].name : value}
			code = chr(ord(code) - 1)
		out['9'] = {}
	else:
		prev = encode_nodes_to_hash([node for node in nodes if node != nodes[0]])
		cur = encode_nodes_to_hash([nodes[0]])
		for key in cur.keys():
			for k in prev.keys():
				out[ key + k ] = dict(cur[key].items() + prev[k].items())

	return out

def generate_encoding_to_csv(graph, nodes, path = "temp.csv"):
	translation = encode_nodes([graph.variables[name] for name in nodes])
	# print translation

	with open(path, 'wb') as csvfile:
		writer = csv.writer(csvfile, dialect = 'excel')
		writer.writerow(["Assignment", "Code"])
		for key in translation.keys():
			writer.writerow([key, translation[key]])

def fill_in_csv(graph, exp_var, explanadum, path = "temp.csv", lake = 0):
	# Assume for now that the codes in csv use the some scheme as implied
    # by this script
    assignments = []
    code_hash = encode_nodes_to_hash([graph.variables[name] for name in exp_var])
    MPE_space = generate_MPE(graph, exp_var, explanadum)
    MAP_threshold = 0.1
    MAP_space = generate_MAP_Ind_Simplification(graph, exp_var, explanadum, MAP_threshold)
    MRE_space = generate_MRE(graph, exp_var, explanadum)

    alpha = 0.01
    beta = 0.2
    ET = generate_explanation_tree(graph, exp_var, explanadum, [], alpha, beta)
    ET_space = sorted(ET.assignment_space(), key = lambda x: -x[1])
    ET_score_space = sorted(
    		[(code_hash[key], calculate_ET_score(graph, code_hash[key], explanadum)) for key in code_hash.keys() if len(code_hash[key])],
    		key = lambda x: -x[1])
    # print ET_score_space

    alpha_CET = 0.0001
    CET = generate_causal_explanation_tree(graph, graph, exp_var, {}, explanadum, [], alpha_CET)
    # print "Printing cet", CET
    CET_space = sorted(CET.assignment_space(), key = lambda x: -x[1])
    CET_score_space = sorted(
    		[(code_hash[key], calculate_CET_score(graph, code_hash[key], {}, explanadum)) for key in code_hash.keys() if len(code_hash[key])],
    		key = lambda x: -x[1] if not math.isnan(x[1]) else 9999999999999)
    # print CET_score_space

    with open(path, 'r') as csvfile:
    	reader = csv.reader(csvfile, dialect = 'excel')
    	for row in reader:
    		processed_code = join([row[0][i] for i in range(len(exp_var))], "")
    		assignments.append((row[0], code_hash[processed_code]))

    # print assignments
    with open(path, 'w') as csvfile:
   		writer = csv.writer(csvfile, dialect = 'excel') 
   		writer.writerow(["CondensedString","MPE_rank","MPE_score","MAP_I_rank","MAP_I_score","MAP_I_para_theta","MRE_rank","MRE_score","ET_rank_for_tree","ET_rank_for_score","ET_score","ET_leaf","ET_ALPHA","ET_BETA","CET_rank_for_tree","CET_rank_for_score","CET_score","CET_leaf","CET_ALPHA"])
   		for code, assignment in assignments:
   			row = ['"' + code + '"']
   			# escape all empty explanations
   			# and all invalid assignemnts for lake.py
   			print lake, code[2] != '9', code[3] != '9'
   			if (not len(assignment)) or (lake and (code[2] != '9' or code[3] != '9')):
   				print "escaping"
   				row += ["NaN" for _ in range(18)]
   				writer.writerow(row)
   				continue

   			rank = space_rank(MPE_space, assignment)
   			if rank:
	   			row.append(rank)# MPE rank
	   			row.append(MPE_space[int(rank) - 1][1]) # MPE score
	   		else:
	   			row.append("NaN")
	   			row.append("NaN")

	   		rank = space_rank(MAP_space, assignment)
	   		if rank:
	   			row.append(rank)# map rank
	   			row.append(MAP_space[int(rank) - 1][1]) # MAP score
	   		else:
	   			row.append("NaN")
	   			row.append("NaN")

	   		row.append(MAP_threshold)

	   		rank = space_rank(MRE_space, assignment)
	   		if rank:
	   			row.append(rank)# mre rank
	   			row.append(MRE_space[int(rank) - 1][1]) # MRE score
	   		else:
	   			row.append("NaN")
	   			row.append("NaN")

	   		rank = space_rank(ET_space, assignment)
	   		if rank:
	   			row.append(rank)# et tree rank
	   		else:
	   			row.append("NaN")

	   		rank = space_rank(ET_score_space, assignment)
	   		# print code, assignment, rank, ET_score_space[int(rank) - 1][1]
	   		if rank:
	   			row.append(rank) # et score rank
	   			row.append(ET_score_space[int(rank) - 1][1]) # ET score
	   		else:
	   			row.append("ERROR ! should not happen") #should not happen

	   		row += [1] if ET.is_leaf(assignment) else [0]
	   		row.append(alpha)
	   		row.append(beta)

	   		rank = space_rank(CET_space, assignment)
	   		if rank:
	   			row.append(rank)# et tree rank
	   		else:
	   			row.append("NaN")

	   		rank = space_rank(CET_score_space, assignment)
	   		if rank:
	   			row.append(rank) # et score rank
	   			row.append(CET_score_space[int(rank) - 1][1]) # ET score
	   		else:
	   			row.append("ERROR ! should not happen") #should not happen

	   		# print "determining leaf for", code
	   		row += [1] if CET.is_leaf(assignment) else [0]
	   		row.append(alpha_CET)

   			writer.writerow(row)


def space_rank(space, assignment):
	assgn_list = [node[0] for node in space]
	value_list = [node[1] for node in space]
	# print "p in space_rank"
	# print space
	# print rankdata(value_list)
	if assignment in assgn_list:
		return rankdata(value_list)[assgn_list.index(assignment)]
	else:
		return 0
	# return [node[0] for node in space].index(assignment) + 1 if assignment in [node[0] for node in space] else 0

def rank_simple(vector):
    return sorted(range(len(vector)), key = lambda x: -vector.__getitem__(x) if not math.isnan(vector.__getitem__(x)) else 9999999999999)

def rankdata(a):
    n = len(a)
    ivec=rank_simple(a)
    svec=[a[rank] if not math.isnan(a[rank]) else 9999999999999 for rank in ivec]
    sumranks = 0
    dupcount = 0
    newarray = [0]*n
    for i in xrange(n):
        sumranks += i
        dupcount += 1
        if i==n-1 or svec[i] != svec[i+1]:
            averank = sumranks / float(dupcount) + 1
            for j in xrange(i-dupcount+1,i+1):
                newarray[ivec[j]] = averank
            sumranks = 0
            dupcount = 0
    return newarray