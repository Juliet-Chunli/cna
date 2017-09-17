"""
Algorithmic Thinking 1
wk 4
Aplication #2
Questions
"""
# general imports
import urllib2
import random
import time
import math
import UPATrial
import numpy
from collections import deque

# Desktop imports
import matplotlib.pyplot as plt


############################################
def copy_graph(graph):
    """
    Make a copy of a graph
    """
    new_graph = {}
    for node in graph:
        new_graph[node] = set(graph[node])
    return new_graph

def delete_node(ugraph, node):
    """
    Delete a node from an undirected graph
    """
    neighbors = ugraph[node]
    ugraph.pop(node)
    for neighbor in neighbors:
    	if neighbor in ugraph and node in ugraph[neighbor]:
        	ugraph[neighbor].remove(node)
    
def targeted_order(ugraph):
    """
    Compute a targeted attack order consisting
    of nodes of maximal degree
    
    Returns:
    A list of nodes
    """
    # copy the graph
    new_graph = copy_graph(ugraph)
    
    order = []    
    while len(new_graph) > 0:
        max_degree = -1
        for node in new_graph:
            if len(new_graph[node]) > max_degree:
                max_degree = len(new_graph[node])
                max_degree_node = node
        
        neighbors = new_graph[max_degree_node]
        new_graph.pop(max_degree_node)
        for neighbor in neighbors:
            new_graph[neighbor].remove(max_degree_node)

        order.append(max_degree_node)
    return order
    


##########################################################
# Code for loading computer network graph

NETWORK_URL = "http://storage.googleapis.com/codeskulptor-alg/alg_rf7.txt"


def load_graph(graph_url):
    """
    Function that loads a graph given the URL
    for a text representation of the graph
    
    Returns a dictionary that models a graph
    """
    graph_file = urllib2.urlopen(graph_url)
    graph_text = graph_file.read()
    graph_lines = graph_text.split('\n')
    graph_lines = graph_lines[ : -1]
    
    print "Loaded graph with", len(graph_lines), "nodes"
    
    answer_graph = {}
    for line in graph_lines:
        neighbors = line.split(' ')
        node = int(neighbors[0])
        answer_graph[node] = set([])
        for neighbor in neighbors[1 : -1]:
            answer_graph[node].add(int(neighbor))

    return answer_graph


def make_complete_graph(num_nodes):
	"""
	Takes the number of nodes num_nodes and returns a dictionary corresponding to a complete directed graph with the specified number of nodes. A complete graph contains all possible edges subject to the restriction that self-loops are not allowed. 
	"""
	if num_nodes <= 0:
		return {}
	dict_graph = {}
	for node in range(num_nodes):
		node_set = set()
		for neighbor in range(num_nodes):
			if node != neighbor:
				node_set.add(neighbor)
		dict_graph[node] = node_set

	return dict_graph


def gen_er_graph(num_nodes, probability):
	""" pseudocode:
	algorithm for generating random undirected graphs (ER graphs)
	Algorithm 1:  ER
	Input: Number of nodes n; probability p,
	Output:  A graph g = (V, E) where g is an element of G(n, p)
  	1  V <-- {0, 1, ... n-1};
  	2  E <-- null;
  	3  foreach {i, j} that is a unique element of V, where i is not j do
  	4      a <-- random(0, 1);    // a is a random real number in [0, 1)
  	5      if a < p then
  	6          E <-- E union {{i, j}};
  	7  return g = (V, E)

  	You may wish to modify your implementation of make_complete_graph from Project 1 to add edges randomly
  	"""
	if num_nodes <= 0:
	 	return {}
  	dict_graph = {}

  	for node in range(num_nodes):
  		node_set = set()	# edges
  		for neighbor in range(num_nodes):
  			a = random.random()			
  			if node != neighbor and a < probability:
  				node_set.add(neighbor)
  		dict_graph[node] = node_set

  	return dict_graph


def test_gen_er_graph():
	""" tests gen_er_graph function """
	print "num_nodes = 0, probability = .004"
	print gen_er_graph(0, .004)
	print "num_nodes = 5, probability = .004"
	print gen_er_graph(5, .004)
	print "num_nodes = 10, probability = .004"
	print gen_er_graph(10, .004)
	print "num_nodes = 10, probability = .5"
	print gen_er_graph(10, .5)
	# print gen_er_graph(1200, .004)


def gen_upa_graph(final_nodes, num_nodes, probability):
	""" 
	generates a random undirected graph iteratively, where in each iteration a new node is created and added to the graph, connected to a subset of the existing nodes.  The subset is chosen based on in-degrees of existing nodes. 

	n: final_nodes
	m: num_nodes
	"""
	if num_nodes > final_nodes or final_nodes < 1:
		return {}
		
	# create random undirected graph on m nodes (num_nodes)
	graph = gen_er_graph(num_nodes, probability)

	V = []
	E = []
	total_indeg = 0

	for key in graph:
		V.append(key)
		E.append([key,graph[key]])	

	# grow the graph by adding n - m (final_nodes - num_nodes) nodes
	# where each new node is connected to m nodes randomly chosen 
	# from the set of existing nodes.  Elimintate duplicates 
	# to avoid parallel edges.
	for node_added in range(num_nodes, final_nodes):
		# for key in graph:
		# 	for value in graph[key]:
		# 		total_indeg += value
		V_prime = set()
		# choose randomly m nodes from V and add them to V_prime 
		# where the probability of choosing node j is (indeg(j) + 1)/(totindeg + |V|)
		# i.e., call DPATrial (line 6 in pseudocode)
		trial = UPATrial.UPATrial(num_nodes)
		V_prime = trial.run_trial(num_nodes)
		for node in V_prime:
			V_prime.add(node)
		V.append(node_added)
		graph[node_added] = V_prime
	return graph


def random_order(graph):
	""" returns a list of nodes in the graph in a random order """
	keys_list = []

	for key in graph:
		keys_list.append(key)
	random.shuffle(keys_list)
	return keys_list


def test_random_order():
	""" tests random_order function """
	graph = {0: {1, 2},
			 1: {0},
			 2: {0},
			 3: {4},
			 4: {3}}

	print "graph:", graph
	ro_graph = random_order(graph)
	print "random_order graph:", ro_graph


def bfs_visited(ugraph, start_node):
	""" takes undirected graph and node as input;
	returns set of all nodes visited by breadth-first search starting at start_node """
	queue = deque()
	visited = set([start_node])

	queue.append(start_node)
	# print "start_node:", start_node

	while len(queue) > 0:
		current_node = queue.pop()
		# print "current_node:", current_node
		if current_node in ugraph:
			for neighbor in ugraph[current_node]:
				if neighbor not in visited:
					visited.add(neighbor)
					queue.append(neighbor)
	return visited


def cc_visited(ugraph):
	""" takes ugraph and returns list of sets where each set consists of all the nodes (and nothing else) in a connected component and there is exactly one set in the list for each connected component in ugraph and nothing else
	"""
	remaining_nodes = []

	for node in ugraph:
		remaining_nodes.append(node)
	c_comp = []

	while len(remaining_nodes) > 0:
		current_node = remaining_nodes.pop()
		working_set = bfs_visited(ugraph, current_node)
		c_comp.append(working_set)
		dummyvar = [remaining_nodes.remove(ws_item) for ws_item in working_set if ws_item in remaining_nodes]
	return c_comp


def largest_cc_size(ugraph):
	""" computes and returns the size of the largest connected component of ugraph; returns an int """
	largest_num = 0
	c_comp = cc_visited(ugraph)

	for group in c_comp:
		if len(group) > largest_num:
			largest_num = len(group)
	return largest_num


def compute_resilience(ugraph, attack_order):
	""" takes an undirected graph and a list of nodes, attack_order;
	iterates through the nodes in attack_order.
	For each node, removes the given node and its edges from ugraph and then computes the size of the largest cc for the resulting graph.  
	returns a list whose k + 1th entry is the size of the largest cc in the graph after removal of the first k nodes in attack_order.  The first entry in the returned list is the size of the largest cc in the original graph"""
	resilience = []
	resilience.append(largest_cc_size(ugraph))
	# iterate through nodes in attack_order
	for target in attack_order:
		# in order to remove given node and its edges from ugraph
		# first create a list of neighbor nodes to visit for removal of edges
		neighbors = bfs_visited(ugraph, target)
		# then visit each neighbor, removing target from its list of neighbors
		for neighbor in neighbors:
			if neighbor in ugraph and target in ugraph[neighbor]:
				ugraph[neighbor].remove(target)
				# delete_node(ugraph, target)
		# next remove the target node
		# del ugraph[target]
		delete_node(ugraph, target)
		# compute size of largest cc for resulting graph
		largest_cc = largest_cc_size(ugraph)
		# append largest cc to result list
		resilience.append(largest_cc)

	# return result list
	# print "\nresilience:", resilience
	return resilience

# Begin Test calls -------------------------------------
# test random_order
# test_random_order()
# test_gen_er_graph()
# End Test calls ---------------------------------------

# Questions---------------------------------------------
# 1.  probability = .004 m = 2 (# edges) 
probability = .004
num_nodes = 1200
final_nodes = 1239
# for each of the 3 graphs, compute a random attack order using random_order and use this attack order in compute_resilience to compute the resilience of each graph
# # network ------------------
network_graph = load_graph(NETWORK_URL)
attack_order_net = random_order(network_graph)
resilience_net = compute_resilience(network_graph, attack_order_net)
# ER -----------------------
er_graph = gen_er_graph(num_nodes, probability)
attack_order_er = random_order(er_graph)
resilience_er = compute_resilience(er_graph, attack_order_er)
# # UPA ----------------------
upa_graph = gen_upa_graph(final_nodes, num_nodes, probability)
attack_order_upa = random_order(upa_graph)
resilience_upa = compute_resilience(upa_graph, attack_order_upa)

# # plot all three resilience curves on a single standard plot (not log/log)
xvals = range(num_nodes)
xvals = range(len(resilience_net))
xvals2 = range(len(resilience_er))
xvals3 = range(len(resilience_upa))
yvals1 = resilience_net
yvals2 = resilience_er
yvals3 = resilience_upa

plt.plot(xvals, yvals1, '-b', label='Network')
plt.plot(xvals2, yvals2, '-r', label='ER')
plt.plot(xvals3, yvals3, '-g', label='UPA')
plt.legend(loc='upper right')
plt.show()
