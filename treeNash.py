import itertools
from random import random
import sys
import networkx as nx

graph = nx.Graph()
def parse(file):
    with open(file, 'r') as f:
        lines = f.readlines()
    number_nodes = int(lines[0].strip())
    threshold = float(lines[1].strip()) #fraction of neighbors to play one to keep one
    adjacency = {}
    for i, line in enumerate(lines[2:2 + number_nodes]):
        row = list(map(int, line.strip().split()))
        adjacency[i+1] = row
    initialize_adjacency_matrix(adjacency)
    print(graph.edges())



    return number_nodes, threshold, adjacency

def initialize_adjacency_matrix(adjacency):
    for key, value in adjacency.items():
        for neighbor in value:
            graph.add_edge(key, neighbor)

    # 
def parse(file):
    with open(file, 'r') as f:
        lines = f.readlines()
    number_nodes = int(lines[0].strip())
    threshold = float(lines[1].strip()) #fraction of neighbors to play one to keep one
    adjacency = {}

    for i in range(1, number_nodes + 1):
        random_state = 1 if random() < 0.5 else 0
        graph.add_node(i, state=random_state, matrix={})
    
    for i, line in enumerate(lines[2:2 + number_nodes]):
        row = list(map(int, line.strip().split()))
        adjacency[i+1] = row
    print("adjacency", adjacency)
    initialize_adjacency_matrix(adjacency)
    print("edges", graph.edges())
    print("nodes", graph.nodes(data=True))

    # Initialize payoff matrix for each node


    
    for node in graph.nodes():
        matrix = {}
        print("node", node)
        print("neighbors", list(graph.neighbors(node)))
        for neighbor in graph.neighbors(node):
            print()
            matrix[neighbor] = {
                (0, 0): 1,
                (0, 1): 0,
                (1, 0): 0,
                (1, 1): 1
            }
        graph.nodes[node]['matrix'] = matrix

"""
edges [(1, 2), (1, 3), (3, 4)]
nodes [(1, {'state': 1, 'matrix': {(0, 0): {(1, 1)}}}), 
       (2, {'state': 1, 'matrix': {}}), 
       (3, {'state': 1, 'matrix': {}}), 
       (4, {'state': 1, 'matrix': {}})]
"""

def count_parents(node):
    return len(list(graph.neighbors(node)))
    


if __name__ == "__main__":
    print(parse("input_test.txt"))
    # print(parse(sys.argv[1]))