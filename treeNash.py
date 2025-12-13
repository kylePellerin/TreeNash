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
        graph.add_node(i, state=random_state, matrix={}, num_parents=0, parents = [])
    
    for i, line in enumerate(lines[2:2 + number_nodes]):
        row = list(map(int, line.strip().split()))
        adjacency[i+1] = row
    print("adjacency", adjacency)
    initialize_adjacency_matrix(adjacency)
    print("edges", graph.edges())

    for node in graph.nodes():
        for a, b in graph.edges():
            if b == node:
                graph.nodes[node]['num_parents'] += 1
                graph.nodes[node]['parents'].append(a)
                
    # Initialize Existence for proof
    for node in graph.nodes():
        print("node:", node)
        if len(get_children(node)) == 0:
            continue 
        else:
            parent_states = generate_binary_combinations(graph.nodes[node]['num_parents'])
            print("num_parents:", graph.nodes[node]['num_parents'])
            print(len(get_children(node)))
            children_states = generate_binary_combinations(len(get_children(node)))

            print("parent_states:", parent_states)
            print("children_states:", children_states)
            for states in children_states:
                for parent_state in parent_states:
                    # Parent plays 0 
                    ones = sum(states[1:] + parent_state)
                    print("parent_state:", parent_state, "states:", states)
                    if ones / (len(parent_state) + states[1:]) >= threshold:
                        graph.nodes[node]['matrix'][states] = (0, states[1:])

                    # Parent plays 1
                    ones_plus_parent = sum(states[1:] + parent_state + 1)
                    if ones_plus_parent / (len(parent_state) + states[1:] + 1) >= threshold:
                        graph.nodes[node]['matrix'][states] = (1, states[1:])

        print("nodes: ", graph.nodes(data=True))

        # else:
        #     parent_states = generate_binary_combinations(graph.nodes[node]['num_parents'])
        #     for child in get_children(node):
        #         for states in parent_states:
        #             ones = sum(states)
        #             zeros = len(states) - ones
        #             if ones / len(states) >= threshold:
        #                 graph.nodes[child]['matrix'][states] = (1, None)
        #             else:
        #                 graph.nodes[child]['matrix'][states] = 0

def get_children(node):
    children = []
    for a, b in graph.edges():
        if a == node:
            children.append(b)
    return children 
"""
nodes with num_parents 
[
    (1, {'state': 1, 'matrix': {}, 'num_parents': 0, 'parents': []}), 
    (2, {'state': 1, 'matrix': {}, 'num_parents': 1, 'parents': [1]}), 
    (3, {'state': 0, 'matrix': {}, 'num_parents': 1, 'parents': [1]}), 
    (4, {'state': 0, 'matrix': {}, 'num_parents': 1, 'parents': [3]})]
None
"""



def generate_binary_combinations(n):
    """Generate all possible combinations of 0s and 1s for n positions."""
    combinations = []
    
    # Generate all numbers from 0 to 2^n - 1
    for i in range(2**n):
        # Convert to binary and pad with zeros
        binary_str = format(i, f'0{n}b')
        # Convert to tuple of integers
        combination = tuple(int(bit) for bit in binary_str)
        combinations.append(combination)
    
    return combinations



if __name__ == "__main__":
    print(parse("input_test.txt"))
    # print(parse(sys.argv[1]))