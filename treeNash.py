import itertools
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
        adjacency[i] = row
    return number_nodes, threshold, adjacency