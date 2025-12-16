import itertools
import sys
import networkx as nx

def parse(file):
    graph = nx.Graph()
    with open(file, 'r') as f:
        lines = f.readlines()
    number_nodes = int(lines[0].strip())
    threshold = float(lines[1].strip()) #fraction of neighbors to play one to keep one
    adjacency = {}
    for i, line in enumerate(lines[2:2 + number_nodes]):
        row = list(map(int, line.strip().split()))
        adjacency[i+1] = row
    for key, value in adjacency.items():
        for neighbor in value:
            graph.add_edge(key, neighbor)
    print(graph.edges())
    print(graph.nodes())
    return graph, threshold

def downstream(G, threshold, curr_node, prev_node):
    """
    Recursively perform the downsteam pass of TreeNash, assuming that this node is not the root.
    Add the dictionary of possible values as an attribute of the node in the graph.
    """
    
    node_dict = {}
    neighbors = list(G.neighbors(curr_node))
    node_threshold = threshold * len(neighbors)
    # Don't recurse on the node we've already seen
    neighbors.remove(prev_node)

    for neighbor in neighbors:
        # Recurse on all unseen neighbors
        downstream(G, threshold, neighbor, curr_node)

    for i in range(2**(len(neighbors)+2)):
        # Try all possible assignments for the node and all its neighbors, given by a binary string
        # Where each digit corresponds to a node's assignment
        binary = f"{i:0{len(neighbors)+2}b}"
        working_values = True
        digit_sum = 0

        # Ignore the assignment if the current node isn't playing its best response.
        for digit in binary[:-1]:
            digit_sum += int(digit)
        if digit_sum >= node_threshold and binary[-1] == '0':
            working_values = False
        if digit_sum < node_threshold and binary[-1] == '1':
            working_values = False
        
        # The previous string's assignment is the second to last in the string
        prev_value = binary[-2]
        # The current string's assignment is the last in the string
        curr_value = binary[-1]

        # Check if the assignment for each neighbor is possible given the current node's assignment
        for j, neighbor in enumerate(neighbors):
            if (binary[j], curr_value) not in G.nodes[neighbor]['dict'].keys():
                working_values = False

        # If this is a valid assignment, add it to the dict
        if working_values:
            if (curr_value, prev_value) in node_dict.keys():
                node_dict[(curr_value, prev_value)].add(tuple(binary[:-2]))
            else:
                node_dict[(curr_value, prev_value)] = {tuple(binary[:-2])}
    
    # Add the dict to the graph as an attribute of the current node.
    G.nodes[curr_node]['dict'] = node_dict

def downstream_root(G, threshold, curr_node):
    """
    Recursively perform the downsteam pass of TreeNash, assuming that this node is the root.
    Add the dictionary of possible values as an attribute of the root in the graph.
    """

    node_dict = {}
    neighbors = list(G.neighbors(curr_node))
    node_threshold = threshold * len(neighbors)

    for neighbor in neighbors:
        # Recurse on all neighbors
        downstream(G, threshold, neighbor, curr_node)

    for i in range(2**(len(neighbors)+1)):
        # Try all possible assignments for the node and all its neighbors, given by a binary string
        # Where each digit corresponds to a node's assignment
        binary = f"{i:0{len(neighbors)+1}b}"
        working_values = True
        digit_sum = 0

        # Ignore the assignment if the current node isn't playing its best response.
        for digit in binary[:-1]:
            digit_sum += int(digit)
        if digit_sum >= node_threshold and binary[-1] == '0':
            working_values = False
        if digit_sum < node_threshold and binary[-1] == '1':
            working_values = False
        
        # The current string's assignment is the last in the string
        curr_value = binary[-1]

        # Check if the assignment for each neighbor is possible given the current node's assignment
        for j, neighbor in enumerate(neighbors):
            if (binary[j], curr_value) not in G.nodes[neighbor]['dict'].keys():
                working_values = False

        # If this is a valid assignment, add it to the dict
        if working_values:
            if (curr_value) in node_dict.keys():
                node_dict[(curr_value)].add(tuple(binary[:-1]))
            else:
                node_dict[(curr_value)] = {tuple(binary[:-1])}

    # Add the dict to the graph as an attribute of the root.
    G.nodes[curr_node]['dict'] = node_dict

def upstream_traversal(G, curr_node, prev_node, curr_action, prev_action):
    """
    Recursively reconstructs the full game states from the witness lists.
    Returns a list of dictionaries (partial Nash Equilibria).
    """
    solutions = []
    
    # 1. Retrieve the valid children configurations for this specific (Self, Parent) state
    # Note: Keys in your dict are strings, so we cast actions to str
    if prev_node is None:
        # Root Logic: Key is just current_action
        key = str(curr_action)
    else:
        # Internal/Leaf Logic: Key is (current_action, prev_action)
        key = (str(curr_action), str(prev_action))

    # If this specific configuration was pruned during downstream, return empty (invalid path)
    if key not in G.nodes[curr_node]['dict']:
        return []

    valid_children_tuples = G.nodes[curr_node]['dict'][key]

    # 2. Get the list of children (neighbors excluding parent)
    # CRITICAL: Must use same order as downstream pass
    children = [n for n in G.neighbors(curr_node) if n != prev_node]

    # 3. Iterate through every valid configuration of children
    for child_config in valid_children_tuples:
        # child_config is a tuple like ('1', '0') corresponding to children
        
        # Start this branch's solution with the current node's action
        branch_solutions = [{curr_node: curr_action}]

        # 4. For each child, get their sub-solutions and merge (Cartesian Product)
        for i, child in enumerate(children):
            child_action = int(child_config[i])
            
            # Recurse: Get all valid sub-trees for this child
            child_sub_solutions = upstream_traversal(G, child, curr_node, child_action, curr_action)
            
            # If any child returns no valid solutions, this whole config is dead
            if not child_sub_solutions:
                branch_solutions = []
                break
            
            # Cartesian Product: Merge current branch solutions with child's sub-solutions
            # If we had 2 ways to solve left branch and 2 ways to solve right, we now have 4 total
            new_branch_solutions = []
            for existing_sol in branch_solutions:
                for child_sol in child_sub_solutions:
                    merged = existing_sol.copy()
                    merged.update(child_sol)
                    new_branch_solutions.append(merged)
            branch_solutions = new_branch_solutions
        
        solutions.extend(branch_solutions)

    return solutions

def solve_nash(graph):
    # Assuming Node 1 is Root
    root = 1
    final_equilibria = []
    
    # Try setting Root to 0 and 1
    for root_action in [0, 1]:
        # Pass None as prev_node for Root
        paths = upstream_traversal(graph, root, None, root_action, None)
        final_equilibria.extend(paths)
        
    return final_equilibria

def main():
    graph, threshold = parse('second_test.txt')
    print(list(graph.neighbors(1)))
    root = 1
    downstream_root(graph, threshold, root)
    for node in graph:
        print(node)
        print(graph.nodes[node]['dict'])
    print("\n--- Calculating Global Equilibria ---")
    equilibria = solve_nash(graph)
    print(f"Found {len(equilibria)} Nash Equilibria:")
    for i, eq in enumerate(equilibria):
        print(f"Eq {i+1}: {dict(sorted(eq.items()))}")
main()


