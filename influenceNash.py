import sys
import networkx as nx

n = 0
matrix = [] # Renamed from graph to avoid conflict with nx.Graph
b = []

def get_input(graph_file_name):
    global n, matrix, b
    
    with open(graph_file_name, "r") as f:
        data = f.read().split()
        data_index = 0
        
        # ead number of noeds
        n = int(float(data[data_index])) 
        data_index += 1
        matrix = [[0.0 for _ in range(n)] for _ in range(n)] # Initialize arrays
        b = [0.0 for _ in range(n)]
    
        for i in range(n):
            for j in range(n):
                # Read the influence weight from i to j
                matrix[i][j] = float(data[data_index])
                data_index += 1
            
            # read i's threshold
            b[i] = float(data[data_index])
            data_index += 1

def convert_to_networkx(): #have oto
    global n, matrix, b
    G = nx.Graph()
    
    for i in range(n):
        node_id = i + 1  # Using 1 based indexing
        G.add_node(node_id)
        G.nodes[node_id]['threshold'] = b[i]
        G.nodes[node_id]['dict'] = {} 

    # we store the incoming weights (influence neighbor -> me) on the edges
    for i in range(n):
        for j in range(i + 1, n): 
            # if we have a connection
            if matrix[i][j] != 0 or matrix[j][i] != 0:
                u, v = i + 1, j + 1
                G.add_edge(u, v)
                
    return G

def downstream(G, curr_node, prev_node): #changed from treenash because we needed to check thresholds for conformance
    node_dict = {}
    neighbors = list(G.neighbors(curr_node))
    
    # don't recurse on the parent
    if prev_node in neighbors:
        neighbors.remove(prev_node)

    for neighbor in neighbors:
        downstream(G, neighbor, curr_node)

    # Retrieve explicit threshold for this specifc node
    node_threshold = G.nodes[curr_node]['threshold']

    for i in range(2**(len(neighbors)+2)):
        binary = f"{i:0{len(neighbors)+2}b}"
        working_values = True
        current_influence = 0.0 #we create thee wight here
        
        for j, child in enumerate(neighbors): #caclualte influence from children
            if binary[j] == '1':
                weight = matrix[child-1][curr_node-1]
                current_influence += weight

        parent_val = binary[-2] #get infleunce from parent
        if parent_val == '1':
            weight = matrix[prev_node-1][curr_node-1]
            current_influence += weight
        curr_value = binary[-1]
        
        if current_influence >= node_threshold and curr_value == '0': #if influence greater than or equal to threshold gotta play 1
            working_values = False
        if current_influence < node_threshold and curr_value == '1': #if influence less than threshold gotta play 0
            working_values = False
        
        for j, child in enumerate(neighbors):
            child_action = binary[j]
            # Look up in child's dict
            if (child_action, curr_value) not in G.nodes[child]['dict'].keys():
                working_values = False

        if working_values:
            prev_value = binary[-2]
            key = (curr_value, prev_value)
            
            if key in node_dict:
                node_dict[key].add(tuple(binary[:-2]))
            else:
                node_dict[key] = {tuple(binary[:-2])}
    
    G.nodes[curr_node]['dict'] = node_dict

def downstream_root(G, curr_node): #downstream changes because of thresholds, need to check conformance
    node_dict = {}
    neighbors = list(G.neighbors(curr_node))
    node_threshold = G.nodes[curr_node]['threshold']

    for neighbor in neighbors:
        downstream(G, neighbor, curr_node)

    for i in range(2**(len(neighbors)+1)):
        binary = f"{i:0{len(neighbors)+1}b}"
        working_values = True
        
        current_influence = 0.0 #get influnece from children
        
        for j, child in enumerate(neighbors):
            if binary[j] == '1':
                weight = matrix[child-1][curr_node-1]
                current_influence += weight

        curr_value = binary[-1]

        #if influence greater than or equal to threshold gotta play 1
        if current_influence >= node_threshold and curr_value == '0': 
            working_values = False
        #if influence less than threshold gotta play 0
        if current_influence < node_threshold and curr_value == '1':
            working_values = False
        
        for j, child in enumerate(neighbors): #check children again for consistency
            child_action = binary[j]
            if (child_action, curr_value) not in G.nodes[child]['dict'].keys():
                working_values = False

        if working_values:
            key = (curr_value)
            if key in node_dict:

                node_dict[key].add(tuple(binary[:-1]))
            else:

                node_dict[key] = {tuple(binary[:-1])}

    G.nodes[curr_node]['dict'] = node_dict

def upstream_traversal(G, curr_node, prev_node, curr_action, prev_action): #same thing as regualr tree nash because we know thresholds are already enforced
    """
    Recursively reconstructs the full game states from the witness lists.
    Returns a list of dictioanries partial Nash Equilibria.
    """
    solutions = []
    if prev_node is None: #if were at root don't need a previous action
        key = str(curr_action)
    else:
        key = (str(curr_action), str(prev_action))

    if key not in G.nodes[curr_node]['dict']: #if there are no chlidren that can play the current action
        return []
    
    valid_children_tuples = G.nodes[curr_node]['dict'][key]
    children = [n for n in G.neighbors(curr_node) if n != prev_node]

    for child_config in valid_children_tuples: #tupels represents a valid configuration of children
        branch_solutions = [{curr_node: curr_action}]

        for i, child in enumerate(children): #each child as an action
            child_action = int(child_config[i])
            child_sub_solutions = upstream_traversal(G, child, curr_node, child_action, curr_action)
            
            if not child_sub_solutions: #no valid solutions for this child
                branch_solutions = []
                break

            new_branch_solutions = []

            for existing_sol in branch_solutions: #each existing solution
                
                for child_sol in child_sub_solutions: #each valid solution for the child
                    merged = existing_sol.copy()
                    merged.update(child_sol)
                    new_branch_solutions.append(merged)
            branch_solutions = new_branch_solutions
        
        solutions.extend(branch_solutions) #add all valid solutiosn for this branch

    return solutions

def solve_nash(graph): #solve for nash eq
    root = 1
    final_equilibria = []
    
    for root_action in [0, 1]:
        paths = upstream_traversal(graph, root, None, root_action, None)
        final_equilibria.extend(paths)
        
    return final_equilibria

def main(file):
    print(f"Reading {file}...")
    get_input(file)
    graph = convert_to_networkx()
    print(f"Nodes: {graph.nodes()}")
    
    root = 1
    downstream_root(graph, root)
    equilibria = solve_nash(graph)
    
    print(f"\nFound {len(equilibria)} Nash Equilibria:")
    for i, eq in enumerate(equilibria):
        print(f"Eq {i+1}: {dict(sorted(eq.items()))}")

if __name__ == "__main__":
        main(sys.argv[1])