import sys
import networkx as nx

# ---------------------------------------------------------
# 1. PROFESSOR'S PARSER (Global Variables)
# ---------------------------------------------------------
n = 0
matrix = [] # Renamed from 'graph' to avoid conflict with nx.Graph
b = []

def get_input(graph_file_name):
    global n, matrix, b
    
    with open(graph_file_name, "r") as f:
        data = f.read().split()
        data_index = 0
        
        # Read number of nodes
        # Use int(float(...)) to handle "7.0" vs "7"
        n = int(float(data[data_index])) 
        data_index += 1
        
        # Initialize arrays
        matrix = [[0.0 for _ in range(n)] for _ in range(n)]
        b = [0.0 for _ in range(n)]
        
        # Read influence weights and node thresholds
        for i in range(n):
            for j in range(n):
                # Read the influence weight from i to j
                matrix[i][j] = float(data[data_index])
                data_index += 1
            
            # Read i's threshold
            b[i] = float(data[data_index])
            data_index += 1

def convert_to_networkx():
    """
    Converts the Professor's Matrix/Globals into the NetworkX tree.
    """
    global n, matrix, b
    G = nx.Graph()
    
    # 1. Add Nodes with their specific Thresholds (b)
    for i in range(n):
        node_id = i + 1  # Using 1-based indexing for your logic
        G.add_node(node_id)
        G.nodes[node_id]['threshold'] = b[i]
        G.nodes[node_id]['dict'] = {} 

    # 2. Add Edges from Matrix
    # We assume the matrix represents a tree structure.
    # Note: LIGs are directed by nature (influence), but your tree logic assumes undirected connectivity.
    # We will store the INCOMING weights (influence neighbor -> me) on the edges.
    for i in range(n):
        for j in range(i + 1, n): 
            # If there is a connection
            if matrix[i][j] != 0 or matrix[j][i] != 0:
                u, v = i + 1, j + 1
                G.add_edge(u, v)
                
    return G

# ---------------------------------------------------------
# 2. ORIGINAL LOGIC (Adapted for Weights)
# ---------------------------------------------------------

def downstream(G, curr_node, prev_node):
    """
    Recursively perform the downstream pass.
    Updated to calculate WEIGHTED sum against explicit node threshold.
    """
    node_dict = {}
    neighbors = list(G.neighbors(curr_node))
    
    # Don't recurse on the parent
    if prev_node in neighbors:
        neighbors.remove(prev_node)

    for neighbor in neighbors:
        downstream(G, neighbor, curr_node)

    # Retrieve explicit threshold for this specific node
    node_threshold = G.nodes[curr_node]['threshold']

    # Iterate through all binary combinations of (Children + Parent + Self)
    # Binary string length = len(children) + 1 (parent) + 1 (self) = len(neighbors) + 2
    for i in range(2**(len(neighbors)+2)):
        binary = f"{i:0{len(neighbors)+2}b}"
        working_values = True
        
        # --- NEW LOGIC: Calculate Weighted Influence ---
        current_influence = 0.0
        
        # 1. Add influence from Children (bits 0 to len-1)
        # binary[j] corresponds to neighbors[j]
        for j, child in enumerate(neighbors):
            if binary[j] == '1':
                # Influence FROM child TO curr_node
                # Convert 1-based IDs to 0-based matrix index
                weight = matrix[child-1][curr_node-1]
                current_influence += weight

        # 2. Add influence from Parent (bit -2)
        parent_val = binary[-2]
        if parent_val == '1':
            # Influence FROM parent TO curr_node
            weight = matrix[prev_node-1][curr_node-1]
            current_influence += weight
            
        # 3. Identify My Action (bit -1)
        curr_value = binary[-1]

        # --- Check Best Response (Nash Condition) ---
        # If influence >= threshold, I MUST play 1. If I play 0, it's invalid.
        if current_influence >= node_threshold and curr_value == '0':
            working_values = False
        # If influence < threshold, I MUST play 0. If I play 1, it's invalid.
        if current_influence < node_threshold and curr_value == '1':
            working_values = False
        
        # 4. Check Consistency with Children's Dicts
        for j, child in enumerate(neighbors):
            child_action = binary[j]
            # Look up in child's dict: Key is (ChildAction, ParentAction) -> (binary[j], curr_value)
            if (child_action, curr_value) not in G.nodes[child]['dict'].keys():
                working_values = False

        # If valid, add to dict
        if working_values:
            prev_value = binary[-2]
            key = (curr_value, prev_value)
            
            if key in node_dict:
                node_dict[key].add(tuple(binary[:-2]))
            else:
                node_dict[key] = {tuple(binary[:-2])}
    
    G.nodes[curr_node]['dict'] = node_dict

def downstream_root(G, curr_node):
    """
    Recursively perform the downstream pass for the Root.
    """
    node_dict = {}
    neighbors = list(G.neighbors(curr_node))
    node_threshold = G.nodes[curr_node]['threshold']

    for neighbor in neighbors:
        downstream(G, neighbor, curr_node)

    # Binary string length = len(children) + 1 (self)
    for i in range(2**(len(neighbors)+1)):
        binary = f"{i:0{len(neighbors)+1}b}"
        working_values = True
        
        # --- Calculate Weighted Influence ---
        current_influence = 0.0
        
        for j, child in enumerate(neighbors):
            if binary[j] == '1':
                weight = matrix[child-1][curr_node-1]
                current_influence += weight

        curr_value = binary[-1]

        # --- Check Best Response ---
        if current_influence >= node_threshold and curr_value == '0':
            working_values = False
        if current_influence < node_threshold and curr_value == '1':
            working_values = False
        
        # Check Children Consistency
        for j, child in enumerate(neighbors):
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

def upstream_traversal(G, curr_node, prev_node, curr_action, prev_action):
    # This logic remains unchanged from your original solution
    solutions = []
    if prev_node is None: 
        key = str(curr_action)
    else:
        key = (str(curr_action), str(prev_action))

    if key not in G.nodes[curr_node]['dict']: 
        return []
    
    valid_children_tuples = G.nodes[curr_node]['dict'][key]
    children = [n for n in G.neighbors(curr_node) if n != prev_node]

    for child_config in valid_children_tuples: 
        branch_solutions = [{curr_node: curr_action}]

        for i, child in enumerate(children): 
            child_action = int(child_config[i])
            child_sub_solutions = upstream_traversal(G, child, curr_node, child_action, curr_action)
            
            if not child_sub_solutions: 
                branch_solutions = []
                break

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
    root = 1
    final_equilibria = []
    
    for root_action in [0, 1]:
        paths = upstream_traversal(graph, root, None, root_action, None)
        final_equilibria.extend(paths)
        
    return final_equilibria

# ---------------------------------------------------------
# 3. MAIN
# ---------------------------------------------------------
def main(file):
    # 1. Use Professor's Parser to fill globals
    print(f"Reading {file}...")
    get_input(file)
    
    # 2. Convert globals to NetworkX Graph
    graph = convert_to_networkx()
    print(f"Nodes: {graph.nodes()}")
    
    # 3. Run Your Original Algorithm (Weighted Version)
    root = 1
    
    print("Running Downstream...")
    downstream_root(graph, root)
    
    print("Running Upstream...")
    equilibria = solve_nash(graph)
    
    print(f"\nFound {len(equilibria)} Nash Equilibria:")
    for i, eq in enumerate(equilibria):
        print(f"Eq {i+1}: {dict(sorted(eq.items()))}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python goonNash_Influence_Final.py <tree_game.txt>")
    else:
        main(sys.argv[1])