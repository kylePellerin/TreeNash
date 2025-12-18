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
        n = int(float(data[data_index])) # Handle "7" or "7.0"
        data_index += 1
        
        # Initialize
        matrix = [[0.0 for _ in range(n)] for _ in range(n)]
        b = [0.0 for _ in range(n)]
        
        # Read matrix and thresholds
        for i in range(n):
            for j in range(n):
                matrix[i][j] = float(data[data_index])
                data_index += 1
            
            b[i] = float(data[data_index])
            data_index += 1

def convert_to_networkx():
    """
    Converts the Professor's Matrix/Globals into the NetworkX tree 
    our algorithm expects.
    """
    global n, matrix, b
    G = nx.Graph()
    
    # 1. Add Nodes with their specific Thresholds (b)
    for i in range(n):
        # User 1-based indexing for logic, Professor uses 0-based in file
        node_id = i + 1 
        G.add_node(node_id)
        G.nodes[node_id]['threshold'] = b[i]
        G.nodes[node_id]['dict'] = {} # Initialize witness dict

    # 2. Add Edges from Matrix
    # We assume the matrix represents a tree structure.
    for i in range(n):
        for j in range(i + 1, n): # Only check upper triangle to avoid duplicates
            # If there is a non-zero weight, an edge exists
            if matrix[i][j] != 0 or matrix[j][i] != 0:
                G.add_edge(i + 1, j + 1)
                
    return G

# ---------------------------------------------------------
# 2. OPTIMIZED TREENASH (Updated for explicit 'b' thresholds)
# ---------------------------------------------------------

def downstream_efficient(G, curr_node, prev_node):
    neighbors = list(G.neighbors(curr_node))
    if prev_node is not None:
        neighbors.remove(prev_node)
        
    for neighbor in neighbors:
        downstream_efficient(G, neighbor, curr_node)

    node_dict = {}
    
    # --- UPDATED: Retrieve explicit threshold 'b' for this node ---
    required_1s = G.nodes[curr_node]['threshold']
    
    my_actions = [0, 1]
    parent_actions = [0, 1] if prev_node is not None else [None]

    for my_action in my_actions:
        for parent_action in parent_actions:
            
            must_play_1 = []
            must_play_0 = []
            flexible = []
            impossible = False
            
            for child in neighbors:
                child_can_play_0 = (0, my_action) in G.nodes[child]['dict']
                child_can_play_1 = (1, my_action) in G.nodes[child]['dict']
                
                if child_can_play_1 and child_can_play_0:
                    flexible.append(child)
                elif child_can_play_1:
                    must_play_1.append(child)
                elif child_can_play_0:
                    must_play_0.append(child)
                else:
                    impossible = True
                    break
            
            if impossible:
                continue

            # Calculate Neighbors Sum
            parent_contribution = 1 if parent_action == 1 else 0
            
            # Note: This logic assumes Unweighted Neighbor Influence (Count)
            # If the professor wants WEIGHTED influence, 'len' must be replaced by 'sum of weights'
            total_min = len(must_play_1) + parent_contribution
            total_max = len(must_play_1) + len(flexible) + parent_contribution
            
            valid = False
            
            # Check Explicit Threshold 'b'
            if my_action == 0:
                if total_min < required_1s:
                    valid = True
            if my_action == 1:
                if total_max >= required_1s:
                    valid = True
            
            if valid:
                key = (my_action, parent_action) if prev_node is not None else (my_action)
                G.nodes[curr_node]['dict'][key] = {
                    'must_1': must_play_1,
                    'must_0': must_play_0,
                    'flexible': flexible
                }

def upstream_efficient(G, curr_node, prev_node, curr_action, prev_action):
    solutions = []
    
    key = (curr_action, prev_action) if prev_node is not None else (curr_action)
    
    if key not in G.nodes[curr_node]['dict']:
        return []
        
    data = G.nodes[curr_node]['dict'][key]
    must_1 = data['must_1']
    must_0 = data['must_0']
    flexible = data['flexible']
    
    # --- UPDATED: Use explicit threshold 'b' ---
    required_1s = G.nodes[curr_node]['threshold']
    
    parent_contribution = 1 if prev_action == 1 else 0
    current_fixed_1s = len(must_1) + parent_contribution
    
    valid_flexible_counts = []
    
    # Determine valid number of flexible children to set to 1
    for k in range(len(flexible) + 1):
        total_score = current_fixed_1s + k
        
        if curr_action == 0 and total_score < required_1s:
            valid_flexible_counts.append(k)
        elif curr_action == 1 and total_score >= required_1s:
            valid_flexible_counts.append(k)
            
    for k in valid_flexible_counts:
        children_playing_1 = must_1 + flexible[:k]
        children_playing_0 = must_0 + flexible[k:]
        
        current_branch_sols = [{curr_node: curr_action}]
        all_children = children_playing_1 + children_playing_0
        all_actions = [1]*len(children_playing_1) + [0]*len(children_playing_0)
        
        valid_branch = True
        for child, action in zip(all_children, all_actions):
            child_sols = upstream_efficient(G, child, curr_node, action, curr_action)
            if not child_sols:
                valid_branch = False
                break
            
            new_sols = []
            for existing in current_branch_sols:
                for cs in child_sols:
                    merged = existing.copy()
                    merged.update(cs)
                    new_sols.append(merged)
            current_branch_sols = new_sols
        
        if valid_branch:
            solutions.extend(current_branch_sols)

    return solutions

# ---------------------------------------------------------
# 3. POTENTIAL FUNCTION (Using Professor's Matrix)
# ---------------------------------------------------------

def calculate_potential_professor(equilibrium, G):
    """
    Calculates Prop 4.3 Potential using the globals 'matrix' and 'b'.
    """
    global matrix, b, n
    rho = 1
    sum_delta_x = 0
    sum_b_delta_x = 0
    
    # Map equilibrium {node_id: action} to list [-1, 1]
    # CAREFUL: equilibrium uses 1-based IDs, matrix uses 0-based index
    x_vector = [0] * n
    for node_id, action in equilibrium.items():
        val = 1 if action == 1 else -1
        x_vector[node_id - 1] = val

    # Calculate Deltas (Avg outgoing weight)
    deltas = []
    for i in range(n):
        row_weights = [matrix[j][i] for j in range(n) if i != j]
        avg_w = sum(row_weights) / len(row_weights) if row_weights else 1
        deltas.append(avg_w)

    for i in range(n):
        # Term 1: delta_i * x_i
        sum_delta_x += deltas[i] * x_vector[i]
        
        # Term 2: b_i * delta_i * x_i
        sum_b_delta_x += b[i] * deltas[i] * x_vector[i]

    phi = rho * ( (sum_delta_x ** 2) - (2 * sum_b_delta_x) )
    return phi

# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------

def main(file):
    # 1. Use Professor's Parser
    print(f"Reading {file} using Professor's format...")
    get_input(file)
    
    # 2. Convert to NetworkX for the TreeNash Logic
    graph = convert_to_networkx()
    print(f"Converted to Tree. Nodes: {len(graph.nodes())}, Edges: {len(graph.edges())}")
    
    # 3. Run TreeNash
    root = 1 # Assuming node 1 is always the root in this logic
    
    print("Running Optimized Downstream...")
    downstream_efficient(graph, root, None)
    
    print("Running Optimized Upstream...")
    final_equilibria = []
    for root_action in [0, 1]:
        sols = upstream_efficient(graph, root, None, root_action, None)
        final_equilibria.extend(sols)
        
    print(f"\nFound {len(final_equilibria)} Nash Equilibria.")
    
    # 4. Calculate Potentials
    best_eq_index = -1
    max_potential = -float('inf')

    for i, eq in enumerate(final_equilibria):
        phi = calculate_potential_professor(eq, graph)
        if phi > max_potential:
            max_potential = phi
            best_eq_index = i + 1

    if best_eq_index != -1:
        print(f"Most Stable (Eq {best_eq_index}): Potential {max_potential:.4f}")
        # Print sorted by Node ID
        print(dict(sorted(final_equilibria[best_eq_index-1].items())))

if __name__ == "__main__":
    if len(sys.argv) > 1:
        main(sys.argv[1])
    else:
        # Default for testing
        print("Please provide a file generated by your professor's script.")