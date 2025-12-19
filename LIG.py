#Binary action (0, 1) linear influence game with n players on a tree structure

# Global variables
n = 0
graph = []
b = []

# Get an influence game as input
def get_input(graph_file_name):
    global n, graph, b
    
    with open(graph_file_name, "r") as f:
        data = f.read().split()
        data_index = 0
        
        # Read first number: number of nodes
        n = int(data[data_index])
        data_index += 1
        
        # Initialize arrays based on n
        graph = [[0.0 for _ in range(n)] for _ in range(n)]
        b = [0.0 for _ in range(n)]
        
        # Read influence weights and node thresholds
        for i in range(n):
            for j in range(n):
                # Read the influence weight from i to j
                graph[i][j] = float(data[data_index])
                data_index += 1
            
            # Read i's threshold
            b[i] = float(data[data_index])
            data_index += 1




import random

def generate_tree_input(filename):
    n = 7  # 7 nodes (0-6)
    
    # Define the tree structure based on your image
    # Edges: (0,1), (0,2), (1,3), (1,4), (2,5), (2,6)
    edges = [
        (0, 1), (0, 2),  # Node 0 connected to 1 and 2
        (1, 3), (1, 4),  # Node 1 connected to 3 and 4
        (2, 5), (2, 6)   # Node 2 connected to 5 and 6
    ]
    


    # Initialize influence matrix (all zeros initially)
    graph = [[0.0 for _ in range(n)] for _ in range(n)]
    
    """
    # For graphical games (binary action): for each action profile over neighborhood (including i) define a payoff for i
    for i in range(n):
        for act_profile in range(2**(1+degree[i])):
            payoff[i][act_profile] = round(random.uniform(0.0, 1.0), 3)
    """

    # Add random influence weights for connected nodes
    for i, j in edges:
        # Random weights between 0 and 1 for both directions
        graph[i][j] = round(random.uniform(0.0, 1.0), 3)
        graph[j][i] = round(random.uniform(0.0, 1.0), 3)
        
    
    # Generate random thresholds between 0 and 1
    thresholds = [round(random.uniform(0.0, 1.0), 3) for _ in range(n)]
    
    # Write to file
    with open(filename, 'w') as f:
        # Write number of nodes
        f.write(f"{n}\n")
        
        # Write influence matrix and thresholds
        for i in range(n):
            # Write row i of influence matrix
            for j in range(n):
                f.write(f"{graph[i][j]} ")
            # Write threshold for node i
            f.write(f"{thresholds[i]}\n")
    
    print(f"Generated random tree input file: {filename}")
    print(f"Tree structure: {edges}")
    print(f"Thresholds: {thresholds}")






# Generate the file
random.seed(42)  # For reproducible results
generate_tree_input("tree_game.txt")

# Also create a function to print the generated data nicely
def print_generated_data(filename):
    print(f"\nContents of {filename}:")
    with open(filename, 'r') as f:
        content = f.read()
        print(content)

print_generated_data("tree_game.txt")


"""
7
0.0 0.639 0.025 0.0 0.0 0.0 0.0 0.374
0.891 0.0 0.0 0.197 0.335 0.0 0.0 0.764
0.553 0.0 0.0 0.0 0.0 0.708 0.02 0.419
0.0 0.685 0.0 0.0 0.0 0.0 0.0 0.481
0.0 0.204 0.0 0.0 0.0 0.0 0.0 0.617
0.0 0.0 0.567 0.0 0.0 0.0 0.0 0.891
0.0 0.0 0.708 0.0 0.0 0.0 0.0 0.344
"""