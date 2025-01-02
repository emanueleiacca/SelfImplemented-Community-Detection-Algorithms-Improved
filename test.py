from collections import defaultdict
from functions import Community_detection_algorithms
# Graph Class (Adjacency List Implementation)
class Graph:
    def __init__(self, edges):
        self.graph = defaultdict(set)
        self.edges = edges
        self.nodes = set()
        for u, v in edges:
            self.add_edge(u, v)

    def add_edge(self, u, v):
        self.graph[u].add(v)
        self.graph[v].add(u)
        self.nodes.update([u, v])

    def degree(self, node):
        return len(self.graph[node])

    def adjacency_matrix(self):
        nodes = sorted(self.nodes)
        n = len(nodes)
        node_idx = {node: i for i, node in enumerate(nodes)}
        adj_matrix = np.zeros((n, n))
        for u in nodes:
            for v in self.graph[u]:
                adj_matrix[node_idx[u], node_idx[v]] = 1
        return adj_matrix, nodes
    
    def adjacency(self):
        for node, neighbors in self.graph.items():
            yield node, list(neighbors)

def calculate_modularity(adj_matrix, communities):
    """
    It evaluates the modularity of the resulting communities.
    """
    m = np.sum(adj_matrix) / 2
    Q = 0.0
    for community in communities:
        # For every pair of nodes in the same community i,j
        for i in community:
            for j in community:
                A_ij = adj_matrix[i, j] # Adjacency matrix value
                k_i = np.sum(adj_matrix[i]) # Degree of node i
                k_j = np.sum(adj_matrix[j]) # Degree of node j
                Q += A_ij - (k_i * k_j) / (2 * m) # Modularity formula numerator
    return Q / (2 * m) # Normalize by 2m

# Test the Girvan-Newman Algorithm

import time
from networkx.algorithms.community import girvan_newman as nx_girvan_newman

# --- Custom Community Detection Algorithms ---
def test_algorithms(graph, custom_method, component_method):
    adj_matrix, nodes = graph.adjacency_matrix()  # Extract the adjacency matrix
    print(f"\n ⚠️--- Testing with Custom Method: {custom_method} | Component Method: {component_method} ---")
    
    # Measure time for Custom Girvan-Newman
    try:
        start_time = time.time()
        custom_communities, custom_removed_edges = Community_detection_algorithms(graph, custom_method, component_method)
        end_time = time.time()
        custom_time = end_time - start_time
        print(f"⏱️ Custom Girvan-Newman Time: {custom_time:.4f} seconds")
        print("Custom Communities:", custom_communities)
        print("Modularity:", calculate_modularity(adj_matrix, custom_communities))
        print("Custom Removed Edges:", custom_removed_edges)
    except NotImplementedError as e:
        print(f"Custom Girvan-Newman: {e}")
        custom_time = None

    # Measure time for NetworkX Girvan-Newman
    print("\n ⚠️Running NetworkX Girvan-Newman Algorithm...")
    start_time = time.time()
    nx_gen = nx_girvan_newman(nx.Graph(graph.graph))  # Generate community splits
    nx_communities = next(iter(nx_gen))  # Extract first split
    end_time = time.time()
    nx_time = end_time - start_time
    nx_communities = [sorted(list(c)) for c in nx_communities]  # Sort nodes in each community
    print(f"⏱️ NetworkX Girvan-Newman Time: {nx_time:.4f} seconds")
    print("NetworkX Communities:", nx_communities)

    # Comparison
    if custom_time:
        print("\n--- Comparison Results ---")
        if sorted([sorted(c) for c in custom_communities]) == sorted(nx_communities):
            print("✅ Communities match!")
        else:
            print("❌ Communities do NOT match!")
        print(f"⏱️ Time Difference: {abs(custom_time - nx_time):.4f} seconds")
