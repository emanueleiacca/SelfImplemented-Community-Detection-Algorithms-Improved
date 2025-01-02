from collections import defaultdict
from functions import Girvan_Newman_methods
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
        custom_communities, custom_removed_edges = Girvan_Newman_methods(graph, custom_method, component_method)
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

# Pre implemented functions for Louvain and Spectral Clustering

import numpy as np
from community import community_louvain  # python-louvain package
from sklearn.cluster import SpectralClustering
import networkx as nx
from collections import defaultdict
import time

# Pre-implemented Spectral Clustering
def pre_implemented_spectral(graph, k):
    adj_matrix = nx.to_numpy_array(graph)
    nodes = list(graph.nodes())

    sc = SpectralClustering(n_clusters=k, affinity='precomputed', random_state=42)
    labels = sc.fit_predict(adj_matrix)
    
    communities = defaultdict(list)
    for i, label in enumerate(labels):
        communities[label].append(nodes[i])
    return list(communities.values())

# Pre-implemented Louvain Method
def pre_louvain_method(nx_graph):
    partition = community_louvain.best_partition(nx_graph)
    grouped = defaultdict(list)
    for node, comm in partition.items():
        grouped[comm].append(node)
    return list(grouped.values())

# Compare method and evaluate metrics
from tabulate import tabulate 

def compare_methods(custom_method, pre_method, name):
    print(f"\n--- Comparing {name} ---")
    print("Custom Communities:", custom_method)
    print("Pre-Implemented Communities:", pre_method)

    if sorted([sorted(c) for c in custom_method]) == sorted([sorted(c) for c in pre_method]):
        print(f"✅ {name} Communities Match!")
    else:
        print(f"❌ {name} Communities Do NOT Match!")

def compare_communities_overlap(custom, pre, name):

    print(f"\n--- {name} Overlap Comparison ---")

    overlap_data = []

    for i, custom_comm in enumerate(custom):
        overlap_scores = []
        for j, pre_comm in enumerate(pre):
            overlap = len(set(custom_comm) & set(pre_comm))
            overlap_scores.append((j, overlap))

        overlap_scores = sorted(overlap_scores, key=lambda x: -x[1])
        best_match = overlap_scores[0]
        
        overlap_data.append([
            f"Custom {i}", 
            f"Pre-Implemented {best_match[0]}",
            len(custom_comm),  # Custom community size
            len(pre[best_match[0]]),  # Best match size
            best_match[1]  # Overlap count
        ])

    headers = ["Custom Community", "Best Match", "Custom Size", "Best Match Size", "Overlap Nodes"]
    print(tabulate(overlap_data, headers=headers, tablefmt="grid"))

# --- Metrics Calculation Function ---
def evaluate_community_metrics(adj_matrix, communities, title):
    """
    Calculate and display Lambiotte Coefficient for nodes and Clauset's Parameter for communities.
    :param adj_matrix: NumPy adjacency matrix of the graph.
    :param communities: List of detected communities.
    :param title: Title for the display output.
    """
    n = adj_matrix.shape[0]
    node_degree = np.sum(adj_matrix, axis=1)

    # --- Calculate Lambiotte Coefficient ---
    lambiotte_coeff = {}
    for node in range(n):
        community = next(c for c in communities if node in c)
        internal_edges = np.sum(adj_matrix[node][community])
        lambiotte_coeff[node] = internal_edges / node_degree[node] if node_degree[node] > 0 else 0

    # --- Calculate Clauset's Parameter ---
    clauset_param = {}
    for i, community in enumerate(communities):
        internal_edges = sum(
            adj_matrix[node_i][node_j] for node_i in community for node_j in community if node_i != node_j
        )
        external_edges = sum(
            adj_matrix[node_i][node_j] for node_i in community for node_j in range(n) if node_j not in community
        )
        clauset_param[i] = internal_edges / (internal_edges + external_edges) if (internal_edges + external_edges) > 0 else 0

    # --- Display Results in Tabular Format ---
    print(f"\n--- {title} ---")
    print("\nLambiotte Coefficient (Node Importance):")
    lambiotte_table = [[node, f"{lambiotte_coeff[node]:.4f}"] for node in sorted(lambiotte_coeff.keys())]
    print(tabulate(lambiotte_table, headers=["Node", "Lambiotte Coefficient"], tablefmt="grid"))

    print("\nClauset's Parameter (Community Strength):")
    clauset_table = [[f"Community {i}", f"{clauset_param[i]:.4f}"] for i in clauset_param.keys()]
    print(tabulate(clauset_table, headers=["Community", "Clauset's Parameter"], tablefmt="grid"))
