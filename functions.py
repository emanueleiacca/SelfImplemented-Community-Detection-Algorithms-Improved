import heapq
from collections import defaultdict, deque
import matplotlib.pyplot as plt
import numpy as np
import copy
import networkx as nx   
from sklearn.preprocessing import normalize
import numpy as np

def find_connected_components(graph, method="dfs_recursive"):
    """
    Find all connected components in an undirected graph using DFS.

    Args:
        graph (dict): Adjacency list representing the graph.
        method (str): 'dfs_recursive' or 'dfs_iterative'.

    Returns:
        list: List of connected components.
    """
    visited = set()
    components = []

    # Recursive version
    # Source: https://www.geeksforgeeks.org/depth-first-search-or-dfs-for-a-graph/
    def dfs_recursive(node, component):
        """
        Recursive Depth-First Search (DFS) function to explore a connected component in the graph.
        
        This function visits all nodes reachable from the given 'node' and adds them to the same 
        connected component. It uses recursion to traverse all unvisited neighbors.
        """
        visited.add(node)  # Mark the current node as visited
        component.append(node)  # Add the current node to the component
        
        #print(f"Step {step_counter}:")
        #print(f"  Current Node: {node}")
        #print(f"  Visited Nodes: {visited}")
        #print(f"  Current Component: {component}")
        #print("----------------------------------")
        #step_counter += 1
    
        # Recursively visit all unvisited neighbors
        for neighbor in graph[node]:
            if neighbor not in visited:  # Process only unvisited nodes
                dfs_recursive(neighbor, component)

    # Iterative version
    def dfs_iterative(start):
        """
        Iterative Depth-First Search using an explicit stack.
        """
        stack = [start]
        component = []  # Local list to store the component nodes

        while stack:
            node = stack.pop()  # Pop the most recently added node (LIFO)
            if node not in visited:
                visited.add(node)
                component.append(node)
                
                # Add unvisited neighbors to the stack
                for neighbor in reversed(graph[node]):
                    if neighbor not in visited:
                        stack.append(neighbor)
        return component

    # Which one to use? Both give same results, O notation is the same but Recursive is more intuitive but may have some problem in larger dataset due to recursion limit that on Py is around 1000 calls

    # Explore all nodes
    for node in graph:
        if node not in visited:
            if method == "dfs_recursive":
                component = []
                dfs_recursive(node, component)
                components.append(component)
            elif method == "dfs_iterative":
                component = dfs_iterative(node)
                components.append(component)
            else:
                raise ValueError("Invalid method. Use 'dfs_recursive' or 'dfs_iterative'.")

    return components


# Source: https://www.geeksforgeeks.org/breadth-first-search-or-bfs-for-a-graph/
def bfs_shortest_paths(graph, source):
    """
    Perform Breadth-First Search (BFS) to calculate the shortest paths from a given source node to all other nodes.
    BFS explores vertices level by level, using a queue data structure. This implementation also tracks the parents 
    of each node to reconstruct shortest paths.
    """
    queue = deque([source]) # Initialize a queue for BFS, starting with the source node

    distances = {node: float('inf') for node in graph} # Set the distance to all nodes as infinity (unvisited)
    distances[source] = 0 # Set the distance to the source node as 0 (starting point).

    # Initialize the parents dictionary:
    # - Tracks all parents for each node along shortest paths.
    parents = defaultdict(list) # parents dict tracks all parents for each node along shortest paths
    
    #step = 0

    while queue: # until all vertices are reachable
        # Dequeue the current node
        current = queue.popleft() # remove from queue

        #print(f"Step {step}:")
        #print(f"  Current Node: {current}")
        #print(f"  Queue: {list(queue)}")
        #print(f"  Distances: {distances}")
        #print(f"  Parents: {dict(parents)}")

        # Explore all adjacent vertices of the current node
        for neighbor in graph[current]:
            if distances[neighbor] == float('inf'): # so not visited
                distances[neighbor] = distances[current] + 1 # update as one level deeper than current node
                queue.append(neighbor) # add to queue to process later

            if distances[neighbor] == distances[current] + 1: # if distance is exactly one level deeper than the current node
                parents[neighbor].append(current) # Add the current node as a parent of the neighbor

    return distances, parents

# Source: https://symbio6.nl/en/blog/analysis/betweenness-centrality (only for the concept)
def calculate_betweenness_bfs(graph):
    """
    Calculate edge betweenness centrality for all edges in an undirected network.

    Betweenness centrality measures the "bridgeness" of an edge by evaluating the fraction
    of shortest paths between all pairs of nodes in the network that pass through that edge.
    """

    betweenness = defaultdict(int) # Dict to store BC values for edges

    for node in graph:
        #print(f"\nProcessing Source Node: {node}")
        distances, parents = bfs_shortest_paths(graph, node) # BFS return shortest paths and their parent relationships
        #print(f"  BFS Distances from {node}: {distances}")
        #print(f"  BFS Parents from {node}: {dict(parents)}")
        node_flow = {n: 1 for n in graph} # Initialize flow for all nodes as 1 (default contribution to shortest paths)

        nodes_by_distance = sorted(distances, key=distances.get, reverse=True) # Process nodes with farthest nodes first
        #print(f"  Nodes by Distance (Reverse): {nodes_by_distance}")

        # Backtrack from farthest nodes to distribute flow across edges
        for target in nodes_by_distance: 
            for parent in parents[target]:
                # Define the edge between parent and target
                edge = tuple(sorted((parent, target)))  # Define edge between parent and target. Sort to avoid duplicates
                
                # Distribute flow proportionally across all shortest paths to the target
                flow = node_flow[target] / len(parents[target])  # Split flow equally
                betweenness[edge] += flow  # Add flow to the edge's betweenness
                node_flow[parent] += flow  # Pass flow back to the parent node
                #print(f"    Updated Betweenness for Edge {edge}: {betweenness[edge]}")
                #print(f"    Flow Distribution: Node {parent} Flow: {node_flow[parent]}")
                
    #for edge, score in betweenness.items():
        #print(f"  Edge {edge}: {score}")

    return betweenness

# Dijkstra's Algorithm 
def dijkstra_adj_list(graph, start_node):
    """
    Dijkstra's Algorithm to find the shortest paths from a start node
    in an adjacency list representation of a graph.
    """
    distances = {node: float('inf') for node in graph}
    distances[start_node] = 0
    previous_nodes = {node: None for node in graph}
    priority_queue = [(0, start_node)]

    while priority_queue:
        current_distance, current_node = heapq.heappop(priority_queue)
        for neighbor in graph[current_node]:
            weight = 1
            distance = current_distance + weight
            if distance < distances[neighbor]:
                distances[neighbor] = distance
                previous_nodes[neighbor] = current_node
                heapq.heappush(priority_queue, (distance, neighbor))

    return distances, previous_nodes

# Betweenness Centrality Calculation: Dijkstra based
def calculate_betweenness_dijkstra(graph):
    """
    Calculate edge betweenness centrality using Dijkstra's Algorithm.
    """
    betweenness = defaultdict(float)

    for node in graph:
        #print(f"\n[DEBUG] Running Dijkstra from source node: {node}")
        distances, parents = dijkstra_adj_list(graph, node)
        node_flow = {n: 1 for n in graph}

        # Sort nodes by distance from the source (farthest nodes processed first)
        nodes_by_distance = sorted(distances, key=distances.get, reverse=True)
        #print(f"[DEBUG] Nodes by distance (farthest to closest): {nodes_by_distance}")

        for target in nodes_by_distance:
            if parents[target] is not None:
                edge = tuple(sorted((parents[target], target)))  # Avoid duplicates
                flow = node_flow[target] / 1  # In unweighted graphs, flow = 1
                betweenness[edge] += flow
                node_flow[parents[target]] += flow
                #print(f"[DEBUG] Edge {edge} receives flow: {flow:.2f}")

 #   print("\n[DEBUG] Final Betweenness Centrality:")
  #  for edge, centrality in betweenness.items():
   #     print(f"  Edge {edge}: {centrality:.2f}")

    return betweenness
    
# Source: https://memgraph.github.io/networkx-guide/algorithms/community-detection/girvan-newman/
# Girvan-Newman Algorithm
def Girvan_Newman_methods(graph, betweenness_method="bfs", component_method="dfs_recursive"):
    """
    Algorithms for community detection.
    Args:
        graph (dict): Input graph as adjacency list.
        betweenness_method (str): 'bfs' or 'dijkstra' for edge betweenness.
        component_method (str): 'dfs_recursive' or 'dfs_iterative' for connected components.
    Returns:
        tuple: (communities, removed_edges)
    """
    graph_copy = copy.deepcopy({node: list(neighbors) for node, neighbors in graph.adjacency()})
    removed_edges = []
    iteration = 0

    while True:
        print(f"\n[DEBUG] Iteration {iteration}: Calculating betweenness centrality...")

        # Calculate betweenness centrality with choosed method
        if betweenness_method == "bfs":
            betweenness = calculate_betweenness_bfs(graph_copy)
        elif betweenness_method == "dijkstra":
            betweenness = calculate_betweenness_dijkstra(graph_copy)
        else:
            raise ValueError("Invalid betweenness method. Choose 'bfs' or 'dijkstra'.")

        # Stop if no edges remain
        if not betweenness:
            break

        # Remove the edge with the highest betweenness
        edge_to_remove = max(betweenness, key=betweenness.get)
        graph_copy[edge_to_remove[0]].remove(edge_to_remove[1])
        graph_copy[edge_to_remove[1]].remove(edge_to_remove[0])
        removed_edges.append(edge_to_remove)
        print(f"[DEBUG] Removed edge: {edge_to_remove}")

        # Find connected components
        communities = find_connected_components(graph_copy, method=component_method)
        print(f"[DEBUG] Communities: {communities}")

        # Terminate when more than one community is detected
        if len(communities) > 1:
            break

        iteration += 1

    return communities, removed_edges

# Visualization
def visualize_communities(graph, communities, removed_edges=[]):
    """
    Visualize the graph and its communities.
    """
    # Assign colors to nodes based on their community
    colors = plt.cm.rainbow(np.linspace(0, 1, len(communities)))
    color_map = {}
    for i, community in enumerate(communities):
        for node in community:
            color_map[node] = colors[i]

    pos = nx.spring_layout(graph)

    plt.figure(figsize=(10, 10))
    nx.draw_networkx_nodes(graph, pos, node_size=500, 
                           node_color=[color_map[node] for node in graph.nodes()],
                           edgecolors='black')
    nx.draw_networkx_labels(graph, pos, font_size=12, font_color='white')

    # Draw edges excluding removed edges
    edges_to_draw = [(u, v) for u, v in graph.edges() if (u, v) not in removed_edges and (v, u) not in removed_edges]
    nx.draw_networkx_edges(graph, pos, edgelist=edges_to_draw, width=2, alpha=0.5)

    plt.title("Girvan-Newman Communities")
    plt.axis("off")
    plt.show()

# Spectral Clustering

# KMeans and Kmeans++ for Section 2.3 for HW4
# Source: https://github.com/emanueleiacca/ADM-HW4/blob/main/functions/functions.py#L329
def initialize_centroids(data, k, method="random",seed=42):
    """
    Initialize centroids using the chosen method.
    Parameters:
        - data: NumPy array of data points.
        - k: Number of clusters.
        - method: "random" for basic initialization or "kmeans++" for K-me ()ans++ initialization.
    """
    if method == "random":
        np.random.seed(seed)  # Set the random seed for reproducibility
        # Randomly select k unique indices
        indices = np.random.choice(data.shape[0], k, replace=False)
        return data[indices]

    elif method == "kmeans++":
        np.random.seed(seed)
        # K-means++ initialization
        centroids = [data[np.random.choice(data.shape[0])]]  # First centroid randomly chosen
        for _ in range(1, k):
            # Compute distances from nearest centroid for all points
            distances = np.min([np.linalg.norm(data - centroid, axis=1) for centroid in centroids], axis=0)
            # Compute probabilities proportional to squared distances
            probabilities = distances ** 2 / np.sum(distances ** 2)
            # Choose next centroid based on probabilities
            next_centroid_index = np.random.choice(data.shape[0], p=probabilities)
            centroids.append(data[next_centroid_index])
        return np.array(centroids)

    else:
        raise ValueError("Invalid method. Choose 'random' or 'kmeans++'.")

def compute_distance(point, centroids):
    """Compute the distance of a point to all centroids and return the nearest one."""
    distances = np.linalg.norm(centroids - point, axis=1)
    return np.argmin(distances)  # Return the index of the closest centroid

def assign_clusters(data, centroids):
    """Assign each point to the nearest centroid."""
    clusters = []
    for point in data:
        cluster_id = compute_distance(point, centroids)
        clusters.append(cluster_id)
    return np.array(clusters)

def update_centroids(data, clusters, k):
    """Update centroids as the mean of points in each cluster."""
    new_centroids = []
    for cluster_id in range(k):
        cluster_points = data[clusters == cluster_id]
        if len(cluster_points) > 0:
            new_centroids.append(cluster_points.mean(axis=0))
        else:  # Handle empty cluster
            new_centroids.append(np.zeros(data.shape[1]))
    return np.array(new_centroids)

def kmeans(data, k, method="random", max_iterations=100, tolerance=1e-4, seed = 42):
    """
    K-means clustering algorithm with option for basic or K-means++ initialization.
    Parameters:
        - data: NumPy array of data points.
        - k: Number of clusters.
        - method: "random" for basic K-means or "kmeans++" for K-means++.
        - max_iterations: Maximum number of iterations.
        - tolerance: Convergence tolerance.
    """
    # Initialize centroids
    centroids = initialize_centroids(data, k, method=method)

    for iteration in range(max_iterations):
        # Assign clusters
        clusters = assign_clusters(data, centroids)

        # Update centroids
        new_centroids = update_centroids(data, clusters, k)

        # Check for convergence
        if np.all(np.abs(new_centroids - centroids) < tolerance):
            print(f"Converged at iteration {iteration}")
            break

        centroids = new_centroids

    return centroids, clusters


# Source: https://rahuljain788.medium.com/implementing-spectral-clustering-from-scratch-a-step-by-step-guide-9643e4836a76
def spectral_clustering(graph, k):
    """
    Spectral Clustering on a graph using normalized Laplacian and custom K-means.
    """
    # Adjacency and Degree Matrices
    adj_matrix, nodes = graph.adjacency_matrix()
    degrees = np.diag(adj_matrix.sum(axis=1))

    # Normalized Laplacian
    D_inv_sqrt = np.diag(1.0 / np.sqrt(np.diag(degrees)))
    D_inv_sqrt = np.nan_to_num(D_inv_sqrt)  # Handle division by zero
    L = np.eye(len(nodes)) - D_inv_sqrt @ adj_matrix @ D_inv_sqrt

    # Eigenvalue Decomposition
    eigenvalues, eigenvectors = np.linalg.eigh(L)
    sorted_indices = np.argsort(eigenvalues)  # Sort eigenvalues
    eigenvalues = eigenvalues[sorted_indices]
    eigenvectors = eigenvectors[:, sorted_indices]
    
    print("Sorted Eigenvalues:", eigenvalues[:k+1])

    # Select and normalize eigenvectors corresponding to smallest non-trivial eigenvalues
    k_smallest_eigenvectors = eigenvectors[:, 1:k+1]  # Skip the first trivial eigenvector
    k_smallest_eigenvectors = normalize(k_smallest_eigenvectors, axis=1)

    # Run K-Means Clustering
    centroids, cluster_assignments = kmeans(k_smallest_eigenvectors, k, method="kmeans++", seed=42)

    # Group Nodes into Communities
    communities = defaultdict(list)
    for i, cluster_id in enumerate(cluster_assignments):
        communities[cluster_id].append(nodes[i])

    return list(communities.values())

# Louvain Algorithm

# Source: https://users.ece.cmu.edu/~lowt/papers/Louvain_accepted.pdf

def louvain_cluster(adj_matrix, max_iter=10):
    """
    Simplified version of the Louvain clustering algorithm using NumPy. 
    The algorithm aims to detect communities in a graph by iteratively optimizing the modularity of the graph. 
    """
    n = adj_matrix.shape[0]  # Number of nodes
    degrees = np.sum(adj_matrix, axis=1)
    inv_m = 1.0 / np.sum(degrees)  # 2m for modularity normalization
    communities = np.arange(n)  # Initialize each node in its own community

    def modularity_gain(node, target_comm, curr_comm):
        """Compute the modularity gain of moving 'node' to 'target_comm'."""
        k_i = degrees[node]
        delta_q = 0.0
        # Direct Contributions
        for neighbor, weight in enumerate(adj_matrix[node]): # It iterates over all neighbors of the node
            if weight > 0:
                # Depending by where the neighbor belongs, add edge weight to the modularity gain
                if communities[neighbor] == target_comm:
                    delta_q += weight
                if communities[neighbor] == curr_comm:
                    delta_q -= weight
        # Indirect Contributions
        # Compute the sum of degrees of nodes in the target and current communities
        sum_in_target = np.sum(degrees[communities == target_comm])
        sum_in_curr = np.sum(degrees[communities == curr_comm])
        # Adjusted formula
        delta_q -= k_i * (sum_in_target - k_i) * inv_m
        delta_q += k_i * (sum_in_curr - k_i) * inv_m
        return delta_q

    # Iterative Community Refinement
    for iteration in range(max_iter): # Either max_iter or until no nodes are moved
        moved = False
        for node in range(n): # for each node
            curr_comm = communities[node]
            max_gain = 0
            best_comm = curr_comm
 
            # Evaluate modularity gain for moving it to each neighboring community
            for neighbor, weight in enumerate(adj_matrix[node]):
                if weight > 0 and communities[neighbor] != curr_comm:
                    target_comm = communities[neighbor]
                    gain = modularity_gain(node, target_comm, curr_comm)
                    if gain > max_gain:
                        max_gain = gain
                        best_comm = target_comm

            # Reassign the node to the best community
            if best_comm != curr_comm:
                communities[node] = best_comm
                moved = True

        if not moved:  # Stop if no nodes were moved
            break

    return extract_communities(communities)

def extract_communities(communities):
    """Group nodes by their community assignments."""
    community_groups = defaultdict(list)
    for node, comm in enumerate(communities):
        community_groups[comm].append(node)
    return list(community_groups.values())

# Additional Metrics

def lambiotte_coefficient(adj_matrix, communities):
    """
    Compute the Lambiotte coefficient for each node.
    """
    n = adj_matrix.shape[0]
    node_importance = {}

    for community in communities: # For each community
        for node in community: # For each node in the community
            k_in = np.sum([adj_matrix[node, neighbor] for neighbor in community]) # Internal degree
            k_total = np.sum(adj_matrix[node])  # Total degree
            L = k_in / k_total if k_total > 0 else 0 # Lambiotte formula # Avoid division by zero
            node_importance[node] = L
    
    return node_importance

def clauset_parameter(adj_matrix, communities):
    """
    Compute Clauset's parameter for each community.
    """
    community_quality = {}

    for idx, community in enumerate(communities): # For each community
        E_in = sum(adj_matrix[i, j] for i in community for j in community if i != j) / 2  # Internal edges
        E_out = sum(adj_matrix[i, j] for i in community for j in range(adj_matrix.shape[0]) if j not in community) # External edges
        
        Q = E_in / (E_in + E_out) if (E_in + E_out) > 0 else 0  # Clauset's formula # Avoid division by zero
        community_quality[idx] = Q

    return community_quality
