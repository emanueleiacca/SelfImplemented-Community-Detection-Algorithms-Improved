import heapq
from collections import defaultdict, deque
import matplotlib.pyplot as plt
import numpy as np
import copy
import networkx as nx   

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
def Community_detection_algorithms(graph, betweenness_method="bfs", component_method="dfs_recursive"):
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
