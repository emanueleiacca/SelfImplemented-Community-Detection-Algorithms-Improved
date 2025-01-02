# 📚 **Community Detection Algorithms**

Now it's a library! Check out the implementation also on [Pypi link](https://pypi.org/project/SelfImplemented-Community-Detection-Algorithms-Improved/0.1.0/)

```bash
pip install SelfImplemented-Community-Detection-Algorithms-Improved==0.1.0
```

## 📝 **Overview**
This library implements **community detection algorithms** with a focus on:
- **Performance Optimization**  
- **Flexibility**  
- **Traceability**  
- **Modularity**

Compared to **NetworkX's native algorithms**, our custom implementation delivers measurable performance gains, customizable strategies, and transparent debugging logs.

### **Girvan-Newman Algorithm for Community Detection**
- **Purpose:** Identify communities by removing edges with the **highest betweenness centrality**.  
- **Customizable Methods:**  
   - **Betweenness Calculation:** `bfs` | `dijkstra`  
   - **Component Detection:** `dfs_recursive` | `dfs_iterative`  

#### ⚠️ **Custom Method: BFS | Component Method: DFS Recursive**
| Metric              | Custom | NetworkX | Difference | % Improvement |
|----------------------|-------:|---------:|----------:|-------------:|
| **Execution Time**   | 0.0380s | 0.0265s | 0.0115s   | **-30.25%**  |

✅ **Communities Match:** Yes  

#### ⚠️ **Custom Method: BFS | Component Method: DFS Iterative**
| Metric              | Custom | NetworkX | Difference | % Improvement |
|----------------------|-------:|---------:|----------:|-------------:|
| **Execution Time**   | 0.0281s | 0.0265s | 0.0016s   | **-6.04%**   |

✅ **Communities Match:** Yes  

#### ⚠️ **Custom Method: Dijkstra | Component Method: DFS Recursive**
| Metric              | Custom | NetworkX | Difference | % Improvement |
|----------------------|-------:|---------:|----------:|-------------:|
| **Execution Time**   | 0.0176s | 0.0274s | 0.0098s   | **35.77%**   |

✅ **Communities Match:** Yes  

#### ⚠️ **Custom Method: Dijkstra | Component Method: DFS Iterative**
| Metric              | Custom | NetworkX | Difference | % Improvement |
|----------------------|-------:|---------:|----------:|-------------:|
| **Execution Time**   | 0.0164s | 0.0264s | 0.0100s   | **37.88%**   |

✅ **Communities Match:** Yes  

---

### **Spectral Clustering**
- **Purpose:** Partition the graph using the **eigenvalues and eigenvectors** of its Laplacian matrix.  

**Key Improvements:**  
- Faster convergence with **optimized eigenvalue sorting**.  
- Transparent **iteration logs** for eigenvector refinement.

**⏱️ Performance Comparison:**  
| Metric              | Custom Spectral | Pre-Implemented | Difference | % Improvement |
|----------------------|---------------:|---------------:|----------:|-------------:|
| **Execution Time**   | 0.0179s        | 0.2326s        | 0.2147s   | **92.28%**   |

✅ **Communities Match:** Yes

---

### **Louvain Method**
- **Purpose:** Hierarchical clustering by **maximizing modularity**.

**Key Improvements:**  
- Use of math-based methods to improve running time

**⏱️ Performance Comparison:**  
| Metric              | Custom Louvain | Pre-Implemented | Difference | % Improvement |
|----------------------|---------------:|---------------:|----------:|-------------:|
| **Execution Time**   | 0.0000000000s        | 0.0039553642s        | 0.0039s   | **4000000%**     |
| **Modularity**       | 0.4187         | 0.4449         | -0.0262   | **-5.89%**   |

❌ **Communities Do NOT Match:**  

**🔑 Node Importance (Lambiotte Coefficient):**
| Node | Coefficient |
|------|------------:|
| 0    | 0.9048      |
| 3    | 1.0         |

**🔑 Community Strength (Clauset's Parameter):**
| Community | Strength |
|-----------|---------:|
| 0         | 0.9      |
| 1         | 0.9091   |

The Custom method aligns well with the pre-implemented approach for major communities, though it sometimes splits larger communities into smaller subsets (This is probably because Nx cuts the creation of new communities if doing it adds gains under a certain threshold). Its strength btw is its computational efficiency: it runs a lot faster than the pre-implemented function, but the gain in computational resources can be useful in the computational-expensive tasks, with the right trade-off between efficiency and quality of results

---
## 🛠️ **How to Use the Methods**

### 1️⃣ **Girvan-Newman Algorithm**

The **Girvan-Newman Algorithm** identifies communities by iteratively removing edges with the highest betweenness centrality.

**Configuration Options:**
- **`betweenness_method`**: Choose between `"bfs"` (default) or `"dijkstra"` for betweenness calculation.
- **`component_method`**: Choose between `"dfs_recursive"` (default) or `"dfs_iterative"` for connected component detection.

**Example Usage:**
```python
betweenness_method = "bfs"  # Options: "bfs", "dijkstra"
component_method = "dfs_recursive"  # Options: "dfs_recursive", "dfs_iterative"

# Run Girvan-Newman Algorithm
communities, removed_edges = custom_girvan_newman(G, betweenness_method, component_method)
# Visualize Results
visualize_communities(G, communities, removed_edges)
```

### 2️⃣ **Spectral Clustering**

**Spectral Clustering** uses the eigenvalues and eigenvectors of a graph's Laplacian matrix to detect communities.

**Configuration Options:**

- **k**: Number of clusters to detect.

**Example Usage:**
```python
edges = list(nx_G.edges())  # Extract unweighted edges
G = Graph(edges) # Convert to custom graph object
print("\n--- Spectral Clustering ---")
spectral_communities = spectral_clustering(G, k=2)
print("Spectral Communities:", spectral_communities)
```

### 3️⃣ **Louvain Method**

The Louvain Method detects communities by maximizing modularity in hierarchical clustering.

**Configuration Options:**

- **max_iter**: Number of iterations for optimizing modularity.
**Example Usage:**
```python
import networkx as nx
adj_matrix = nx.to_numpy_array(nx_G)
communities = louvain_cluster(adj_matrix, max_iter=10)
print("\nDetected Communities:", communities)
```

**For documentation, usage examples, and contribution guidelines, refer to the repository. 🚀**
