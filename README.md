### 1. **Performance Optimization**
- Custom implementation shows slight performance improvement with specific graph structures.
- Efficient handling of recursion limits in DFS with iterative fallback.

### 2. **Flexibility**
- Supports both BFS and Dijkstra for betweenness centrality calculations.
- Allows switching between recursive and iterative DFS for connected components.

### 3. **Debugging and Traceability**
- Clear debug logs for each iteration of edge removal.
- Detailed community snapshots at each stage.

### 4. **Modularity**
- Separation of concerns in the source code (`bfs`, `dfs`, `dijkstra`, etc.).
- Easier to maintain and extend.

### Comparison Results
- ✅ Communities match with NetworkX.
- ⏱️ Slight time improvement observed in iterative BFS/DFS approach.
- 🔍 Enhanced modularity for community analysis.

### 