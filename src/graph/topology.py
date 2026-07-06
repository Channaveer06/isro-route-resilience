import networkx as nx
import numpy as np
from scipy.spatial import distance

def build_graph_from_skeleton(skeleton):
    """
    Extracts a graph (nodes as intersections/endpoints, edges as roads) from a skeleton.
    Simplified version for the hackathon using active pixel coordinates.
    """
    G = nx.Graph()
    y_coords, x_coords = np.where(skeleton > 0)
    
    # Add nodes
    for i in range(len(y_coords)):
        G.add_node(i, pos=(x_coords[i], y_coords[i]))
        
    # Connect adjacent pixels (8-connectivity)
    # This is a naive implementation; a true sknw library is preferred in production
    points = np.column_stack((x_coords, y_coords))
    if len(points) > 0:
        dist_matrix = distance.cdist(points, points, 'euclidean')
        # Connect nodes that are exactly 1 or sqrt(2) pixels apart
        for i in range(len(points)):
            neighbors = np.where((dist_matrix[i] > 0) & (dist_matrix[i] <= 1.5))[0]
            for n in neighbors:
                if not G.has_edge(i, n):
                    G.add_edge(i, n, weight=dist_matrix[i, n])
                    
    return G

class DisjointSet:
    def __init__(self, nodes):
        self.parent = {n: n for n in nodes}
        self.rank = {n: 0 for n in nodes}

    def find(self, i):
        if self.parent[i] == i:
            return i
        self.parent[i] = self.find(self.parent[i])
        return self.parent[i]

    def union(self, i, j):
        root_i = self.find(i)
        root_j = self.find(j)
        if root_i != root_j:
            if self.rank[root_i] < self.rank[root_j]:
                self.parent[root_i] = root_j
            elif self.rank[root_i] > self.rank[root_j]:
                self.parent[root_j] = root_i
            else:
                self.parent[root_j] = root_i
                self.rank[root_i] += 1

def heal_topology(G, max_distance=50.0):
    """
    Heals 'broken' roads (caused by occlusions like trees) by connecting
    disconnected components using Minimum Spanning Tree (Kruskal's approach).
    Crucially, it allows broken endpoints to snap to ANY node in a different
    component, seamlessly fixing broken T-junctions and side roads.
    """
    components = list(nx.connected_components(G))
    if len(components) <= 1:
        return G # Fully connected

    # Find endpoints (degree 1)
    endpoints = [n for n in G.nodes if G.degree(n) == 1]
    
    # If no endpoints, just use all nodes
    if not endpoints:
        endpoints = list(G.nodes)

    potential_edges = []
    
    # For every endpoint, find the closest node in EVERY other component
    for u in endpoints:
        pos_u = np.array(G.nodes[u]['pos'])
        
        for comp in components:
            if u in comp:
                continue
                
            # Find the closest node in this component to the endpoint 'u'
            min_dist = max_distance + 1
            best_v = None
            
            # Subsample large components for speed, but keep dense enough for accurate snapping
            sample_comp = list(comp)[::max(1, len(comp)//50)] 
            
            for v in sample_comp:
                pos_v = np.array(G.nodes[v]['pos'])
                dist = np.linalg.norm(pos_u - pos_v)
                if dist < min_dist:
                    min_dist = dist
                    best_v = v
                    
            if best_v is not None:
                # Cost is purely distance to allow clean perpendicular T-junction snaps
                potential_edges.append((min_dist, u, best_v, min_dist))

    # Sort edges by cost (Kruskal's)
    potential_edges.sort(key=lambda x: x[0])
    
    ds = DisjointSet(G.nodes)
    # Pre-union existing edges
    for u, v in G.edges:
        ds.union(u, v)

    # Apply MST logic
    for cost, u, v, dist in potential_edges:
        if ds.find(u) != ds.find(v):
            G.add_edge(u, v, weight=dist, healed=True)
            ds.union(u, v)
                
    return G
