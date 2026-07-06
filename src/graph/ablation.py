import networkx as nx

def simulate_node_ablation(G, node_to_remove):
    """
    Simulates the failure of a critical node (e.g., severe flooding or occlusion).
    Returns a new graph with the node removed.
    """
    G_ablated = G.copy()
    if G_ablated.has_node(node_to_remove):
        G_ablated.remove_node(node_to_remove)
    return G_ablated

def calculate_resilience_index(G_original, G_ablated):
    """
    Calculates the Resilience Index strictly defined as the ratio of the 
    average shortest path length in the baseline network to that in the perturbed network.
    A lower R indicates a highly vulnerable network.
    """
    if len(G_original) == 0 or len(G_ablated) == 0:
        return 0.0

    # Get largest connected components to calculate meaningful shortest paths
    lcc_orig = max(nx.connected_components(G_original), key=len)
    lcc_abl = max(nx.connected_components(G_ablated), key=len)
    
    G_orig_sub = G_original.subgraph(lcc_orig)
    G_abl_sub = G_ablated.subgraph(lcc_abl)

    # To save time on large networks in the dashboard, we sample nodes
    num_samples = min(50, len(G_orig_sub))
    if num_samples < 2:
        return 0.0
        
    import random
    random.seed(42) # Deterministic for UI
    
    nodes_orig = list(G_orig_sub.nodes)
    nodes_abl = list(G_abl_sub.nodes)
    
    # Calculate approx average shortest path for baseline
    path_lengths_orig = []
    for _ in range(num_samples):
        u, v = random.sample(nodes_orig, 2)
        try:
            path_lengths_orig.append(nx.shortest_path_length(G_orig_sub, source=u, target=v, weight='weight'))
        except nx.NetworkXNoPath:
            pass
            
    avg_path_orig = sum(path_lengths_orig) / len(path_lengths_orig) if path_lengths_orig else float('inf')
    
    # Calculate approx average shortest path for ablated
    path_lengths_abl = []
    for _ in range(num_samples):
        u, v = random.sample(nodes_abl, 2)
        try:
            path_lengths_abl.append(nx.shortest_path_length(G_abl_sub, source=u, target=v, weight='weight'))
        except nx.NetworkXNoPath:
            pass
            
    avg_path_abl = sum(path_lengths_abl) / len(path_lengths_abl) if path_lengths_abl else float('inf')

    # If ablation completely destroys paths
    if avg_path_abl == float('inf') or avg_path_abl == 0:
        return 0.0
        
    # Apply a catastrophic fragmentation penalty if the network breaks into smaller pieces
    # This prevents the "tiny component = short path = >100% survival" mathematical illusion
    fragmentation_penalty = len(G_orig_sub) / len(G_abl_sub) if len(G_abl_sub) > 0 else float('inf')
    adjusted_avg_path_abl = avg_path_abl * fragmentation_penalty
    
    resilience_index = avg_path_orig / adjusted_avg_path_abl
    
    # Cap at 1.0 (100%) in case of minor sampling variance
    return min(resilience_index, 1.0)
