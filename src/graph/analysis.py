import networkx as nx

def calculate_betweenness_centrality(G):
    """
    Calculates betweenness centrality for all nodes in the graph to find bottlenecks.
    High centrality indicates critical routes.
    """
    # For large graphs, k (number of sample nodes) can be reduced for speed.
    centrality = nx.betweenness_centrality(G, weight='weight', normalized=True)
    nx.set_node_attributes(G, centrality, 'centrality')
    return G

def find_critical_nodes(G, top_k=10):
    """
    Returns the top_k nodes with the highest betweenness centrality.
    """
    centrality_dict = nx.get_node_attributes(G, 'centrality')
    sorted_nodes = sorted(centrality_dict.items(), key=lambda x: x[1], reverse=True)
    return sorted_nodes[:top_k]
