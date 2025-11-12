import networkx as nx
import numpy as np

def largest_component(G: nx.Graph):
    if G.number_of_nodes() == 0:
        return G
    comps = list(nx.connected_components(G))
    if not comps:
        return G
    largest = max(comps, key=len)
    return G.subgraph(largest).copy()

def avg_degree(G: nx.Graph) -> float:
    n = G.number_of_nodes()
    m = G.number_of_edges()
    return 0.0 if n == 0 else (2.0 * m) / n

def avg_clustering(G: nx.Graph) -> float:
    if G.number_of_nodes() == 0:
        return 0.0
    return nx.average_clustering(G)

def avg_path_length_LCC(G: nx.Graph) -> float:
    if G.number_of_nodes() == 0:
        return 0.0
    LCC = largest_component(G)
    if LCC.number_of_nodes() <= 1:
        return 0.0
    try:
        return nx.average_shortest_path_length(LCC)
    except Exception:
        return float("nan")

def modularity_Q(G: nx.Graph, opinions=None):
    """
    Estima Q con comunidades encontradas por greedy modularity.
    Opcionalmente puedes ponderar una versión 'ideológica' usando opiniones,
    pero aquí usamos sólo estructura (estándar).
    """
    if G.number_of_nodes() == 0:
        return 0.0
    try:
        comms = nx.algorithms.community.greedy_modularity_communities(G)
        return nx.algorithms.community.modularity(G, comms)
    except Exception:
        return float("nan")

def structural_metrics_full(G: nx.Graph, opinions=None):
    comps = list(nx.connected_components(G))
    n_comp = len(comps) if comps else 0
    largest = max((len(c) for c in comps), default=0)
    n = G.number_of_nodes()
    frac_largest = (largest / n) if n > 0 else 0.0

    # homofilia como correlación (si hay opiniones)
    homophily = float("nan")
    if opinions is not None:
        op_map = dict(enumerate(opinions))
        avg_neigh = []
        for i in range(len(opinions)):
            neighs = list(G.neighbors(i))
            if neighs:
                avg_neigh.append(np.mean([op_map[nn] for nn in neighs]))
            else:
                avg_neigh.append(opinions[i])
        try:
            homophily = float(np.corrcoef(opinions, np.array(avg_neigh))[0,1])
        except Exception:
            homophily = float("nan")

    return {
        "n_components": n_comp,
        "frac_largest": frac_largest,
        "avg_degree": avg_degree(G),
        "clustering": avg_clustering(G),
        "path_length_LCC": avg_path_length_LCC(G),
        "modularity_Q": modularity_Q(G),
        "homophily": homophily,
    }
