import networkx as nx
import numpy as np

def structural_metrics(G: nx.Graph, opinions):
    comps = list(nx.connected_components(G))
    n_comp = len(comps)
    largest = max(len(c) for c in comps) if comps else 0
    frac_largest = largest / G.number_of_nodes() if G.number_of_nodes() > 0 else 0
    # homofilia: correlación entre opinión y grado medio de sus vecinos
    opinion_dict = dict(enumerate(opinions))
    avg_neighbor_op = []
    for i in range(len(opinions)):
        neighs = list(G.neighbors(i))
        if neighs:
            avg_neighbor_op.append(np.mean([opinion_dict[n] for n in neighs]))
        else:
            avg_neighbor_op.append(opinions[i])
    homophily = float(np.corrcoef(opinions, avg_neighbor_op)[0,1])
    return {
        "n_components": n_comp,
        "frac_largest": frac_largest,
        "homophily": homophily,
    }
