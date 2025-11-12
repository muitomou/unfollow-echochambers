import networkx as nx
from typing import Dict

def make_graph(cfg: Dict) -> nx.Graph:
    gtype = cfg["graph"]["type"]
    n = cfg["n_nodes"]
    if gtype == "er":
        # usa k aprox como grado medio: p = k / (n-1)
        k = cfg["graph"].get("k", 6)
        p = k / max(n-1, 1)
        G = nx.erdos_renyi_graph(n, p, seed=cfg["seed"])
    elif gtype == "ws":
        k = cfg["graph"]["k"]
        p = cfg["graph"]["p"]
        G = nx.watts_strogatz_graph(n, k, p, seed=cfg["seed"])
    elif gtype == "ba":
        m = cfg["graph"].get("m", 3)
        G = nx.barabasi_albert_graph(n, m, seed=cfg["seed"])
    else:
        raise ValueError(f"unknown graph type {gtype}")
    return G
