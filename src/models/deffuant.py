import numpy as np
import networkx as nx
from typing import Tuple, Dict

class DeffuantModel:
    """
    Opiniones en [0,1]. En cada step:
      - elige arista (u,v) al azar
      - si |x_u - x_v| <= epsilon, ambos se mueven hacia el otro: 
            x_u <- x_u + mu*(x_v - x_u)
            x_v <- x_v + mu*(x_u_old - x_v)
      - si > epsilon, no hay cambio (hoy sin 'unfollow')
    """
    def __init__(self, G: nx.Graph, epsilon: float, mu: float, seed: int = 0):
        self.G = G
        self.rng = np.random.default_rng(seed)
        self.epsilon = float(epsilon)
        self.mu = float(mu)
        # opiniones iniciales ~ U(0,1)
        self.x = np.array(self.rng.random(self.G.number_of_nodes()), dtype=float)
        # mapea node -> idx
        self.node_to_idx = {n: i for i, n in enumerate(self.G.nodes())}

    def step(self) -> bool:
        # Escoge una arista al azar. Si el grafo no tiene aristas, nada que hacer
        if self.G.number_of_edges() == 0:
            return False
        u, v = list(self.G.edges())[self.rng.integers(self.G.number_of_edges())]
        iu, iv = self.node_to_idx[u], self.node_to_idx[v]
        duv = abs(self.x[iu] - self.x[iv])
        if duv <= self.epsilon:
            xu, xv = self.x[iu], self.x[iv]
            self.x[iu] = xu + self.mu * (xv - xu)
            self.x[iv] = xv + self.mu * (xu - xv)
            return True
        return False

    def opinions(self) -> np.ndarray:
        return self.x.copy()

    def summary_metrics(self) -> Dict[str, float]:
        # métricas simples del estado
        x = self.x
        return {
            "mean": float(np.mean(x)),
            "std": float(np.std(x)),
            "entropy_bins_20": float(entropy_hist(x, bins=20)),
        }

def entropy_hist(vals: np.ndarray, bins: int = 20) -> float:
    hist, _ = np.histogram(vals, bins=bins, range=(0.0, 1.0), density=True)
    # evitar log(0): agrega epsilon pequeño
    p = hist / (hist.sum() + 1e-12)
    p = np.where(p <= 0, 1e-12, p)
    return float(-(p * np.log(p)).sum())
