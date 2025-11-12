import numpy as np
import networkx as nx
from typing import Dict, Tuple

class DeffuantCoevo:
    def __init__(self, G: nx.Graph, epsilon: float, mu: float,
                 p_cut: float, friend_of_friend: bool = True,
                 max_attempts: int = 5, seed: int = 0):
        self.G = G
        self.rng = np.random.default_rng(seed)
        self.epsilon = epsilon
        self.mu = mu
        self.p_cut = p_cut
        self.friend_of_friend = friend_of_friend
        self.max_attempts = max_attempts
        self.x = np.array(self.rng.random(self.G.number_of_nodes()))
        self.node_to_idx = {n: i for i, n in enumerate(self.G.nodes())}

    def step(self):
        if self.G.number_of_edges() == 0:
            return False
        u, v = list(self.G.edges())[self.rng.integers(self.G.number_of_edges())]
        iu, iv = self.node_to_idx[u], self.node_to_idx[v]
        diff = abs(self.x[iu] - self.x[iv])

        if diff <= self.epsilon:
            # Interacción normal (Deffuant)
            xu, xv = self.x[iu], self.x[iv]
            self.x[iu] = xu + self.mu * (xv - xu)
            self.x[iv] = xv + self.mu * (xu - xv)
            return True
        else:
            # Rechazo -> posible unfollow
            if self.rng.random() < self.p_cut:
                if self.G.has_edge(u, v):
                    self.G.remove_edge(u, v)
                if self.friend_of_friend:
                    self._rewire_homophilic(u)
            return False

    def _rewire_homophilic(self, u):
        """Intentar conectar con un amigo de un amigo similar."""
        # vecinos de vecinos
        neighbors2 = set()
        for n in self.G.neighbors(u):
            neighbors2.update(self.G.neighbors(n))
        neighbors2.discard(u)
        # filtrar los que ya son amigos
        existing = set(self.G.neighbors(u))
        candidates = [n for n in neighbors2 if n not in existing]

        if not candidates:
            return
        self.rng.shuffle(candidates)
        xu = self.x[self.node_to_idx[u]]

        for w in candidates[:self.max_attempts]:
            xw = self.x[self.node_to_idx[w]]
            if abs(xu - xw) <= self.epsilon:
                self.G.add_edge(u, w)
                return  # sólo una conexión nueva

    def opinions(self):
        return self.x.copy()
