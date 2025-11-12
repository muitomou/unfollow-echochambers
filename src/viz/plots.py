import os
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

def plot_opinion_hist(opinions, outpath, title="Opinions histogram"):
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    plt.figure(figsize=(6,4))
    plt.hist(opinions, bins=20, range=(0,1))
    plt.title(title)
    plt.xlabel("Opinion")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()

def plot_graph_snapshot(G: nx.Graph, opinions, outpath, title="Graph snapshot"):
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    plt.figure(figsize=(6,6))
    pos = nx.spring_layout(G, seed=123)  # layout reproducible
    # color por opinión
    nx.draw_networkx_nodes(G, pos, node_color=opinions, vmin=0, vmax=1, node_size=40)
    nx.draw_networkx_edges(G, pos, alpha=0.2, width=0.5)
    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()
