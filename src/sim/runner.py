import os
import argparse
import yaml
import pandas as pd
from tqdm import trange

from src.graphs.generators import make_graph
from src.models.deffuant import DeffuantModel
from src.viz.plots import plot_opinion_hist, plot_graph_snapshot

def main(config_path: str):
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    results_dir = cfg["io"]["results_dir"]
    figs_dir = cfg["io"]["figs_dir"]
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(figs_dir, exist_ok=True)

    # Grafo
    G = make_graph(cfg)

    # Modelo
    model = DeffuantModel(
        G=G,
        epsilon=cfg["deffuant"]["epsilon"],
        mu=cfg["deffuant"]["mu"],
        seed=cfg["seed"],
    )

    # Simulación
    rows = []
    n_steps = cfg["sim"]["n_steps"]
    snap_every = cfg["sim"]["snapshot_every"]

    for t in trange(n_steps, desc="Simulating"):
        changed = model.step()
        if (t % snap_every == 0) or (t == n_steps - 1):
            opinions = model.opinions()
            metrics = model.summary_metrics()
            metrics.update({"t": t})
            rows.append(metrics)

            # Figuras
            hist_out = os.path.join(figs_dir, f"hist_t{t}.png")
            plot_opinion_hist(opinions, hist_out, title=f"Opinions @ t={t}")

            snap_out = os.path.join(figs_dir, f"snapshot_t{t}.png")
            plot_graph_snapshot(G, opinions, snap_out, title=f"Graph @ t={t}")

    # Guardar CSV
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(results_dir, "metrics.csv"), index=False)
    print(f"Saved metrics to {os.path.join(results_dir, 'metrics.csv')}")
    print(f"Saved figures to {figs_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    main(args.config)
