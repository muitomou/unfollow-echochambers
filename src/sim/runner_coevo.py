import os, argparse, yaml, pandas as pd
from tqdm import trange
from src.graphs.generators import make_graph
from src.models.deffuant_coevo import DeffuantCoevo
from src.viz.plots import plot_opinion_hist, plot_graph_snapshot
from src.viz.network_metrics import structural_metrics

def main(config_path):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    os.makedirs(cfg["io"]["results_dir"], exist_ok=True)
    os.makedirs(cfg["io"]["figs_dir"], exist_ok=True)

    G = make_graph(cfg)
    model = DeffuantCoevo(
        G=G,
        epsilon=cfg["deffuant"]["epsilon"],
        mu=cfg["deffuant"]["mu"],
        p_cut=cfg["coevo"]["p_cut"],
        friend_of_friend=cfg["coevo"]["friend_of_friend"],
        max_attempts=cfg["coevo"]["max_attempts"],
        seed=cfg["seed"],
    )

    rows = []
    for t in trange(cfg["sim"]["n_steps"], desc="Co-evolving"):
        model.step()
        if t % cfg["sim"]["snapshot_every"] == 0 or t == cfg["sim"]["n_steps"]-1:
            opinions = model.opinions()
            s_metrics = structural_metrics(G, opinions)
            s_metrics.update({
                "t": t,
                "mean_opinion": opinions.mean(),
                "std_opinion": opinions.std(),
            })
            rows.append(s_metrics)
            plot_opinion_hist(opinions,
                f"{cfg['io']['figs_dir']}/hist_t{t}.png",
                title=f"Opinions @ t={t}")
            plot_graph_snapshot(G, opinions,
                f"{cfg['io']['figs_dir']}/snapshot_t{t}.png",
                title=f"Graph @ t={t}")

    pd.DataFrame(rows).to_csv(
        f"{cfg['io']['results_dir']}/metrics_coevo.csv", index=False)
    print("Done. Metrics saved.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    main(args.config)
