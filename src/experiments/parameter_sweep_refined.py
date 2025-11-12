import os, yaml, pandas as pd, numpy as np
from tqdm import tqdm
from src.graphs.generators import make_graph
from src.models.deffuant_coevo import DeffuantCoevo
from src.metrics.network_stats import structural_metrics_full

BASE_CFG_PATH = "experiments/configs/coevo.yaml"
OUT_DIR = "experiments/results_sweep_refined"
os.makedirs(OUT_DIR, exist_ok=True)

with open(BASE_CFG_PATH) as f:
    base_cfg = yaml.safe_load(f)

# --- Rejilla  ---
epsilons = np.linspace(0.10, 0.40, 7)      # 0.10, 0.15, ..., 0.40
p_cuts    = np.linspace(0.01, 0.30, 10)     # 0.01, 0.04, ..., 0.30
seeds     = [41, 42, 43, 44, 45]            # 5 seeds (robustez)

N_STEPS = 10000
N_NODES = 400

rows = []

for eps in tqdm(epsilons, desc="epsilon"):
    for pcut in p_cuts:
        metrics_list = []
        for sd in seeds:
            cfg = base_cfg.copy()
            cfg["deffuant"]["epsilon"] = float(eps)
            cfg["coevo"]["p_cut"]      = float(pcut)
            cfg["sim"]["n_steps"]      = int(N_STEPS)
            cfg["n_nodes"]             = int(N_NODES)
            cfg["seed"]                = int(sd)

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

            for _ in range(cfg["sim"]["n_steps"]):
                model.step()

            opinions = model.opinions()
            met = structural_metrics_full(G, opinions)
            # guarda también std_opinion final por seed
            met.update({
                "std_opinion": float(np.std(opinions)),
                "epsilon": float(eps),
                "p_cut": float(pcut),
                "seed": int(sd),
            })
            metrics_list.append(met)

        df = pd.DataFrame(metrics_list)

        # agrega promedios y desvíos estándar por celda (ε, p_cut)
        agg = df.drop(columns=["seed"]).agg(['mean','std'])
        flat = {}
        for col in agg.columns:
            flat[f"{col}_mean"] = agg.loc['mean', col]
            flat[f"{col}_std"]  = agg.loc['std',  col]

        flat.update({"epsilon": float(eps), "p_cut": float(pcut), "n_seeds": len(seeds)})
        rows.append(flat)

df_out = pd.DataFrame(rows)
csv_path = os.path.join(OUT_DIR, "sweep_refined_mean_std.csv")
df_out.to_csv(csv_path, index=False)
print(f"✅ Barrido refinado guardado en {csv_path}")
