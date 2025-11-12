import os
import yaml
import pandas as pd
import numpy as np
from tqdm import tqdm
import networkx as nx

from src.graphs.generators import make_graph
from src.models.deffuant_coevo import DeffuantCoevo
from src.viz.network_metrics import structural_metrics

# --- Configuración base ---
BASE_CFG_PATH = "experiments/configs/coevo.yaml"
OUTPUT_DIR = "experiments/results_sweep"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- Cargar configuración base ---
with open(BASE_CFG_PATH) as f:
    base_cfg = yaml.safe_load(f)

# --- Rango de parámetros a explorar ---
epsilons = [0.1, 0.15, 0.2, 0.25, 0.3]
p_cuts = [0.01, 0.05, 0.1, 0.15, 0.2]

# --- Resultados ---
rows = []

for eps in tqdm(epsilons, desc="Explorando epsilon"):
    for pcut in p_cuts:
        cfg = base_cfg.copy()
        cfg["deffuant"]["epsilon"] = eps
        cfg["coevo"]["p_cut"] = pcut
        cfg["sim"]["n_steps"] = 8000
        cfg["n_nodes"] = 300

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

        # Simulación
        for _ in range(cfg["sim"]["n_steps"]):
            model.step()

        opinions = model.opinions()
        s_metrics = structural_metrics(G, opinions)
        s_metrics.update({
            "epsilon": eps,
            "p_cut": pcut,
            "mean_opinion": opinions.mean(),
            "std_opinion": opinions.std(),
        })
        rows.append(s_metrics)

# --- Guardar resultados ---
df = pd.DataFrame(rows)
df.to_csv(os.path.join(OUTPUT_DIR, "sweep_results.csv"), index=False)
print(f"----Barrido completado. Guardado en {OUTPUT_DIR}/sweep_results.csv----")
