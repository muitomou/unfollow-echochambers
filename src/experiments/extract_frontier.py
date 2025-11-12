import os, pandas as pd, numpy as np

IN_CSV  = "experiments/results_sweep_refined/sweep_refined_mean_std.csv"
OUT_CSV = "experiments/results_sweep_refined/frontier_curve.csv"

df = pd.read_csv(IN_CSV)

frontier = []
for eps, group in df.groupby("epsilon"):
    group = group.sort_values("p_cut")
    # criterio: cae la fracción del componente mayor
    thr = 0.95
    crossed = group[group["frac_largest_mean"] < thr]
    if not crossed.empty:
        row = crossed.iloc[0]
        frontier.append({"epsilon": eps, "p_cut_star": row["p_cut"], "frac_largest_mean": row["frac_largest_mean"]})
    else:
        frontier.append({"epsilon": eps, "p_cut_star": np.nan, "frac_largest_mean": group["frac_largest_mean"].min()})

out = pd.DataFrame(frontier)
os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
out.to_csv(OUT_CSV, index=False)
print(f"✅ Frontera guardada en {OUT_CSV}")
