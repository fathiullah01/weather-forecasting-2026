# =============================================================
# STEP 2: PSO FEATURE SELECTION
# Uses pyswarms GlobalBestPSO to select best input features
# =============================================================

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os, sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import *
from pyswarms.single.global_best import GlobalBestPSO
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler

print("=" * 60)
print("  STEP 2: PSO Feature Selection")
print("=" * 60)

# -- Load data --------------------------------------------------
df          = pd.read_csv(os.path.join(RESULT_DIR,"preprocessed_data.csv"))
all_features= pd.read_csv(os.path.join(RESULT_DIR,"all_features.csv"),
                           header=None)[0].tolist()
state_code  = open(os.path.join(RESULT_DIR,"state_code.txt")).read().strip()

X = df[all_features].values
y = df["target_precipitation"].values
X_scaled = StandardScaler().fit_transform(X)
n_features = X_scaled.shape[1]

print(f"\n[INFO] Candidate features : {n_features}")
print(f"[INFO] Samples            : {len(y)}")
print(f"[PSO]  Particles          : {PSO_N_PARTICLES_FS}")
print(f"[PSO]  Iterations         : {PSO_N_ITERATIONS_FS}")

# -- PSO objective ----------------------------------------------
def pso_objective(particle_matrix, X, y):
    n   = particle_matrix.shape[0]
    fit = np.zeros(n)
    for i, p in enumerate(particle_matrix):
        sel = p > 0.5
        if sel.sum() == 0:
            fit[i] = 1.0; continue
        rf  = RandomForestRegressor(n_estimators=50, max_depth=6,
                                    random_state=42, n_jobs=-1)
        fit[i] = 1.0 - np.mean(cross_val_score(rf, X[:,sel], y,
                                                cv=3, scoring="r2"))
    return fit

try:
    opt = GlobalBestPSO(
        n_particles = PSO_N_PARTICLES_FS,
        dimensions  = n_features,
        options     = {"c1": PSO_C1, "c2": PSO_C2, "w": PSO_W},
        bounds      = (np.zeros(n_features), np.ones(n_features)))
    best_cost, best_pos = opt.optimize(pso_objective, iters=PSO_N_ITERATIONS_FS,
                                       X=X_scaled, y=y)
    cost_history = opt.cost_history
    pso_ok = True
except Exception as e:
    print(f"[PSO] Warning: {e} - using all features")
    best_pos     = np.ones(n_features) * 0.6
    cost_history = [1.0]
    pso_ok       = False

sel_mask   = best_pos > 0.5
sel_names  = [all_features[i] for i in range(n_features) if sel_mask[i]]
drop_names = [all_features[i] for i in range(n_features) if not sel_mask[i]]

print(f"\n[PSO] Features selected : {sel_mask.sum()} / {n_features}")
print("  Selected:")
for f in sel_names:
    print(f"    [OK] {f.replace(state_code+'_','').replace('_lag1',' (lag-1)')}")
print("  Dropped:")
for f in drop_names:
    print(f"    [--] {f.replace(state_code+'_','').replace('_lag1',' (lag-1)')}")

# -- Table 5 - PSO feature weights -----------------------------
short_names = [f.replace(f"{state_code}_","").replace("_lag1"," (lag1)")
               for f in all_features]
feat_df = pd.DataFrame({
    "Feature"   : all_features,
    "ShortName" : short_names,
    "PSO_Weight": best_pos.round(4),
    "Selected"  : sel_mask.astype(int),
}).sort_values("PSO_Weight", ascending=False)
feat_df.to_csv(os.path.join(TABLE_DIR,"table5_pso_feature_selection.csv"), index=False)
print("\n[TABLE] table5_pso_feature_selection.csv saved")

# -- Figure 6 - PSO convergence --------------------------------
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(range(1, len(cost_history)+1), cost_history,
        color="#E91E63", linewidth=2, marker="o", markersize=3)
ax.fill_between(range(1, len(cost_history)+1), cost_history,
                alpha=0.15, color="#E91E63")
ax.set_xlabel("Iteration", fontsize=11)
ax.set_ylabel("Best Cost  (1 - R2)", fontsize=11)
ax.set_title("Figure 6 - PSO Feature Selection Convergence Curve",
             fontsize=13, fontweight="bold")
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR,"fig6_pso_convergence.png"),
            dpi=FIG_DPI, bbox_inches="tight")
plt.close()
print("[FIGURE] fig6_pso_convergence.png saved")

# -- Figure 7 - Feature weights bar ----------------------------
colors = ["#4CAF50" if s else "#F44336" for s in sel_mask]
fig, ax = plt.subplots(figsize=(14, 5))
ax.bar(range(n_features), best_pos, color=colors, edgecolor="white", alpha=0.85)
ax.axhline(0.5, color="black", linestyle="--", linewidth=1.2,
           label="Selection threshold (0.5)")
ax.set_xticks(range(n_features))
ax.set_xticklabels(short_names, rotation=45, ha="right", fontsize=7)
ax.set_ylabel("PSO Best Position Value", fontsize=11)
ax.set_title("Figure 7 - PSO Feature Weights  (Green=Selected, Red=Dropped)",
             fontsize=13, fontweight="bold")
ax.legend(fontsize=9); ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR,"fig7_pso_feature_weights.png"),
            dpi=FIG_DPI, bbox_inches="tight")
plt.close()
print("[FIGURE] fig7_pso_feature_weights.png saved")

# -- Save outputs -----------------------------------------------
np.save(os.path.join(RESULT_DIR,"pso_selected_mask.npy"), sel_mask)
np.save(os.path.join(RESULT_DIR,"pso_best_pos.npy"), best_pos)
pd.Series(sel_names).to_csv(os.path.join(RESULT_DIR,"selected_feature_names.csv"),
                              index=False, header=False)
print(f"\n{'='*60}")
print("  STEP 2 COMPLETE")
print(f"{'='*60}\n")
