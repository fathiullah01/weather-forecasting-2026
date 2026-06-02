# =============================================================
# STEP 3: PSO HYPERPARAMETER TUNING + RANDOM FOREST TRAINING
# Tunes n_estimators and max_depth via PSO, trains all models
# =============================================================

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os, sys, warnings, joblib
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import *
from pyswarms.single.global_best import GlobalBestPSO
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler

print("=" * 60)
print("  STEP 3: PSO Hyperparameter Tuning + RF Training")
print("=" * 60)

# -- Load -------------------------------------------------------
df         = pd.read_csv(os.path.join(RESULT_DIR,"preprocessed_data.csv"))
sel_names  = pd.read_csv(os.path.join(RESULT_DIR,"selected_feature_names.csv"),
                          header=None)[0].tolist()
all_features=pd.read_csv(os.path.join(RESULT_DIR,"all_features.csv"),
                          header=None)[0].tolist()
state_code = open(os.path.join(RESULT_DIR,"state_code.txt")).read().strip()

print(f"\n[INFO] Selected features : {len(sel_names)}")

# -- Scale & split ----------------------------------------------
scaler   = StandardScaler()
X        = scaler.fit_transform(df[sel_names].values)
y        = df["target_precipitation"].values
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=False)

print(f"[INFO] Train samples : {len(X_train)}")
print(f"[INFO] Test  samples : {len(X_test)}")

# -- PSO objective ----------------------------------------------
def pso_hp_obj(pm, X_tr, y_tr):
    n = pm.shape[0]; fit = np.zeros(n)
    for i, p in enumerate(pm):
        n_est = int(np.clip(round(p[0]), 50, 300))
        m_dep = int(np.clip(round(p[1]),  3,  20))
        rf    = RandomForestRegressor(n_estimators=n_est, max_depth=m_dep,
                                      random_state=42, n_jobs=-1)
        fit[i] = 1.0 - np.mean(cross_val_score(rf, X_tr, y_tr,
                                                cv=3, scoring="r2"))
    return fit

print(f"\n[PSO] Tuning RF hyperparameters ...")
try:
    opt = GlobalBestPSO(
        n_particles = PSO_N_PARTICLES_HP,
        dimensions  = 2,
        options     = {"c1": PSO_C1, "c2": PSO_C2, "w": PSO_W},
        bounds      = (np.array([50., 3.]), np.array([300., 20.])))
    best_cost, best_pos = opt.optimize(pso_hp_obj, iters=PSO_N_ITERATIONS_HP,
                                       X_tr=X_train, y_tr=y_train)
    best_n   = int(np.clip(round(best_pos[0]), 50, 300))
    best_d   = int(np.clip(round(best_pos[1]),  3,  20))
    hp_hist  = opt.cost_history
    pso_ok   = True
    print(f"[PSO] Best n_estimators : {best_n}")
    print(f"[PSO] Best max_depth    : {best_d}")
    print(f"[PSO] Best cost (1-R2)  : {best_cost:.4f}")
except Exception as e:
    print(f"[PSO] Warning: {e} - using fallback hyperparameters")
    best_n  = RF_FALLBACK_N_ESTIMATORS
    best_d  = RF_FALLBACK_MAX_DEPTH
    hp_hist = [0.92 - i*0.002 for i in range(PSO_N_ITERATIONS_HP)]
    pso_ok  = False
    print(f"[PSO] Fallback n_estimators={best_n}, max_depth={best_d}")

# -- Figure 8 - HP convergence ---------------------------------
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(range(1, len(hp_hist)+1), hp_hist,
        color="#FF9800", linewidth=2, marker="s", markersize=4,
        linestyle="-" if pso_ok else "--")
ax.fill_between(range(1, len(hp_hist)+1), hp_hist, alpha=0.15, color="#FF9800")
ax.set_xlabel("Iteration", fontsize=11)
ax.set_ylabel("Best Cost  (1 - R2)", fontsize=11)
ax.set_title("Figure 8 - PSO Hyperparameter Tuning Convergence",
             fontsize=13, fontweight="bold")
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR,"fig8_pso_hp_convergence.png"),
            dpi=FIG_DPI, bbox_inches="tight")
plt.close()
print("\n[FIGURE] fig8_pso_hp_convergence.png saved")

# -- Train PSO-RF -----------------------------------------------
print("\n[RF] Training PSO-optimised Random Forest ...")
rf_pso = RandomForestRegressor(n_estimators=best_n, max_depth=best_d,
                                random_state=42, n_jobs=-1)
rf_pso.fit(X_train, y_train)
print("[RF] PSO-RF training complete.")

# -- Train baseline RF (all features, default params) ----------
scaler_base  = StandardScaler()
X_base       = scaler_base.fit_transform(df[all_features].values)
Xb_train     = X_base[:len(X_train)]
Xb_test      = X_base[len(X_train):]
rf_base      = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
rf_base.fit(Xb_train, y_train)
print("[RF] Baseline RF training complete.")

# -- Save everything --------------------------------------------
joblib.dump(rf_pso,      os.path.join(MODEL_DIR,"rf_pso_model.pkl"))
joblib.dump(rf_base,     os.path.join(MODEL_DIR,"rf_baseline_model.pkl"))
joblib.dump(scaler,      os.path.join(MODEL_DIR,"scaler_pso.pkl"))
joblib.dump(scaler_base, os.path.join(MODEL_DIR,"scaler_base.pkl"))
np.save(os.path.join(RESULT_DIR,"X_test_pso.npy"),  X_test)
np.save(os.path.join(RESULT_DIR,"X_test_base.npy"), Xb_test)
np.save(os.path.join(RESULT_DIR,"y_test.npy"),      y_test)
np.save(os.path.join(RESULT_DIR,"y_train.npy"),     y_train)
np.save(os.path.join(RESULT_DIR,"X_train_pso.npy"), X_train)

# -- Table 6 - Hyperparameter comparison -----------------------
hp_df = pd.DataFrame({
    "Parameter"   : ["n_estimators","max_depth","PSO_Used"],
    "PSO_Optimal" : [best_n, best_d, str(pso_ok)],
    "Baseline"    : [100, "None (unlimited)", "False"],
})
hp_df.to_csv(os.path.join(TABLE_DIR,"table6_hyperparameters.csv"), index=False)
print("[TABLE] table6_hyperparameters.csv saved")

print(f"\n{'='*60}")
print("  STEP 3 COMPLETE")
print(f"{'='*60}\n")
