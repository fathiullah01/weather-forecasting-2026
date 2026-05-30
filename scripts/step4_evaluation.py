# =============================================================
# STEP 4: MODEL EVALUATION & COMPARISON
# PSO-RF vs Baseline RF vs ARIMA
# Metrics: MAE, RMSE, R2, Accuracy, MSE, MAPE
# Figures: actual vs predicted, residuals, scatter, metrics,
#          learning curve, seasonal breakdown
# =============================================================

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import joblib, os, sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import *
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.arima.model import ARIMA

print("=" * 60)
print("  STEP 4: Model Evaluation & Comparison")
print("=" * 60)

# -- Load -------------------------------------------------------
rf_pso      = joblib.load(os.path.join(MODEL_DIR,"rf_pso_model.pkl"))
rf_base     = joblib.load(os.path.join(MODEL_DIR,"rf_baseline_model.pkl"))
X_test_pso  = np.load(os.path.join(RESULT_DIR,"X_test_pso.npy"))
X_test_base = np.load(os.path.join(RESULT_DIR,"X_test_base.npy"))
y_test      = np.load(os.path.join(RESULT_DIR,"y_test.npy"))
y_train     = np.load(os.path.join(RESULT_DIR,"y_train.npy"))
X_train_pso = np.load(os.path.join(RESULT_DIR,"X_train_pso.npy"))
df          = pd.read_csv(os.path.join(RESULT_DIR,"preprocessed_data.csv"))
sel_names   = pd.read_csv(os.path.join(RESULT_DIR,"selected_feature_names.csv"),
                           header=None)[0].tolist()
state_code  = open(os.path.join(RESULT_DIR,"state_code.txt")).read().strip()

# -- Predictions ------------------------------------------------
y_pred_pso  = rf_pso.predict(X_test_pso)
y_pred_base = rf_base.predict(X_test_base)

# -- ARIMA ------------------------------------------------------
print("\n[ARIMA] Fitting ARIMA(2,1,2) ...")
try:
    arima_fit    = ARIMA(y_train, order=(2,1,2)).fit()
    y_pred_arima = arima_fit.forecast(steps=len(y_test))
    print("[ARIMA] Complete.")
except Exception as e:
    print(f"[ARIMA] Warning: {e} - using naive forecast")
    y_pred_arima = np.full(len(y_test), y_train[-1])

# -- Metrics ----------------------------------------------------
def mape(y_true, y_pred):
    mask = y_true != 0
    return np.mean(np.abs((y_true[mask]-y_pred[mask])/y_true[mask]))*100

def acc_thresh(y_true, y_pred, t=1.0):
    return np.mean(np.abs(y_true-y_pred) <= t)*100

def metrics(y_true, y_pred, label):
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mse  = mean_squared_error(y_true, y_pred)
    r2   = r2_score(y_true, y_pred)
    acc  = acc_thresh(y_true, y_pred, RAIN_THRESHOLD_MM)
    mp   = mape(y_true, y_pred)
    print(f"\n  [{label}]")
    print(f"    MAE={mae:.4f}  RMSE={rmse:.4f}  MSE={mse:.4f}  "
          f"R2={r2:.4f}  Acc={acc:.2f}%  MAPE={mp:.2f}%")
    return {"Model":label,"MAE":round(mae,4),"RMSE":round(rmse,4),
            "MSE":round(mse,4),"R2":round(r2,4),
            "Accuracy_%":round(acc,2),"MAPE_%":round(mp,2)}

print("\n[METRICS] Evaluating all models ...")
results = [
    metrics(y_test, y_pred_pso,  "PSO + Random Forest"),
    metrics(y_test, y_pred_base, "Baseline Random Forest"),
    metrics(y_test, y_pred_arima,"ARIMA(2,1,2)"),
]
res_df = pd.DataFrame(results)
res_df.to_csv(os.path.join(TABLE_DIR,"table7_model_comparison.csv"), index=False)
print(f"\n[TABLE] table7_model_comparison.csv saved")
print(res_df.to_string(index=False))

# -- Table 8 - Literature comparison ---------------------------
lit_df = pd.DataFrame(LITERATURE_REFS)
# Fill in this study's results
mask_this = lit_df["Author"] == "This Study"
lit_df.loc[mask_this, "MAE"]  = round(results[0]["MAE"],  4)
lit_df.loc[mask_this, "RMSE"] = round(results[0]["RMSE"], 4)
lit_df.loc[mask_this, "R2"]   = round(results[0]["R2"],   4)
lit_df.loc[mask_this, "MSE"]  = round(results[0]["MSE"],  4)
lit_df.loc[mask_this, "MAPE"] = str(round(results[0]["MAPE_%"],2)) + "%"
lit_df.to_csv(os.path.join(TABLE_DIR,"table8_literature_comparison.csv"), index=False)
print("[TABLE] table8_literature_comparison.csv saved")

n_plot = min(300, len(y_test))

# -- Figure 9 - Actual vs Predicted (PSO-RF) -------------------
fig, ax = plt.subplots(figsize=(14, 5))
ax.plot(range(n_plot), y_test[:n_plot],
        label="Actual", color="#1565C0", linewidth=1.5, alpha=0.9)
ax.plot(range(n_plot), y_pred_pso[:n_plot],
        label="PSO-RF Predicted", color="#E91E63",
        linewidth=1.4, linestyle="--", alpha=0.85)
ax.set_xlabel("Day Index (Test Set)", fontsize=11)
ax.set_ylabel("Precipitation (mm/day)", fontsize=11)
ax.set_title("Figure 9 - Actual vs PSO-RF Predicted Precipitation",
             fontsize=13, fontweight="bold")
ax.legend(fontsize=10); ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR,"fig9_actual_vs_predicted.png"),
            dpi=FIG_DPI, bbox_inches="tight")
plt.close()
print("\n[FIGURE] fig9_actual_vs_predicted.png saved")

# -- Figure 10 - Three model comparison ------------------------
fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True)
for ax, (lbl, yp, col) in zip(axes, [
    ("PSO + Random Forest", y_pred_pso,   "#E91E63"),
    ("Baseline RF",          y_pred_base,  "#FF9800"),
    ("ARIMA(2,1,2)",         y_pred_arima, "#9C27B0"),
]):
    ax.plot(range(n_plot), y_test[:n_plot],
            label="Actual", color="#1565C0", linewidth=1.5)
    ax.plot(range(n_plot), yp[:n_plot],
            label=lbl, color=col, linewidth=1.4, linestyle="--")
    ax.set_ylabel("Precipitation", fontsize=9)
    ax.legend(fontsize=9, loc="upper right"); ax.grid(alpha=0.3)
axes[-1].set_xlabel("Day Index (Test Set)", fontsize=11)
fig.suptitle("Figure 10 - Model Comparison: Actual vs Predicted",
             fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR,"fig10_three_model_comparison.png"),
            dpi=FIG_DPI, bbox_inches="tight")
plt.close()
print("[FIGURE] fig10_three_model_comparison.png saved")

# -- Figure 11 - Metrics bar chart -----------------------------
met_cols = ["MAE","RMSE","R2","Accuracy_%","MAPE_%"]
fig, axes = plt.subplots(1, len(met_cols), figsize=(22, 5))
colors3 = ["#E91E63","#FF9800","#9C27B0"]
x = np.arange(3)
for ax, mc in zip(axes, met_cols):
    vals = [r[mc] for r in results]
    bars = ax.bar(x, vals, color=colors3, edgecolor="white",
                  alpha=0.85, width=0.5)
    ax.set_title(mc, fontsize=11, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(["PSO-RF","Base-RF","ARIMA"], fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x()+bar.get_width()/2,
                bar.get_height()+max(vals)*0.01,
                f"{val:.3f}", ha="center", va="bottom", fontsize=7)
fig.suptitle("Figure 11 - Model Performance Metrics Comparison",
             fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR,"fig11_metrics_comparison.png"),
            dpi=FIG_DPI, bbox_inches="tight")
plt.close()
print("[FIGURE] fig11_metrics_comparison.png saved")

# -- Figure 12 - Scatter actual vs predicted -------------------
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
for ax, (lbl, yp, col) in zip(axes, [
    ("PSO-RF",    y_pred_pso,   "#E91E63"),
    ("Base-RF",   y_pred_base,  "#FF9800"),
    ("ARIMA",     y_pred_arima, "#9C27B0"),
]):
    ax.scatter(y_test, yp, alpha=0.35, color=col, s=15,
               edgecolors="white", linewidths=0.2)
    lims = [min(y_test.min(), yp.min()), max(y_test.max(), yp.max())]
    ax.plot(lims, lims, "k--", linewidth=1.3, label="Perfect fit")
    r2val = r2_score(y_test, yp)
    ax.set_title(f"{lbl}  (R2={r2val:.3f})", fontsize=11, fontweight="bold")
    ax.set_xlabel("Actual (mm/day)", fontsize=9)
    ax.set_ylabel("Predicted (mm/day)", fontsize=9)
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
fig.suptitle("Figure 12 - Actual vs Predicted Scatter (All Models)",
             fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR,"fig12_scatter_all_models.png"),
            dpi=FIG_DPI, bbox_inches="tight")
plt.close()
print("[FIGURE] fig12_scatter_all_models.png saved")

# -- Figure 13 - Residual error plot ---------------------------
residuals = y_test - y_pred_pso
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].plot(range(n_plot), residuals[:n_plot],
             color="#1565C0", linewidth=1, alpha=0.7)
axes[0].axhline(0, color="red", linestyle="--", linewidth=1.2)
axes[0].axhline(residuals.std(), color="orange", linestyle=":",
                linewidth=1, label=f"+1 SD ({residuals.std():.2f})")
axes[0].axhline(-residuals.std(), color="orange", linestyle=":",
                linewidth=1, label=f"-1 SD")
axes[0].set_xlabel("Day Index", fontsize=10)
axes[0].set_ylabel("Residual (Actual - Predicted)", fontsize=10)
axes[0].set_title("Residual Plot Over Time", fontsize=11, fontweight="bold")
axes[0].legend(fontsize=8); axes[0].grid(alpha=0.3)
axes[1].hist(residuals, bins=40, color="#E91E63", edgecolor="white", alpha=0.85)
axes[1].axvline(0, color="black", linestyle="--", linewidth=1.5)
axes[1].axvline(residuals.mean(), color="blue", linestyle="-",
                linewidth=1.5, label=f"Mean={residuals.mean():.3f}")
axes[1].set_xlabel("Residual Value", fontsize=10)
axes[1].set_ylabel("Frequency", fontsize=10)
axes[1].set_title("Residual Distribution", fontsize=11, fontweight="bold")
axes[1].legend(fontsize=8); axes[1].grid(axis="y", alpha=0.3)
fig.suptitle("Figure 13 - PSO-RF Residual Error Analysis",
             fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR,"fig13_residual_analysis.png"),
            dpi=FIG_DPI, bbox_inches="tight")
plt.close()
print("[FIGURE] fig13_residual_analysis.png saved")

# -- Figure 14 - Learning curve --------------------------------
from sklearn.model_selection import learning_curve
rf_params = rf_pso.get_params()
train_sizes, train_scores, val_scores = learning_curve(
    RandomForestRegressor(n_estimators=rf_params["n_estimators"],
                           max_depth=rf_params["max_depth"],
                           random_state=42, n_jobs=-1),
    X_train_pso, y_train,
    train_sizes=np.linspace(0.1, 1.0, 8),
    cv=3, scoring="r2", n_jobs=-1)
tr_mean = train_scores.mean(axis=1)
va_mean = val_scores.mean(axis=1)
tr_std  = train_scores.std(axis=1)
va_std  = val_scores.std(axis=1)
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(train_sizes, tr_mean, "o-", color="#1565C0", label="Training R2")
ax.plot(train_sizes, va_mean, "s-", color="#E91E63", label="Validation R2")
ax.fill_between(train_sizes, tr_mean-tr_std, tr_mean+tr_std, alpha=0.15, color="#1565C0")
ax.fill_between(train_sizes, va_mean-va_std, va_mean+va_std, alpha=0.15, color="#E91E63")
ax.set_xlabel("Training Set Size", fontsize=11)
ax.set_ylabel("R2 Score", fontsize=11)
ax.set_title("Figure 14 - Learning Curve (PSO-RF Model)",
             fontsize=13, fontweight="bold")
ax.legend(fontsize=10); ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR,"fig14_learning_curve.png"),
            dpi=FIG_DPI, bbox_inches="tight")
plt.close()
print("[FIGURE] fig14_learning_curve.png saved")

# -- Figure 15 - Seasonal performance --------------------------
n_test       = len(y_test)
df_test_full = df.iloc[-n_test:].copy()
df_test_full["Actual"]    = y_test
df_test_full["PSO_RF"]    = y_pred_pso
df_test_full["Base_RF"]   = y_pred_base
df_test_full["ARIMA_Pred"]= np.array(y_pred_arima)

def get_season(m):
    if m in [12,1,2]:  return "Harmattan"
    if m in [3,4,5]:   return "Dry/Hot"
    if m in [6,7,8]:   return "Rainy Peak"
    return "Late Rains"

df_test_full["Season"] = df_test_full["MONTH"].apply(get_season)
season_order = ["Harmattan","Dry/Hot","Rainy Peak","Late Rains"]
sea_metrics  = []
for s in season_order:
    sub = df_test_full[df_test_full["Season"]==s]
    if len(sub) < 5: continue
    sea_metrics.append({
        "Season": s,
        "PSO_RF_MAE"  : round(mean_absolute_error(sub["Actual"], sub["PSO_RF"]),4),
        "PSO_RF_RMSE" : round(np.sqrt(mean_squared_error(sub["Actual"], sub["PSO_RF"])),4),
        "PSO_RF_R2"   : round(r2_score(sub["Actual"], sub["PSO_RF"]),4),
        "Base_RF_MAE" : round(mean_absolute_error(sub["Actual"], sub["Base_RF"]),4),
        "ARIMA_MAE"   : round(mean_absolute_error(sub["Actual"], sub["ARIMA_Pred"]),4),
    })
sea_df = pd.DataFrame(sea_metrics)
sea_df.to_csv(os.path.join(TABLE_DIR,"table9_seasonal_performance.csv"), index=False)
print("[TABLE] table9_seasonal_performance.csv saved")

x_s = np.arange(len(sea_df))
w   = 0.25
fig, ax = plt.subplots(figsize=(12, 5))
ax.bar(x_s-w, sea_df["PSO_RF_MAE"],  w, label="PSO-RF",   color="#E91E63", alpha=0.85)
ax.bar(x_s,   sea_df["Base_RF_MAE"], w, label="Base-RF",  color="#FF9800", alpha=0.85)
ax.bar(x_s+w, sea_df["ARIMA_MAE"],   w, label="ARIMA",    color="#9C27B0", alpha=0.85)
ax.set_xticks(x_s)
ax.set_xticklabels(sea_df["Season"], fontsize=10)
ax.set_ylabel("MAE (mm/day)", fontsize=11)
ax.set_title("Figure 15 - Seasonal MAE Comparison (All Models)",
             fontsize=13, fontweight="bold")
ax.legend(fontsize=10); ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR,"fig15_seasonal_performance.png"),
            dpi=FIG_DPI, bbox_inches="tight")
plt.close()
print("[FIGURE] fig15_seasonal_performance.png saved")

# -- Save predictions -------------------------------------------
pred_df = pd.DataFrame({
    "Actual"          : y_test,
    "PSO_RF_Pred"     : y_pred_pso.round(4),
    "Baseline_RF_Pred": y_pred_base.round(4),
    "ARIMA_Pred"      : np.array(y_pred_arima).round(4),
    "Residual_PSO_RF" : residuals.round(4),
})
pred_df.to_csv(os.path.join(RESULT_DIR,"all_predictions.csv"), index=False)
print("\n[DATA] all_predictions.csv saved")

print(f"\n{'='*60}")
print("  STEP 4 COMPLETE")
print(f"{'='*60}\n")
