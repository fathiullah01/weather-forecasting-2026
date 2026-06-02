# =============================================================
# STEP 1: DATA LOADING & PREPROCESSING
# Loads Country(Nigeria) NASA-POWER weather data for all 37 states
# =============================================================

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import *

for d in [FIG_DIR, TABLE_DIR, RESULT_DIR, MODEL_DIR]:
    os.makedirs(d, exist_ok=True)

print("=" * 60)
print("  STEP 1: Loading and Preprocessing Country Weather Data")
print("=" * 60)

# -- Load dataset -----------------------------------------------
if not os.path.exists(DATASET_FILE):
    print(f"\n[ERROR] Dataset not found: {DATASET_FILE}")
    print("  Please run: python download_country_weather_data.py first.")
    sys.exit(1)

df_raw = pd.read_csv(DATASET_FILE, index_col=0, parse_dates=True)
df_raw.index.name = "DATE"
df_raw = df_raw.sort_index()

print(f"\n[INFO] Raw dataset shape   : {df_raw.shape}")
print(f"[INFO] Date range          : {df_raw.index.min().date()} -> {df_raw.index.max().date()}")
print(f"[INFO] States in dataset   : {df_raw['STATE'].nunique()}")
print(f"[INFO] States              : {sorted(df_raw['STATE'].unique())}")

# -- Select mode: single or all states -------------------------
if FORECAST_MODE == "single":
    df_raw = df_raw[df_raw["STATE"] == SINGLE_STATE].copy()
    print(f"\n[CONFIG] FORECAST_MODE=single -> using {SINGLE_STATE} only")
else:
    print(f"\n[CONFIG] FORECAST_MODE=all -> using all {df_raw['STATE'].nunique()} states")

# -- Work with primary state for EDA and model training --------
df_primary = df_raw[df_raw["STATE"] == PRIMARY_STATE].copy()
print(f"[INFO] Primary state for training: {PRIMARY_STATE} ({len(df_primary)} rows)")

# Build column names for primary state
state_code   = PRIMARY_STATE.upper().replace(" ", "_")
feat_cols    = [f"{state_code}_{f}" for f in FEATURE_NAMES
                if f"{state_code}_{f}" in df_primary.columns]
target_col   = f"{state_code}_{TARGET_VARIABLE}"

if target_col not in df_primary.columns:
    # Try to find the right column
    candidates = [c for c in df_primary.columns if TARGET_VARIABLE in c]
    if candidates:
        target_col = candidates[0]
        state_code = target_col.split("_")[0]
        feat_cols  = [f"{state_code}_{f}" for f in FEATURE_NAMES
                      if f"{state_code}_{f}" in df_primary.columns]
    else:
        print(f"[ERROR] Target column '{target_col}' not found.")
        print(f"  Available: {list(df_primary.columns)}")
        sys.exit(1)

print(f"[INFO] Feature columns     : {len(feat_cols)}")
print(f"[INFO] Target column       : {target_col}")

# -- Select required columns ------------------------------------
keep_cols = ["STATE", "CAPITAL", "YEAR", "MONTH", "DAY"] + feat_cols + [target_col]
keep_cols = [c for c in keep_cols if c in df_primary.columns]
df = df_primary[keep_cols].copy()

# -- Missing value imputation -----------------------------------
missing_before = df.isnull().sum().sum()
df[feat_cols]  = df[feat_cols].fillna(df[feat_cols].mean())
df[target_col] = df[target_col].fillna(df[target_col].mean())
print(f"\n[INFO] Missing values before : {missing_before}")
print(f"[INFO] Missing values after  : {df.isnull().sum().sum()}")

# -- Lag features ----------------------------------------------
for col in feat_cols:
    df[f"{col}_lag1"] = df[col].shift(1)
df["target_precipitation"] = df[target_col].shift(-1)
df = df.dropna().reset_index(drop=True)
print(f"\n[INFO] Dataset after lag features : {df.shape}")

# -- Save feature/target references ----------------------------
lag_cols     = [f"{c}_lag1" for c in feat_cols]
all_features = feat_cols + lag_cols
pd.Series(feat_cols).to_csv(os.path.join(RESULT_DIR,"feat_cols.csv"),
                             index=False, header=False)
pd.Series(all_features).to_csv(os.path.join(RESULT_DIR,"all_features.csv"),
                                index=False, header=False)
with open(os.path.join(RESULT_DIR,"state_code.txt"), "w") as f:
    f.write(state_code)

# -- Table 1 - Summary statistics ------------------------------
summary = df[feat_cols + [target_col]].describe().round(3)
summary.to_csv(os.path.join(TABLE_DIR, "table1_summary_statistics.csv"))
print(f"\n[TABLE] table1_summary_statistics.csv saved")
print(summary.to_string())

# -- Table 2 - Missing values report ---------------------------
miss_df = pd.DataFrame({
    "Column"        : feat_cols + [target_col],
    "Missing_Before": [missing_before // len(feat_cols + [target_col])] * len(feat_cols + [target_col]),
    "Missing_After" : [0] * len(feat_cols + [target_col]),
    "Imputation"    : ["Column Mean"] * len(feat_cols + [target_col]),
})
miss_df.to_csv(os.path.join(TABLE_DIR, "table2_missing_values.csv"), index=False)
print(f"[TABLE] table2_missing_values.csv saved")

# -- Figure 1 - Feature distributions --------------------------
ncols = 5
nrows = int(np.ceil(len(feat_cols) / ncols))
fig, axes = plt.subplots(nrows, ncols, figsize=(20, nrows * 4))
axes = axes.flatten()
short = lambda c: c.replace(f"{state_code}_","").replace("_"," ").title()
for i, col in enumerate(feat_cols):
    axes[i].hist(df[col], bins=40, color="#2196F3", edgecolor="white", alpha=0.85)
    axes[i].set_title(short(col), fontsize=10, fontweight="bold")
    axes[i].set_xlabel("Value", fontsize=8)
    axes[i].set_ylabel("Frequency", fontsize=8)
    axes[i].grid(axis="y", alpha=0.3)
for j in range(i+1, len(axes)):
    axes[j].set_visible(False)
plt.suptitle(f"Figure 1 - Feature Distributions ({PRIMARY_STATE} State, NASA POWER)",
             fontsize=13, fontweight="bold", y=1.01)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR,"fig1_feature_distributions.png"),
            dpi=FIG_DPI, bbox_inches="tight")
plt.close()
print("\n[FIGURE] fig1_feature_distributions.png saved")

# -- Figure 2 - Correlation heatmap ----------------------------
corr_cols   = feat_cols + [target_col]
corr_matrix = df[corr_cols].corr().round(2)
corr_matrix.to_csv(os.path.join(TABLE_DIR,"table3_correlation_matrix.csv"))
fig, ax = plt.subplots(figsize=(13, 11))
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(corr_matrix, mask=mask, annot=True, fmt=".2f",
            cmap="coolwarm", center=0, linewidths=0.5,
            ax=ax, annot_kws={"size": 7})
ax.set_title("Figure 2 - Feature Correlation Heatmap",
             fontsize=13, fontweight="bold", pad=15)
lbls = [short(c) for c in corr_cols]
ax.set_xticklabels(lbls, rotation=45, ha="right", fontsize=7)
ax.set_yticklabels(lbls, rotation=0, fontsize=7)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR,"fig2_correlation_heatmap.png"),
            dpi=FIG_DPI, bbox_inches="tight")
plt.close()
print("[FIGURE] fig2_correlation_heatmap.png saved")
print("[TABLE] table3_correlation_matrix.csv saved")

# -- Figure 3 - Monthly precipitation trend --------------------
monthly = df.groupby("MONTH")[target_col].mean().round(3)
fig, ax  = plt.subplots(figsize=(12, 5))
colors_m = ["#1565C0" if v >= monthly.mean() else "#42A5F5" for v in monthly.values]
bars = ax.bar(monthly.index, monthly.values,
              color=colors_m, edgecolor="white", alpha=0.88)
ax.set_xlabel("Month", fontsize=11)
ax.set_ylabel("Avg Daily Precipitation (mm/day)", fontsize=11)
ax.set_title(f"Figure 3 - Average Monthly Precipitation ({PRIMARY_STATE} State)",
             fontsize=13, fontweight="bold")
ax.set_xticks(range(1,13))
ax.set_xticklabels(["Jan","Feb","Mar","Apr","May","Jun",
                    "Jul","Aug","Sep","Oct","Nov","Dec"])
ax.axhline(monthly.mean(), color="red", linestyle="--",
           linewidth=1.2, label=f"Annual mean: {monthly.mean():.2f} mm")
ax.legend(fontsize=9)
ax.grid(axis="y", alpha=0.3)
for bar, val in zip(bars, monthly.values):
    ax.text(bar.get_x()+bar.get_width()/2,
            bar.get_height()+0.05, f"{val:.1f}",
            ha="center", va="bottom", fontsize=8)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR,"fig3_monthly_precipitation.png"),
            dpi=FIG_DPI, bbox_inches="tight")
plt.close()
print("[FIGURE] fig3_monthly_precipitation.png saved")

# -- Figure 4 - Yearly precipitation trend ---------------------
yearly = df.groupby("YEAR")[target_col].mean().round(3)
fig, ax = plt.subplots(figsize=(14, 5))
ax.plot(yearly.index, yearly.values, color="#1565C0",
        linewidth=2, marker="o", markersize=4)
ax.fill_between(yearly.index, yearly.values,
                alpha=0.15, color="#1565C0")
ax.set_xlabel("Year", fontsize=11)
ax.set_ylabel("Avg Daily Precipitation (mm/day)", fontsize=11)
ax.set_title(f"Figure 4 - Yearly Precipitation Trend ({PRIMARY_STATE} State, 2000-2025)",
             fontsize=13, fontweight="bold")
ax.axhline(yearly.mean(), color="red", linestyle="--",
           linewidth=1.2, label=f"Overall mean: {yearly.mean():.2f} mm")
ax.legend(fontsize=9)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR,"fig4_yearly_trend.png"),
            dpi=FIG_DPI, bbox_inches="tight")
plt.close()
print("[FIGURE] fig4_yearly_trend.png saved")

# -- Figure 5 - Seasonal boxplot -------------------------------
def get_season(month):
    if month in [12, 1, 2]:  return "Harmattan\n(Dec-Feb)"
    if month in [3, 4, 5]:   return "Dry/Hot\n(Mar-May)"
    if month in [6, 7, 8]:   return "Rainy Peak\n(Jun-Aug)"
    return "Late Rains\n(Sep-Nov)"

df["Season"] = df["MONTH"].apply(get_season)
season_order = ["Harmattan\n(Dec-Feb)","Dry/Hot\n(Mar-May)",
                "Rainy Peak\n(Jun-Aug)","Late Rains\n(Sep-Nov)"]
fig, ax = plt.subplots(figsize=(11, 6))
df_box = [df[df["Season"]==s][target_col].values for s in season_order]
bp = ax.boxplot(df_box, patch_artist=True,
                medianprops=dict(color="white", linewidth=2))
ax.set_xticks(range(1, len(season_order)+1))
ax.set_xticklabels(season_order)
colors_s = ["#90CAF9","#FFCC02","#4CAF50","#FF9800"]
for patch, c in zip(bp["boxes"], colors_s):
    patch.set_facecolor(c); patch.set_alpha(0.8)
ax.set_ylabel("Daily Precipitation (mm/day)", fontsize=11)
ax.set_title(f"Figure 5 - Seasonal Precipitation Distribution ({PRIMARY_STATE} State)",
             fontsize=13, fontweight="bold")
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR,"fig5_seasonal_boxplot.png"),
            dpi=FIG_DPI, bbox_inches="tight")
plt.close()
print("[FIGURE] fig5_seasonal_boxplot.png saved")

# -- Table 4 - Seasonal statistics -----------------------------
seasonal_stats = df.groupby("Season")[target_col].agg(
    ["mean","std","min","max","count"]).round(3)
seasonal_stats.to_csv(os.path.join(TABLE_DIR,"table4_seasonal_statistics.csv"))
print("[TABLE] table4_seasonal_statistics.csv saved")

# -- Save preprocessed data -------------------------------------
df.to_csv(os.path.join(RESULT_DIR,"preprocessed_data.csv"), index=False)
print(f"\n[DATA] preprocessed_data.csv saved -> results/")
print(f"\n{'='*60}")
print("  STEP 1 COMPLETE")
print(f"{'='*60}\n")
