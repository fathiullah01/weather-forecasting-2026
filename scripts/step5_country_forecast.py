# =============================================================
# STEP 5: COUNTRY WEATHER FORECASTING DASHBOARD
# Forecasts next-day precipitation for:
#   - All states (FORECAST_MODE = "all")
#   - One specific state (FORECAST_MODE = "single")
# All variables read from config.py
# =============================================================

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
import joblib, os, sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import *
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

print("=" * 60)
print("  STEP 5: Country Weather Forecasting Dashboard")
print(f"  Mode   : {FORECAST_MODE.upper()}")
print("=" * 60)

# -- States loaded from config.py ------------------------------
# Format in config: (state_key, state_name, capital_city, lat, lon)
# To change states edit the STATES list in config.py only
STATES_LIST = [(s[1], s[2], s[3], s[4]) for s in STATES]

# -- Load trained PSO-RF model ---------------------------------
rf_pso     = joblib.load(os.path.join(MODEL_DIR, "rf_pso_model.pkl"))
rf_params  = rf_pso.get_params()
sel_names  = pd.read_csv(os.path.join(RESULT_DIR, "selected_feature_names.csv"),
                          header=None)[0].tolist()
state_code = open(os.path.join(RESULT_DIR, "state_code.txt")).read().strip()

# -- Load full dataset -----------------------------------------
df_raw = pd.read_csv(DATASET_FILE, index_col=0, parse_dates=True)
df_raw = df_raw.sort_index()

# -- Filter states based on mode -------------------------------
if FORECAST_MODE == "single":
    states_to_run = [(s, c, la, lo) for s, c, la, lo in STATES_LIST
                     if s == SINGLE_STATE]
    if not states_to_run:
        print(f"[ERROR] State '{SINGLE_STATE}' not found.")
        print(f"  Available states: {[s[0] for s in STATES_LIST]}")
        print(f"  Check SINGLE_STATE in config.py")
        sys.exit(1)
else:
    states_to_run = STATES_LIST

print(f"\n[INFO] Running forecast for {len(states_to_run)} state(s)...")

# -- Per-state forecast helper ---------------------------------
def build_state_data(state_name, df_raw):
    sc   = state_name.upper().replace(" ", "_")
    cols = [c for c in df_raw.columns if c.startswith(sc + "_") and
            any(f in c for f in FEATURE_NAMES + [TARGET_VARIABLE])]
    if len(cols) < 3:
        return None, None, None, None
    df_s = df_raw[df_raw["STATE"] == state_name][
        ["STATE", "CAPITAL", "YEAR", "MONTH", "DAY"] + cols].copy()
    tgt  = f"{sc}_{TARGET_VARIABLE}"
    if tgt not in df_s.columns:
        return None, None, None, None
    feats = [c for c in cols if c != tgt]
    for c in feats:
        df_s[f"{c}_lag1"] = df_s[c].shift(1)
    df_s["target"] = df_s[tgt].shift(-1)
    df_s = df_s.dropna().reset_index(drop=True)
    all_f = feats + [f"{c}_lag1" for c in feats]
    return df_s, all_f, feats, tgt

# -- Run per-state forecast ------------------------------------
state_results = []
print(f"\n{'State':<16} {'Capital':<16} {'R2':>6} {'MAE':>7} "
      f"{'RMSE':>7} {'Forecast(mm)':>13} {'Rain?':>6}")
print("-" * 72)

for state_name, capital, lat, lon in states_to_run:
    df_s, all_f, feats, tgt = build_state_data(state_name, df_raw)
    if df_s is None or len(df_s) < 100:
        continue

    X_s  = df_s[all_f].values
    y_s  = df_s["target"].values
    sc_s = StandardScaler()
    X_sc = sc_s.fit_transform(X_s)

    split      = int(len(X_sc) * 0.8)
    X_tr, X_te = X_sc[:split], X_sc[split:]
    y_tr, y_te = y_s[:split],  y_s[split:]

    rf_s = RandomForestRegressor(
        n_estimators = rf_params["n_estimators"],
        max_depth    = rf_params["max_depth"],
        random_state = 42, n_jobs=-1)
    rf_s.fit(X_tr, y_tr)

    y_pr = rf_s.predict(X_te)
    mae  = mean_absolute_error(y_te, y_pr)
    rmse = np.sqrt(mean_squared_error(y_te, y_pr))
    r2   = r2_score(y_te, y_pr)
    nxt  = rf_s.predict(X_sc[-1].reshape(1, -1))[0]
    rain = "YES" if nxt >= RAIN_THRESHOLD_MM else "No"

    print(f"  {state_name:<14} {capital:<16} {r2:>6.3f} {mae:>7.4f} "
          f"{rmse:>7.4f} {nxt:>13.3f} {rain:>6}")

    state_results.append({
        "State"               : state_name,
        "Capital"             : capital,
        "Latitude"            : lat,
        "Longitude"           : lon,
        "R2"                  : round(r2,   4),
        "MAE"                 : round(mae,  4),
        "RMSE"                : round(rmse, 4),
        "Next_Day_Forecast_mm": round(nxt,  3),
        "Rain_Likely"         : rain,
    })

st_df = pd.DataFrame(state_results)
st_df.to_csv(os.path.join(TABLE_DIR, "table10_country_state_forecasts.csv"),
             index=False)
print(f"\n[TABLE] table10_country_state_forecasts.csv saved ({len(st_df)} states)")

if len(st_df) == 0:
    print("[ERROR] No state results. Check dataset columns.")
    sys.exit(1)

# ==============================================================
# FIGURES
# ==============================================================

# -- Figure 16 - Feature importance (PSO-RF, primary state) ---
importances = rf_pso.feature_importances_
feat_labels = [f.replace(f"{state_code}_", "").replace("_lag1", " (lag1)")
               for f in sel_names]
imp_df = pd.DataFrame({"Feature": feat_labels, "Importance": importances})
imp_df = imp_df.sort_values("Importance", ascending=True)

fig, ax = plt.subplots(figsize=(10, max(6, len(imp_df) * 0.4)))
bars = ax.barh(imp_df["Feature"], imp_df["Importance"],
               color="#2196F3", edgecolor="white", alpha=0.85)
ax.set_xlabel("Feature Importance (Gini)", fontsize=11)
ax.set_title(f"Figure 16 - PSO-RF Feature Importances ({PRIMARY_STATE} State)",
             fontsize=13, fontweight="bold")
ax.grid(axis="x", alpha=0.3)
for bar, val in zip(bars, imp_df["Importance"]):
    ax.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height() / 2,
            f"{val:.3f}", va="center", fontsize=8)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "fig16_feature_importance.png"),
            dpi=FIG_DPI, bbox_inches="tight")
plt.close()
print("\n[FIGURE] fig16_feature_importance.png saved")

# -- Figure 17 - Next-day forecast by state --------------------
st_sorted  = st_df.sort_values("Next_Day_Forecast_mm", ascending=True)
bar_colors = ["#4CAF50" if r == "YES" else "#90A4AE"
              for r in st_sorted["Rain_Likely"]]
fig, ax = plt.subplots(figsize=(14, max(8, len(st_sorted) * 0.45)))
bars = ax.barh(st_sorted["State"], st_sorted["Next_Day_Forecast_mm"],
               color=bar_colors, edgecolor="white", alpha=0.88)
ax.axvline(RAIN_THRESHOLD_MM, color="red", linestyle="--", linewidth=1.3,
           label=f"Rain threshold ({RAIN_THRESHOLD_MM} mm)")
ax.set_xlabel("Next-Day Precipitation Forecast (mm/day)", fontsize=11)
ax.set_title(f"Figure 17 - Next-Day Precipitation Forecast by {COUNTRY_NAME} State\n"
             "(Green = Rain Likely, Grey = Dry)",
             fontsize=13, fontweight="bold")
for bar, row in zip(bars, st_sorted.itertuples()):
    ax.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height() / 2,
            f"R2={row.R2:.2f}", va="center", fontsize=7)
ax.legend(fontsize=9)
ax.grid(axis="x", alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "fig17_country_state_forecast.png"),
            dpi=FIG_DPI, bbox_inches="tight")
plt.close()
print("[FIGURE] fig17_country_state_forecast.png saved")

# -- Figure 18 - R2 score by state ----------------------------
st_r2    = st_df.sort_values("R2", ascending=True)
colors18 = ["#4CAF50" if v >= 0.6 else "#FF9800" if v >= 0.4 else "#F44336"
            for v in st_r2["R2"]]
fig, ax = plt.subplots(figsize=(14, max(8, len(st_r2) * 0.45)))
bars = ax.barh(st_r2["State"], st_r2["R2"],
               color=colors18, edgecolor="white", alpha=0.88)
ax.axvline(0.6, color="green",  linestyle="--", linewidth=1.2, label="Good (R2>=0.6)")
ax.axvline(0.4, color="orange", linestyle="--", linewidth=1.2, label="Fair (R2>=0.4)")
ax.set_xlabel("R2 Score", fontsize=11)
ax.set_title(f"Figure 18 - PSO-RF Model R2 Score per {COUNTRY_NAME} State",
             fontsize=13, fontweight="bold")
ax.legend(fontsize=9)
ax.grid(axis="x", alpha=0.3)
for bar, val in zip(bars, st_r2["R2"]):
    ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height() / 2,
            f"{val:.3f}", va="center", fontsize=8)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "fig18_country_r2_scores.png"),
            dpi=FIG_DPI, bbox_inches="tight")
plt.close()
print("[FIGURE] fig18_country_r2_scores.png saved")

# -- Figure 19 - Map-style scatter (lat/lon colored) ----------
fig, ax = plt.subplots(figsize=(12, 10))
sc = ax.scatter(st_df["Longitude"], st_df["Latitude"],
                c=st_df["Next_Day_Forecast_mm"],
                cmap="YlOrRd", s=200, edgecolors="black",
                linewidths=0.6, alpha=0.9, zorder=3)
cbar = plt.colorbar(sc, ax=ax, shrink=0.7)
cbar.set_label("Forecast Precipitation (mm/day)", fontsize=10)
for _, row in st_df.iterrows():
    ax.annotate(row["State"],
                xy=(row["Longitude"], row["Latitude"]),
                xytext=(3, 3), textcoords="offset points",
                fontsize=6.5, color="black")
ax.set_xlabel("Longitude", fontsize=11)
ax.set_ylabel("Latitude", fontsize=11)
ax.set_title(f"Figure 19 - {COUNTRY_NAME} Precipitation Forecast Map\n"
             "(Next-Day Forecast by State Capital)",
             fontsize=13, fontweight="bold")
# Map bounds auto-calculated from actual state coordinates
lats = [s[3] for s in STATES]
lons = [s[4] for s in STATES]
ax.set_xlim(min(lons) - 1, max(lons) + 1)
ax.set_ylim(min(lats) - 1, max(lats) + 1)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "fig19_country_forecast_map.png"),
            dpi=FIG_DPI, bbox_inches="tight")
plt.close()
print("[FIGURE] fig19_country_forecast_map.png saved")

# -- Figure 20 - MAE comparison across states -----------------
st_mae = st_df.sort_values("MAE", ascending=False)
fig, ax = plt.subplots(figsize=(14, max(8, len(st_mae) * 0.45)))
ax.barh(st_mae["State"], st_mae["MAE"],
        color="#FF5722", edgecolor="white", alpha=0.85)
ax.set_xlabel("MAE (mm/day)", fontsize=11)
ax.set_title(f"Figure 20 - PSO-RF Mean Absolute Error per {COUNTRY_NAME} State",
             fontsize=13, fontweight="bold")
ax.grid(axis="x", alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "fig20_country_mae_comparison.png"),
            dpi=FIG_DPI, bbox_inches="tight")
plt.close()
print("[FIGURE] fig20_country_mae_comparison.png saved")

# -- Final console summary ------------------------------------
print(f"\n{'='*60}")
print(f"  {COUNTRY_NAME.upper()} FORECAST SUMMARY")
print(f"{'='*60}")
print(st_df[["State", "Capital", "R2", "Next_Day_Forecast_mm", "Rain_Likely"]]
      .sort_values("R2", ascending=False).to_string(index=False))

rain_count = (st_df["Rain_Likely"] == "YES").sum()
dry_count  = (st_df["Rain_Likely"] == "No").sum()
print(f"\n  States expecting rain today : {rain_count}")
print(f"  States expecting dry day   : {dry_count}")
print(f"  Avg R2 across all states   : {st_df['R2'].mean():.3f}")
print(f"  Best performing state      : {st_df.loc[st_df['R2'].idxmax(), 'State']}")
print(f"  Lowest R2 state            : {st_df.loc[st_df['R2'].idxmin(), 'State']}")

print(f"\n{'='*60}")
print("  STEP 5 COMPLETE - All figures and tables generated.")
print(f"{'='*60}\n")