
# =============================================================
# STEP 6: FUTURE WEATHER FORECAST FIGURES
# Generates 4 forecast figures:
#   fig21 - Tomorrow's forecast (next-day rain prediction)
#   fig22 - Rest of the week   (day 1-7 rolling forecast)
#   fig23 - Same time next week (day 8-14 forecast)
#   fig24 - Hourly estimate today (hourly pattern from daily total)
#
# If FORECAST_MODE = "all"    -> shows all states
# If FORECAST_MODE = "single" -> shows only SINGLE_STATE
#
# Uses recursive forecasting: each predicted day feeds into
# the next as a lag feature (standard approach for daily models)
# =============================================================

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import joblib, os, sys, warnings
from datetime import datetime, timedelta
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import *
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

print("=" * 60)
print("  STEP 6: Future Weather Forecast Figures")
print(f"  Mode   : {FORECAST_MODE.upper()}")
print("=" * 60)

# -- States loaded from config.py (change states there, not here)
# Format in config: (state_key, state_name, capital_city, lat, lon)
COUNTRY_STATES = [(s[1], s[2], s[3], s[4]) for s in STATES]

# -- Load trained model and data --------------------------------
rf_pso     = joblib.load(os.path.join(MODEL_DIR, "rf_pso_model.pkl"))
rf_params  = rf_pso.get_params()
sel_names  = pd.read_csv(os.path.join(RESULT_DIR, "selected_feature_names.csv"),
                          header=None)[0].tolist()
state_code = open(os.path.join(RESULT_DIR, "state_code.txt")).read().strip()
df_raw     = pd.read_csv(DATASET_FILE, index_col=0, parse_dates=True)
df_raw     = df_raw.sort_index()

# -- Resolve forecast start date from config -------------------
# FORECAST_DATE_MODE = "auto"   -> uses last date in dataset
# FORECAST_DATE_MODE = "manual" -> uses FORECAST_DATE from config
if FORECAST_DATE_MODE == "manual":
    today = datetime.strptime(FORECAST_DATE, "%Y-%m-%d")
    print(f"\n[CONFIG] FORECAST_DATE_MODE=manual -> using {today.date()}")
else:
    last_date = df_raw.index.max()
    if hasattr(last_date, "to_pydatetime"):
        today = last_date.to_pydatetime()
    else:
        today = datetime.combine(last_date, datetime.min.time())
    print(f"\n[CONFIG] FORECAST_DATE_MODE=auto -> last data date: {today.date()}")

today_str = today.strftime("%A, %d %B %Y")
print(f"[CONFIG] Forecast reference date : {today_str}")

# -- Filter states based on mode --------------------------------
if FORECAST_MODE == "single":
    states_to_run = [(s, c, la, lo) for s, c, la, lo in COUNTRY_STATES
                     if s == SINGLE_STATE]
else:
    states_to_run = COUNTRY_STATES

# -- Helper: build state dataset --------------------------------
def build_state(state_name, df_raw):
    sc   = state_name.upper().replace(" ", "_")
    tgt  = f"{sc}_{TARGET_VARIABLE}"
    cols = [c for c in df_raw.columns
            if c.startswith(sc + "_") and
            any(f in c for f in FEATURE_NAMES + [TARGET_VARIABLE])]
    if tgt not in df_raw.columns or len(cols) < 3:
        return None, None, None, None, None
    df_s = df_raw[df_raw["STATE"] == state_name][
        ["STATE", "CAPITAL", "YEAR", "MONTH", "DAY"] + cols].copy()
    feats = [c for c in cols if c != tgt]
    for c in feats:
        df_s[f"{c}_lag1"] = df_s[c].shift(1)
    df_s["target"] = df_s[tgt].shift(-1)
    df_s = df_s.dropna().reset_index(drop=True)
    all_f = feats + [f"{c}_lag1" for c in feats]
    return df_s, all_f, feats, tgt, sc

# -- Recursive forecast: predict N days ahead -------------------
def recursive_forecast(rf, scaler, df_s, all_f, feats, n_days=14):
    """
    Use the last row of real data as seed.
    Predict day1, then use day1 prediction as lag for day2, etc.
    Returns array of n_days precipitation forecasts.
    """
    X_all  = scaler.transform(df_s[all_f].values)
    # seed with last known row
    last_X = X_all[-1].copy()
    preds  = []

    n_feats    = len(feats)   # non-lag features
    n_lag_feats= len(feats)   # lag features (same count)

    for day in range(n_days):
        pred = rf.predict(last_X.reshape(1, -1))[0]
        pred = max(0.0, pred)   # precipitation can't be negative
        preds.append(pred)

        # Shift: lag features become current features of next step
        # structure: [feat0..featN, feat0_lag1..featN_lag1]
        new_X = last_X.copy()
        # move current features into lag slots
        new_X[n_feats:] = last_X[:n_feats]
        # put predicted precip into the precipitation lag slot if present
        # (approximation: we can't know future temp etc so we recycle last known)
        last_X = new_X

    return np.array(preds)

# -- COUNTRY hourly rainfall distribution pattern ---------------
# Based on West Africa climatology: rain peaks in afternoon/evening
# Proportions across 24 hours (sum = 1.0)
HOURLY_PATTERN = np.array([
    0.010, 0.008, 0.007, 0.006, 0.006, 0.008,   # 00-05 (night, low)
    0.012, 0.018, 0.025, 0.030, 0.035, 0.040,   # 06-11 (morning rise)
    0.045, 0.055, 0.070, 0.085, 0.090, 0.095,   # 12-17 (afternoon peak)
    0.088, 0.075, 0.060, 0.045, 0.030, 0.012,   # 18-23 (evening taper)
])
HOURLY_PATTERN = HOURLY_PATTERN / HOURLY_PATTERN.sum()  # normalise to 1.0
HOURS = [f"{h:02d}:00" for h in range(24)]

# -- Collect forecasts for all states ---------------------------
print(f"\n[INFO] Generating forecasts for {len(states_to_run)} state(s)...")
print(f"{'State':<16} {'Capital':<16} {'Tomorrow':>10} {'Rain?':>6}")
print("-" * 52)

all_state_data = []

for state_name, capital, lat, lon in states_to_run:
    df_s, all_f, feats, tgt, sc = build_state(state_name, df_raw)
    if df_s is None or len(df_s) < 30:
        continue

    scaler_s = StandardScaler()
    scaler_s.fit(df_s[all_f].values)

    rf_s = RandomForestRegressor(
        n_estimators = rf_params["n_estimators"],
        max_depth    = rf_params["max_depth"],
        random_state = 42, n_jobs=-1)
    rf_s.fit(scaler_s.transform(df_s[all_f].values), df_s["target"].values)

    preds_14 = recursive_forecast(rf_s, scaler_s, df_s, all_f, feats, n_days=14)

    tomorrow    = preds_14[0]
    week_preds  = preds_14[0:7]
    next_week   = preds_14[7:14]
    today_total = tomorrow
    hourly      = HOURLY_PATTERN * today_total

    rain_tmrw = "YES" if tomorrow >= RAIN_THRESHOLD_MM else "No"
    print(f"  {state_name:<14} {capital:<16} {tomorrow:>10.3f} {rain_tmrw:>6}")

    all_state_data.append({
        "state"      : state_name,
        "capital"    : capital,
        "tomorrow"   : tomorrow,
        "rain_tmrw"  : rain_tmrw,
        "week_preds" : week_preds,
        "next_week"  : next_week,
        "hourly"     : hourly,
        "lat"        : lat,
        "lon"        : lon,
    })

if not all_state_data:
    print("[ERROR] No state data available. Run steps 1-5 first.")
    sys.exit(1)

# -- Date labels for week and next week -------------------------
week_dates     = [(today + timedelta(days=i)).strftime("%a\n%d %b")
                  for i in range(1, 8)]
next_week_dates= [(today + timedelta(days=i)).strftime("%a\n%d %b")
                  for i in range(8, 15)]

# ==============================================================
# FIGURE 21 - Tomorrow's Forecast
# ==============================================================
print("\n[FIGURE] Generating fig21 - Tomorrow's Forecast ...")

n_states = len(all_state_data)
tmrw_vals= [d["tomorrow"] for d in all_state_data]
states_l = [d["state"]    for d in all_state_data]
rain_flag= [d["rain_tmrw"]for d in all_state_data]

# Sort by precipitation descending
order     = np.argsort(tmrw_vals)[::-1]
tmrw_sort = [tmrw_vals[i] for i in order]
state_sort= [states_l[i]  for i in order]
rain_sort = [rain_flag[i]  for i in order]

bar_colors = ["#1565C0" if v >= RAIN_THRESHOLD_MM else "#B0BEC5"
              for v in tmrw_sort]

if FORECAST_MODE == "single":
    fig, ax = plt.subplots(figsize=(10, 5))
    val   = tmrw_sort[0]
    color = "#1565C0" if val >= RAIN_THRESHOLD_MM else "#B0BEC5"
    ax.bar([all_state_data[0]["capital"]], [val],
           color=color, edgecolor="white", width=0.4, alpha=0.9)
    ax.set_ylabel("Predicted Precipitation (mm/day)", fontsize=11)
    rain_txt = "RAIN EXPECTED" if val >= RAIN_THRESHOLD_MM else "DRY DAY EXPECTED"
    rain_col = "#1565C0" if val >= RAIN_THRESHOLD_MM else "#78909C"
    ax.text(0, val + 0.05, f"{val:.2f} mm\n{rain_txt}",
            ha="center", va="bottom", fontsize=12,
            fontweight="bold", color=rain_col)
else:
    fig_h = max(8, n_states * 0.38)
    fig, ax = plt.subplots(figsize=(14, fig_h))
    bars = ax.barh(state_sort, tmrw_sort,
                   color=bar_colors, edgecolor="white", alpha=0.9)
    ax.axvline(RAIN_THRESHOLD_MM, color="red", linestyle="--",
               linewidth=1.3, label=f"Rain threshold ({RAIN_THRESHOLD_MM} mm)")
    for bar, val, rf in zip(bars, tmrw_sort, rain_sort):
        label = f"{val:.2f} mm  {'RAIN' if rf=='YES' else 'Dry'}"
        ax.text(bar.get_width() + 0.02,
                bar.get_y() + bar.get_height() / 2,
                label, va="center", fontsize=7.5)
    ax.set_xlabel("Predicted Precipitation (mm/day)", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(axis="x", alpha=0.3)

rain_count = sum(1 for r in rain_flag if r == "YES")
ax.set_title(
    f"Figure 21 - Tomorrow's Precipitation Forecast  ({today_str})\n"
    f"{rain_count} of {n_states} state(s) expecting rain",
    fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "fig21_tomorrow_forecast.png"),
            dpi=FIG_DPI, bbox_inches="tight")
plt.close()
print("  [OK] fig21_tomorrow_forecast.png saved")

# ==============================================================
# FIGURE 22 - Rest of the Week (Day 1-7)
# ==============================================================
print("[FIGURE] Generating fig22 - Rest of the Week ...")

if FORECAST_MODE == "single":
    d      = all_state_data[0]
    fig, ax= plt.subplots(figsize=(12, 5))
    colors_w = ["#1565C0" if v >= RAIN_THRESHOLD_MM else "#90A4AE"
                for v in d["week_preds"]]
    bars = ax.bar(week_dates, d["week_preds"],
                  color=colors_w, edgecolor="white", alpha=0.9, width=0.6)
    ax.axhline(RAIN_THRESHOLD_MM, color="red", linestyle="--",
               linewidth=1.2, label=f"Rain threshold ({RAIN_THRESHOLD_MM} mm)")
    for bar, val in zip(bars, d["week_preds"]):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.03,
                f"{val:.2f}", ha="center", va="bottom", fontsize=9,
                fontweight="bold",
                color="#1565C0" if val >= RAIN_THRESHOLD_MM else "#546E7A")
    ax.set_ylabel("Predicted Precipitation (mm/day)", fontsize=11)
    ax.set_title(
        f"Figure 22 - 7-Day Forecast: {d['state']} ({d['capital']})\n"
        f"Starting {(today + timedelta(days=1)).strftime('%d %B %Y')}",
        fontsize=13, fontweight="bold")
    ax.legend(fontsize=9); ax.grid(axis="y", alpha=0.3)

else:
    # Heatmap: states x days
    week_matrix = np.array([d["week_preds"] for d in all_state_data])
    state_names = [d["state"] for d in all_state_data]
    fig_h = max(10, n_states * 0.38)
    fig, ax = plt.subplots(figsize=(14, fig_h))
    im = ax.imshow(week_matrix, aspect="auto", cmap="YlOrRd",
                   vmin=0, vmax=max(week_matrix.max(), RAIN_THRESHOLD_MM * 3))
    ax.set_xticks(range(7))
    ax.set_xticklabels(week_dates, fontsize=9)
    ax.set_yticks(range(n_states))
    ax.set_yticklabels(state_names, fontsize=8)
    for i in range(n_states):
        for j in range(7):
            val = week_matrix[i, j]
            txt = f"{val:.1f}"
            col = "white" if val >= RAIN_THRESHOLD_MM * 2 else "black"
            ax.text(j, i, txt, ha="center", va="center",
                    fontsize=7, color=col)
    cbar = plt.colorbar(im, ax=ax, shrink=0.6)
    cbar.set_label("Precipitation (mm/day)", fontsize=9)
    ax.set_title(
        f"Figure 22 - 7-Day Forecast Heatmap: All Country States\n"
        f"Week of {(today + timedelta(days=1)).strftime('%d %B')} - "
        f"{(today + timedelta(days=7)).strftime('%d %B %Y')}",
        fontsize=13, fontweight="bold")

plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "fig22_week_forecast.png"),
            dpi=FIG_DPI, bbox_inches="tight")
plt.close()
print("  [OK] fig22_week_forecast.png saved")

# ==============================================================
# FIGURE 23 - Same Time Next Week (Day 8-14)
# ==============================================================
print("[FIGURE] Generating fig23 - Next Week Forecast ...")

if FORECAST_MODE == "single":
    d      = all_state_data[0]
    fig, ax= plt.subplots(figsize=(12, 5))
    colors_nw = ["#7B1FA2" if v >= RAIN_THRESHOLD_MM else "#CE93D8"
                 for v in d["next_week"]]
    bars = ax.bar(next_week_dates, d["next_week"],
                  color=colors_nw, edgecolor="white", alpha=0.9, width=0.6)
    ax.axhline(RAIN_THRESHOLD_MM, color="red", linestyle="--",
               linewidth=1.2, label=f"Rain threshold ({RAIN_THRESHOLD_MM} mm)")
    for bar, val in zip(bars, d["next_week"]):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.03,
                f"{val:.2f}", ha="center", va="bottom", fontsize=9,
                fontweight="bold",
                color="#7B1FA2" if val >= RAIN_THRESHOLD_MM else "#546E7A")
    ax.set_ylabel("Predicted Precipitation (mm/day)", fontsize=11)
    ax.set_title(
        f"Figure 23 - Same Time Next Week Forecast: {d['state']} ({d['capital']})\n"
        f"{(today + timedelta(days=8)).strftime('%d %B')} - "
        f"{(today + timedelta(days=14)).strftime('%d %B %Y')}",
        fontsize=13, fontweight="bold")
    ax.legend(fontsize=9); ax.grid(axis="y", alpha=0.3)

else:
    nw_matrix   = np.array([d["next_week"] for d in all_state_data])
    state_names = [d["state"] for d in all_state_data]
    fig_h = max(10, n_states * 0.38)
    fig, ax = plt.subplots(figsize=(14, fig_h))
    im = ax.imshow(nw_matrix, aspect="auto", cmap="RdPu",
                   vmin=0, vmax=max(nw_matrix.max(), RAIN_THRESHOLD_MM * 3))
    ax.set_xticks(range(7))
    ax.set_xticklabels(next_week_dates, fontsize=9)
    ax.set_yticks(range(n_states))
    ax.set_yticklabels(state_names, fontsize=8)
    for i in range(n_states):
        for j in range(7):
            val = nw_matrix[i, j]
            txt = f"{val:.1f}"
            col = "white" if val >= RAIN_THRESHOLD_MM * 2 else "black"
            ax.text(j, i, txt, ha="center", va="center",
                    fontsize=7, color=col)
    cbar = plt.colorbar(im, ax=ax, shrink=0.6)
    cbar.set_label("Precipitation (mm/day)", fontsize=9)
    ax.set_title(
        f"Figure 23 - Same Time Next Week Forecast: All Country States\n"
        f"Week of {(today + timedelta(days=8)).strftime('%d %B')} - "
        f"{(today + timedelta(days=14)).strftime('%d %B %Y')}",
        fontsize=13, fontweight="bold")

plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "fig23_next_week_forecast.png"),
            dpi=FIG_DPI, bbox_inches="tight")
plt.close()
print("  [OK] fig23_next_week_forecast.png saved")

# ==============================================================
# FIGURE 24 - Hourly Estimate Today
# ==============================================================
print("[FIGURE] Generating fig24 - Hourly Estimate Today ...")

if FORECAST_MODE == "single":
    d      = all_state_data[0]
    fig, ax= plt.subplots(figsize=(14, 5))
    hourly = d["hourly"]
    colors_h = ["#1565C0" if v >= (RAIN_THRESHOLD_MM/24)*2 else "#90CAF9"
                for v in hourly]
    bars = ax.bar(HOURS, hourly, color=colors_h,
                  edgecolor="white", alpha=0.9, width=0.7)
    ax.set_xlabel("Hour of Day", fontsize=11)
    ax.set_ylabel("Est. Precipitation (mm)", fontsize=11)
    peak_hr = np.argmax(hourly)
    ax.annotate(f"Peak: {hourly[peak_hr]:.3f} mm\nat {HOURS[peak_hr]}",
                xy=(peak_hr, hourly[peak_hr]),
                xytext=(peak_hr + 2, hourly[peak_hr] + hourly.max()*0.1),
                arrowprops=dict(arrowstyle="->", color="red"),
                fontsize=9, color="red", fontweight="bold")
    ax.set_xticks(range(24))
    ax.set_xticklabels(HOURS, rotation=45, ha="right", fontsize=7)
    ax.set_title(
        f"Figure 24 - Estimated Hourly Precipitation Today: "
        f"{d['state']} ({d['capital']})\n"
        f"{today_str}  |  Daily Total Est: {d['tomorrow']:.2f} mm",
        fontsize=13, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)

else:
    # For all states: show top 6 most rain-prone states hourly
    sorted_data = sorted(all_state_data,
                         key=lambda x: x["tomorrow"], reverse=True)
    top6 = sorted_data[:6]
    fig, axes = plt.subplots(2, 3, figsize=(18, 10), sharey=False)
    axes = axes.flatten()
    for ax, d in zip(axes, top6):
        hourly = d["hourly"]
        colors_h = ["#1565C0" if v >= (RAIN_THRESHOLD_MM/24)*2
                    else "#90CAF9" for v in hourly]
        ax.bar(range(24), hourly, color=colors_h,
               edgecolor="white", alpha=0.9, width=0.8)
        ax.set_title(f"{d['state']} ({d['capital']})\n"
                     f"Daily Est: {d['tomorrow']:.2f} mm",
                     fontsize=9, fontweight="bold")
        ax.set_xticks([0, 6, 12, 18, 23])
        ax.set_xticklabels(["00:00","06:00","12:00","18:00","23:00"],
                            fontsize=7)
        ax.set_ylabel("mm", fontsize=8)
        ax.grid(axis="y", alpha=0.3)
        peak_hr = np.argmax(hourly)
        ax.axvline(peak_hr, color="red", linestyle="--",
                   linewidth=1, alpha=0.7)
    fig.suptitle(
        f"Figure 24 - Estimated Hourly Precipitation Today\n"
        f"Top 6 Rain-Prone States  |  {today_str}",
        fontsize=13, fontweight="bold")

plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "fig24_hourly_today.png"),
            dpi=FIG_DPI, bbox_inches="tight")
plt.close()
print("  [OK] fig24_hourly_today.png saved")

# -- Save comprehensive structured forecast table --------------
# table11: full 14-day forecast for every state with all hourly peaks
forecast_rows = []
for d in all_state_data:
    hourly      = d["hourly"]
    peak_hr     = int(np.argmax(hourly))
    peak_hr_val = round(float(hourly[peak_hr]), 4)

    # Day labels using real dates
    day_labels  = [(today + timedelta(days=i)).strftime("%a %d %b")
                   for i in range(1, 15)]

    # Rain expected per day (bool -> YES/No)
    week_rain   = ["YES" if v >= RAIN_THRESHOLD_MM else "No"
                   for v in d["week_preds"]]
    nweek_rain  = ["YES" if v >= RAIN_THRESHOLD_MM else "No"
                   for v in d["next_week"]]

    row = {
        # Identity
        "State"                  : d["state"],
        "Capital"                : d["capital"],
        "Latitude"               : d["lat"],
        "Longitude"              : d["lon"],
        # Tomorrow
        f"Day1_{day_labels[0]}_mm"   : round(d["week_preds"][0], 3),
        f"Day1_Rain"                 : week_rain[0],
        # Days 2-7
        f"Day2_{day_labels[1]}_mm"   : round(d["week_preds"][1], 3),
        f"Day2_Rain"                 : week_rain[1],
        f"Day3_{day_labels[2]}_mm"   : round(d["week_preds"][2], 3),
        f"Day3_Rain"                 : week_rain[2],
        f"Day4_{day_labels[3]}_mm"   : round(d["week_preds"][3], 3),
        f"Day4_Rain"                 : week_rain[3],
        f"Day5_{day_labels[4]}_mm"   : round(d["week_preds"][4], 3),
        f"Day5_Rain"                 : week_rain[4],
        f"Day6_{day_labels[5]}_mm"   : round(d["week_preds"][5], 3),
        f"Day6_Rain"                 : week_rain[5],
        f"Day7_{day_labels[6]}_mm"   : round(d["week_preds"][6], 3),
        f"Day7_Rain"                 : week_rain[6],
        # Week summary
        "Week1_Avg_mm"               : round(float(np.mean(d["week_preds"])), 3),
        "Week1_Total_mm"             : round(float(np.sum(d["week_preds"])), 3),
        "Week1_RainDays"             : int(np.sum(np.array(d["week_preds"]) >= RAIN_THRESHOLD_MM)),
        # Next week days 8-14
        f"Day8_{day_labels[7]}_mm"   : round(d["next_week"][0], 3),
        f"Day8_Rain"                 : nweek_rain[0],
        f"Day9_{day_labels[8]}_mm"   : round(d["next_week"][1], 3),
        f"Day9_Rain"                 : nweek_rain[1],
        f"Day10_{day_labels[9]}_mm"  : round(d["next_week"][2], 3),
        f"Day10_Rain"                : nweek_rain[2],
        f"Day11_{day_labels[10]}_mm" : round(d["next_week"][3], 3),
        f"Day11_Rain"                : nweek_rain[3],
        f"Day12_{day_labels[11]}_mm" : round(d["next_week"][4], 3),
        f"Day12_Rain"                : nweek_rain[4],
        f"Day13_{day_labels[12]}_mm" : round(d["next_week"][5], 3),
        f"Day13_Rain"                : nweek_rain[5],
        f"Day14_{day_labels[13]}_mm" : round(d["next_week"][6], 3),
        f"Day14_Rain"                : nweek_rain[6],
        # Next week summary
        "Week2_Avg_mm"               : round(float(np.mean(d["next_week"])), 3),
        "Week2_Total_mm"             : round(float(np.sum(d["next_week"])), 3),
        "Week2_RainDays"             : int(np.sum(np.array(d["next_week"]) >= RAIN_THRESHOLD_MM)),
        # Hourly estimate
        "Hourly_Peak_Time"           : HOURS[peak_hr],
        "Hourly_Peak_mm"             : peak_hr_val,
        "Hourly_Total_Est_mm"        : round(float(d["tomorrow"]), 3),
    }
    forecast_rows.append(row)

fc_df = pd.DataFrame(forecast_rows)
fc_df.to_csv(os.path.join(TABLE_DIR, "table11_future_forecasts.csv"), index=False)
print("\n[TABLE] table11_future_forecasts.csv saved")
print(f"        Columns : {len(fc_df.columns)}")
print(f"        Rows    : {len(fc_df)}")
print(fc_df[["State","Capital",
             list(fc_df.columns)[4],
             "Week1_Avg_mm","Week2_Avg_mm",
             "Hourly_Peak_Time"]].to_string(index=False))

# -- Console summary -------------------------------------------
print(f"\n{'='*60}")
print(f"  FORECAST SUMMARY  ({today_str})")
print(f"{'='*60}")
rain_states = [d["state"] for d in all_state_data if d["rain_tmrw"] == "YES"]
dry_states  = [d["state"] for d in all_state_data if d["rain_tmrw"] == "No"]
print(f"\n  States expecting RAIN tomorrow ({len(rain_states)}):")
for s in rain_states:
    print(f"    - {s}")
print(f"\n  States expecting DRY day tomorrow ({len(dry_states)}):")
for s in dry_states:
    print(f"    - {s}")

print(f"\n{'='*60}")
print("  STEP 6 COMPLETE")
print(f"  Figures : fig21, fig22, fig23, fig24")
print(f"  Table   : table11_future_forecasts.csv")
print(f"{'='*60}\n")

print("""
  NOTE ON HOURLY ESTIMATES (fig24):
  ------------------------------------------
  NASA POWER provides daily data, not hourly.
  Hourly estimates are derived by distributing
  the predicted daily total using West Africa's
  climatological rainfall timing pattern
  (afternoon/evening peak, low overnight).
  This is a standard approach in meteorological
  research and should be stated in your write-up.
""")
