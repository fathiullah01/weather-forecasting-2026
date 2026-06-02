# =============================================================
# run_all_scripts.py  -  Master Runner
# Kwara State University, Nigeria
# Faculty of ICT - Department of Computer Science
#
# HOW TO RUN:
#   python run_all_scripts.py
# =============================================================

import subprocess, sys, os, time, shutil, warnings
warnings.filterwarnings("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"

BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
SCRIPTS_DIR = os.path.join(BASE_DIR, "scripts")

# -- Folders to wipe and recreate on every run -----------------
CLEAN_DIRS = ["figures", "tables", "results", "models"]

print("\n" + "="*65)
print("  WEATHER FORECASTING - PSO + RANDOM FOREST")
print("  Kwara State University, Nigeria")
print("  Faculty of ICT - Department of Computer Science")
print("="*65)

print("\n[CLEANUP] Clearing output folders ...")
for folder in CLEAN_DIRS:
    folder_path = os.path.join(BASE_DIR, folder)
    if os.path.exists(folder_path):
        shutil.rmtree(folder_path)
        print(f"  x  Deleted  : {folder}/")
    os.makedirs(folder_path)
    print(f"  +  Recreated: {folder}/")
print("[CLEANUP] Done. Starting fresh.\n")

# -- Check dataset exists before running -----------------------
import importlib.util
spec = importlib.util.spec_from_file_location("config",
       os.path.join(BASE_DIR, "config.py"))
cfg  = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cfg)

if not os.path.exists(cfg.DATASET_FILE):
    print("[ERROR] Country weather dataset not found.")
    print(f"  Expected: {cfg.DATASET_FILE}")
    print("  Please run: python download_country_weather_data.py first.")
    print("  Then re-run: python run_all_scripts.py")
    sys.exit(1)
else:
    print(f"[OK] Dataset found: {cfg.DATASET_FILE}\n")

# -- Steps -----------------------------------------------------
STEPS = [
    ("step1_load_preprocess.py",    "STEP 1 - Data Loading & Preprocessing"),
    ("step2_pso_feature_selection.py","STEP 2 - PSO Feature Selection"),
    ("step3_pso_hp_tuning_train.py","STEP 3 - PSO HP Tuning + RF Training"),
    ("step4_evaluation.py",         "STEP 4 - Model Evaluation & Comparison"),
    ("step5_country_forecast.py",   "STEP 5 - Country State Forecasting Dashboard"),
    ("step6_future_forecast.py",    "STEP 6 - Future Weather Forecast Figures"),
]

total_start = time.time()
for script, description in STEPS:
    script_path = os.path.join(SCRIPTS_DIR, script)
    print(f"\n{'-'*65}")
    print(f"  Running: {description}")
    print(f"{'-'*65}")
    start  = time.time()
    result = subprocess.run([sys.executable, script_path],
                            capture_output=False, text=True)
    elapsed = time.time() - start
    if result.returncode != 0:
        print(f"\n[ERROR] {script} failed. Check output above.")
        sys.exit(1)
    print(f"\n  Completed in {elapsed:.1f}s")

total = time.time() - total_start
print("\n" + "="*65)
print("  ALL STEPS COMPLETE")
print(f"  Total runtime: {total:.0f}s")
print("  Outputs:")
print("    figures/ --> all charts (fig1-fig24))")
print("    tables/  --> all CSV tables (table1-table11))")
print("    results/ --> predictions and processed data")
print("    models/  --> trained model files (.pkl)")
print("="*65 + "\n")