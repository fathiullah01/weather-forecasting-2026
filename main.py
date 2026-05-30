# =============================================================
# main.py  -  Project Launcher
# Kwara State University, Malete
# Faculty of ICT - Department of Computer Science
#
# HOW TO RUN:
#   python main.py
#
# This script will:
#   1. Install all required Python dependencies
#   2. Download Nigeria weather data (skips if already exists)
#   3. Run the full project pipeline (steps 1-6)
# =============================================================

import subprocess, sys, os, importlib.util

# -- Load config -----------------------------------------------
_cfg_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.py")
spec = importlib.util.spec_from_file_location("config", _cfg_path)
cfg  = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cfg)

BASE_DIR = cfg.BASE_DIR
DATASET  = cfg.DATASET_FILE

# -- Scripts to run in order -----------------------------------
SCRIPTS = [
    ("install_dependencies.py",  "Installing Dependencies"),
    ("download_weather_data.py", "Downloading Weather Data"),
    ("run_all_scripts.py",       "Running Full Project Pipeline"),
]

for script, label in SCRIPTS:
    path = os.path.join(BASE_DIR, script)

    # Skip download if dataset already exists
    if script == "download_weather_data.py" and os.path.exists(DATASET):
        print(f"\n{'='*65}")
        print(f"  [SKIP] Dataset already exists. Skipping download.")
        print(f"{'='*65}\n")
        continue

    print(f"\n{'='*65}")
    print(f"  {label}")
    print(f"  Running: {script}")
    print(f"{'='*65}\n")

    result = subprocess.run([sys.executable, path])

    if result.returncode != 0:
        print(f"\n[ERROR] {script} failed. Fix the error above and retry.")
        sys.exit(1)

print("\n" + "="*65)
print("  ALL DONE. Check figures/, tables/, results/ for outputs.")
print("="*65 + "\n")