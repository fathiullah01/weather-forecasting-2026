# =============================================================
# install_dependencies.py
# Installs all Python libraries required for this project.
# Run this once before running anything else.
#
# HOW TO RUN:
#   python install_dependencies.py
# =============================================================

import subprocess, sys, importlib, os

print("\n" + "="*65)
print("  PROJECT DEPENDENCY INSTALLER")
print("  Kwara State University, Malete")
print("  Faculty of ICT - Department of Computer Science")
print("="*65 + "\n")

# -- Required packages -----------------------------------------
# Format: (pip_package_name, import_name, version, description)
REQUIRED = [
    ("pandas",           "pandas",        ">=1.3.0",  "Data loading and manipulation"),
    ("numpy",            "numpy",         ">=1.21.0", "Numerical computation"),
    ("matplotlib",       "matplotlib",    ">=3.4.0",  "Plotting and figures"),
    ("seaborn",          "seaborn",       ">=0.11.0", "Statistical visualisation"),
    ("scikit-learn",     "sklearn",       ">=0.24.0", "Random Forest, PSO evaluation, metrics"),
    ("pyswarms",         "pyswarms",      ">=1.3.0",  "Particle Swarm Optimisation"),
    ("statsmodels",      "statsmodels",   ">=0.12.0", "ARIMA baseline model"),
    ("joblib",           "joblib",        ">=1.0.0",  "Model saving and loading"),
    ("requests",         "requests",      ">=2.25.0", "NASA POWER API download"),
    ("tqdm",             "tqdm",          ">=4.60.0", "Download progress bars"),
]

# -- Check which are already installed -------------------------
print("[CHECK] Scanning installed packages ...\n")
to_install   = []
already_have = []

for pip_name, import_name, version, description in REQUIRED:
    try:
        importlib.import_module(import_name)
        already_have.append((pip_name, description))
        print(f"  [OK]     {pip_name:<20} already installed")
    except ImportError:
        to_install.append((pip_name, version, description))
        print(f"  [MISSING]{pip_name:<20} needs installing")

print()

# -- Install missing packages ----------------------------------
if not to_install:
    print("[OK] All packages already installed. Nothing to do.\n")
else:
    print(f"[INSTALL] Installing {len(to_install)} missing package(s) ...\n")
    failed = []

    for pip_name, version, description in to_install:
        package_spec = f"{pip_name}{version}"
        print(f"  Installing: {pip_name}  ({description})")
        print(f"  Command  : pip install \"{package_spec}\"")

        result = subprocess.run(
            [sys.executable, "-m", "pip", "install",
             package_spec, "--upgrade", "--quiet"],
            capture_output=False
        )

        if result.returncode == 0:
            print(f"  [OK] {pip_name} installed successfully.\n")
        else:
            # Try without version pin as fallback
            print(f"  [RETRY] Trying without version pin ...")
            result2 = subprocess.run(
                [sys.executable, "-m", "pip", "install",
                 pip_name, "--upgrade", "--quiet"],
                capture_output=False
            )
            if result2.returncode == 0:
                print(f"  [OK] {pip_name} installed (no version pin).\n")
            else:
                print(f"  [FAIL] Could not install {pip_name}.\n")
                failed.append(pip_name)

    # -- Final report ------------------------------------------
    print("="*65)
    if failed:
        print(f"  [WARNING] {len(failed)} package(s) failed to install:")
        for f in failed:
            print(f"    - {f}")
        print("\n  Try installing manually:")
        print(f"    pip install {' '.join(failed)}")
        print("="*65 + "\n")
        sys.exit(1)
    else:
        print(f"  [OK] All {len(to_install)} package(s) installed successfully.")
        print("="*65 + "\n")

# -- Final verification ----------------------------------------
print("[VERIFY] Verifying all imports ...\n")
all_ok  = True
for pip_name, import_name, version, description in REQUIRED:
    try:
        mod = importlib.import_module(import_name)
        ver = getattr(mod, "__version__", "unknown")
        print(f"  [OK] {import_name:<20} version {ver}")
    except ImportError:
        print(f"  [FAIL] {import_name:<20} COULD NOT BE IMPORTED")
        all_ok = False

print()
if all_ok:
    print("="*65)
    print("  ALL DEPENDENCIES VERIFIED. Project is ready to run.")
    print("  Next step: python main.py")
    print("="*65 + "\n")
else:
    print("="*65)
    print("  [ERROR] Some imports failed. See above for details.")
    print("="*65 + "\n")
    sys.exit(1)