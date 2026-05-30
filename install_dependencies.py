# =============================================================
# install_dependencies.py
# Installs all Python libraries required for this project.
# =============================================================

import subprocess, sys

print("\n" + "="*65)
print("  PROJECT DEPENDENCY INSTALLER")
print("  Kwara State University, Malete")
print("  Faculty of ICT - Department of Computer Science")
print("="*65 + "\n")

# -- Step 1: Check pip -----------------------------------------
print("[SETUP] Checking pip ...")
result = subprocess.run(
    [sys.executable, "-m", "pip", "--version"],
    capture_output=True, text=True)

if result.returncode != 0:
    print("[SETUP] pip not found. Installing pip ...")
    bootstrap = subprocess.run(
        [sys.executable, "-m", "ensurepip", "--upgrade"],
        capture_output=False)
    if bootstrap.returncode != 0:
        print("[ERROR] Could not install pip.")
        print("  Visit: https://pip.pypa.io/en/stable/installation/")
        sys.exit(1)
    print("[OK] pip installed.\n")
else:
    print(f"[OK] {result.stdout.strip()}\n")

# -- Step 2: Upgrade pip first ---------------------------------
print("[SETUP] Upgrading pip to latest version ...")
subprocess.run(
    [sys.executable, "-m", "pip", "install",
     "--upgrade", "pip",
     "--trusted-host", "pypi.org",
     "--trusted-host", "files.pythonhosted.org"],
    capture_output=False)
print()

# -- Step 3: Install packages ----------------------------------
print("[INSTALL] Installing required packages ...")
print("          (this may take a few minutes)\n")

PACKAGES = [
    "pandas",
    "numpy",
    "matplotlib",
    "seaborn",
    "scikit-learn",
    "pyswarms",
    "statsmodels",
    "joblib",
    "requests",
    "tqdm",
]

result = subprocess.run(
    [sys.executable, "-m", "pip", "install"] + PACKAGES + [
        "--trusted-host", "pypi.org",
        "--trusted-host", "files.pythonhosted.org",
        "--prefer-binary",   # use pre-built wheels, avoids build errors
        "--upgrade",
    ],
    capture_output=False
)

if result.returncode != 0:
    print("\n[RETRY] Standard install failed.")
    print("[RETRY] Trying with --only-binary and pre-release fallback ...\n")

    result2 = subprocess.run(
        [sys.executable, "-m", "pip", "install"] + PACKAGES + [
            "--trusted-host", "pypi.org",
            "--trusted-host", "files.pythonhosted.org",
            "--prefer-binary",
            "--pre",           # allow pre-release builds (helps with Python 3.14)
            "--upgrade",
        ],
        capture_output=False
    )

    if result2.returncode != 0:
        print("\n[ERROR] Installation failed.")
        print("  Your Python version may be too new for some packages.")
        print(f"  Current Python: {sys.version}")
        print("  Recommended  : Python 3.10, 3.11, or 3.12")
        print("  Download from: https://www.python.org/downloads/")
        sys.exit(1)

print("\n" + "="*65)
print("  ALL DEPENDENCIES INSTALLED SUCCESSFULLY.")
print("  Next step: python main.py")
print("="*65 + "\n")