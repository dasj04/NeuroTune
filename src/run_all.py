"""
run_all.py — Run the entire BCI Music System pipeline in one go.
Run from the project root: python3 src/run_all.py
"""
import subprocess, sys, os

# Resolve directories so scripts can be found regardless of cwd
SRC_DIR  = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SRC_DIR)

steps = [
    ("step1_load_deap.py",        "Step 1 — Loading DEAP EEG dataset"),
    ("step3_extract_features.py", "Step 3 — Extracting alpha/beta band features"),
    ("step4_classify_state.py",   "Step 4 — Training SVM & classifying mental states"),
    ("step6_realtime_demo.py",    "Step 6 — Real-time pipeline demo (EEG → music)"),
    ("step7_full_report.py",      "Step 7 — Generating full analysis report"),
]

os.makedirs(os.path.join(BASE_DIR, "data"),   exist_ok=True)
os.makedirs(os.path.join(BASE_DIR, "output"), exist_ok=True)

for script, label in steps:
    print(f"\n{'='*60}")
    print(f"  ▶  {label}  ({script})")
    print(f"{'='*60}")
    result = subprocess.run(
        [sys.executable, os.path.join(SRC_DIR, script)],
        cwd=BASE_DIR,
        capture_output=False
    )
    if result.returncode != 0:
        print(f"❌ Error in {script}. Stopping.")
        sys.exit(1)

print("\n" + "="*60)
print("  ✅  All steps completed successfully!")
print("  📁  Check the output/ folder for your visualizations.")
print("="*60)
