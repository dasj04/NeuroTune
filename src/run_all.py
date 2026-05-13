"""
run_all.py — Run the entire BCI Music System pipeline in one go.
Run from the project root: python3 src/run_all.py
"""
import subprocess, sys, os

# Resolve directories so scripts can be found regardless of cwd
SRC_DIR  = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SRC_DIR)

steps = [
    ("step1_load_deap.py",        "Loading DEAP data"),
    ("step3_extract_features.py", "Extracting features"),
    ("step4_classify_state.py",   "Classifying mental states"),
    ("step5_music_controller.py", "Running music controller"),
    ("step6_realtime_demo.py",    "Real-time pipeline demo"),
    ("step7_full_report.py",      "Generating final report"),
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
