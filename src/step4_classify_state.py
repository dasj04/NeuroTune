"""
STEP 4 — Mental State Classifier
==================================
Trains an SVM on the extracted EEG features (alpha power, beta power,
alpha/beta ratio) to classify each window as "relaxed" (0) or "active" (1).

After training, the model is applied to trial-0 data to produce a
classified state sequence used by steps 5 and 7.

Run: python3 step4_classify_state.py
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pickle
import os

os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

# ── Load features ──────────────────────────────────────────────────────────────
X = np.load("data/processed/features.npy")
y = np.load("data/processed/feature_labels.npy")
print(f"Loaded {len(X)} feature windows\n")

# ==========================================================
# Train / Test Split
# ==========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ==========================================================
# Standardize Features
# ==========================================================

scaler  = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)

# ==========================================================
# Train SVM
# ==========================================================

model = SVC(kernel="rbf", class_weight="balanced")  # handles imbalanced arousal labels
print("Training SVM classifier...")
model.fit(X_train, y_train)

# ==========================================================
# Evaluate
# ==========================================================

pred = model.predict(X_test)
acc  = accuracy_score(y_test, pred)
print(f"\nAccuracy: {acc*100:.2f}%\n")
print(classification_report(y_test, pred))

# ==========================================================
# Save model
# ==========================================================

os.makedirs("data/processed", exist_ok=True)
with open("data/processed/deap_svm.pkl", "wb") as f:
    pickle.dump({"model": model, "scaler": scaler}, f)
print("✅ Model saved to data/processed/deap_svm.pkl")

# ==========================================================
# Classify trial-0 session for demonstration
# ==========================================================

times        = np.load("data/times.npy")
alpha_powers = np.load("data/alpha_powers.npy")
beta_powers  = np.load("data/beta_powers.npy")

ratio  = alpha_powers / (beta_powers + 1e-10)
X0     = np.column_stack([alpha_powers, beta_powers, ratio])
X0_sc  = scaler.transform(X0)
states = model.predict(X0_sc)

# Majority-vote smoothing with a window of 3 samples
k = 1
smoothed_states = np.array([
    1 if np.mean(states[max(0, i-k):min(len(states), i+k+1)]) >= 0.5 else 0
    for i in range(len(states))
])

# ── Visualize classification results ──────────────────────────────────────────
os.makedirs("output", exist_ok=True)
fig, axes = plt.subplots(3, 1, figsize=(14, 10))
fig.suptitle("BCI Music System — Mental State Classification", fontsize=14, fontweight="bold")

# Plot 1: Alpha vs Beta power over time
ax1 = axes[0]
ax1.plot(times, alpha_powers, color="green", linewidth=1.5, label="Alpha power (relaxed)")
ax1.plot(times, beta_powers,  color="red",   linewidth=1.5, label="Beta power (active)")
ax1.fill_between(times, alpha_powers, beta_powers,
                 where=(alpha_powers > beta_powers),
                 alpha=0.2, color="green", label="Alpha dominant")
ax1.fill_between(times, alpha_powers, beta_powers,
                 where=(beta_powers >= alpha_powers),
                 alpha=0.2, color="red",   label="Beta dominant")
ax1.set_title("Alpha vs Beta Band Power Over Time")
ax1.set_ylabel("Power (µV²)")
ax1.legend(loc="upper right", fontsize=8)
ax1.grid(True, alpha=0.3)

# Plot 2: Classified states (smoothed)
ax2 = axes[1]
colors = ["#4CAF50" if s == 0 else "#F44336" for s in smoothed_states]
ax2.bar(times, [1]*len(times), width=1.0, color=colors, alpha=0.8)
ax2.set_title("Detected Mental State Over Time (SVM, Smoothed)")
ax2.set_ylabel("State")
ax2.set_yticks([])
relaxed_patch = mpatches.Patch(color="#4CAF50", label="RELAXED → Calm music")
active_patch  = mpatches.Patch(color="#F44336", label="ACTIVE  → Energetic music")
ax2.legend(handles=[relaxed_patch, active_patch], loc="upper right")
ax2.grid(True, alpha=0.3, axis="x")

# Plot 3: Alpha/Beta ratio
ax3 = axes[2]
ax3.plot(times, ratio, color="#9C27B0", linewidth=1.5, label="Alpha/Beta ratio")
ax3.axhline(y=1.0, color="black", linestyle="--", linewidth=1, label="Threshold (ratio=1)")
ax3.fill_between(times, ratio, 1, where=(ratio > 1), alpha=0.2, color="green", label="Relaxed zone")
ax3.fill_between(times, ratio, 1, where=(ratio < 1), alpha=0.2, color="red",   label="Active zone")
ax3.set_title("Alpha/Beta Ratio (>1 = Relaxed, <1 = Active)")
ax3.set_xlabel("Time (seconds)")
ax3.set_ylabel("Ratio")
ax3.legend(loc="upper right", fontsize=8)
ax3.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("output/classification_results.png", dpi=150, bbox_inches="tight")
print("\n✅ Classification plot saved to output/classification_results.png")

# Save classified states for use in music controller and report
np.save("data/classified_states.npy", smoothed_states)
np.save("data/times.npy", times)
print("✅ Classified states saved to data/classified_states.npy")
print("\nNext → Run: python3 step5_music_controller.py")
plt.show()
