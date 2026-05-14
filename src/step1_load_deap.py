import pickle
import numpy as np
import os

FILE_PATH = "data/raw/deap/s01.dat"

os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

print("Loading DEAP dataset..."+ FILE_PATH + "\n")

# Load subject
with open(FILE_PATH, "rb") as f:
    subject = pickle.load(f, encoding="latin1")

# Extract EEG data + labels
data = subject["data"]
labels = subject["labels"]

print("Original data shape:", data.shape)
print("Labels shape:", labels.shape)

# ==========================================================
# KEEP ONLY EEG CHANNELS
# DEAP:
# first 32 channels = EEG
# remaining = peripheral signals
# ==========================================================

eeg_data = data[:, :32, :]

print("\nEEG-only shape:", eeg_data.shape)

# ==========================================================
# REMOVE BASELINE
# First 3 seconds = baseline
# 128 Hz × 3 = 384 samples
# ==========================================================

eeg_data = eeg_data[:, :, 384:]

print("After baseline removal:", eeg_data.shape)

# ==========================================================
# CREATE RELAXED / ACTIVE LABELS
# Using AROUSAL score
#
# labels[:,1]
#
# arousal < 5  → relaxed
# arousal >= 5 → active
# ==========================================================

arousal = labels[:, 1]

binary_labels = np.where(arousal >= 5, 1, 0)

print("\nBinary labels:")
print(binary_labels)

# ==========================================================
# SAVE FOR NEXT STEPS
# ==========================================================

os.makedirs("data/processed", exist_ok=True)
os.makedirs("output", exist_ok=True)

np.save("data/processed/eeg_data.npy", eeg_data)
np.save("data/processed/labels.npy", binary_labels)

# Save sampling rate and a single-channel session signal for later steps
np.save("data/sampling_rate.npy", np.array([128.0]))
np.save("data/eeg_session.npy", eeg_data[0, 0])   # trial 0, channel 0

print("\n✅ DEAP EEG data saved")
print("Next → Run: python3 step2_visualize_eeg.py")