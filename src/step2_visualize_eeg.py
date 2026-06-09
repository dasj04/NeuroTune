"""
STEP 2 — Visualize the Raw EEG Signal
=======================================
This script loads the EEG data and plots it so you can SEE the brain signal.
You'll notice the wave looks different in relaxed vs active sections.

Run after step 1: python3 step2_visualize_eeg.py
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os

os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

# ── Load data ──────────────────────────────────────────────────────────────────
print("Loading DEAP EEG data...")
trial = 0
channel = 0
eeg_data = np.load("data/processed/eeg_data.npy")
eeg = eeg_data[trial, channel]

labels   = np.load("data/processed/labels.npy")

fs = 128

print("EEG shape:", eeg_data.shape)
print("Labels shape:", labels.shape)

n_samples = len(eeg)
duration  = len(eeg) / fs
time      = np.linspace(0, duration, n_samples)

print(f"  Duration      : {duration:.0f} seconds")
print(f"  Sampling rate : {fs} Hz")
print(f"  Total samples : {n_samples}\n")

# ── Plot 1: Full raw EEG signal ────────────────────────────────────────────────
fig, axes = plt.subplots(3, 1, figsize=(14, 10))
fig.suptitle("BCI Music System — EEG Signal Analysis", fontsize=14, fontweight="bold")

ax1 = axes[0]
ax1.plot(time, eeg, color="#2196F3", linewidth=0.6, alpha=0.85)
ax1.set_title("Raw EEG Signal (60 seconds)", fontsize=12)
ax1.set_xlabel("Time (seconds)")
ax1.set_ylabel("Amplitude (µV)")
ax1.set_xlim(0, duration)

# Highlight state regions
ax1.axvspan(0,  15, alpha=0.12, color="green",  label="Relaxed")
ax1.axvspan(15, 30, alpha=0.12, color="red",    label="Active")
ax1.axvspan(30, 45, alpha=0.12, color="green")
ax1.axvspan(45, 60, alpha=0.12, color="red")
ax1.legend(loc="upper right")
ax1.grid(True, alpha=0.3)

# ── Plot 2: Zoomed view — relaxed vs active side by side ─────────────────────
ax2 = axes[1]
# Show 2 seconds from relaxed (t=5 to 7) and 2 seconds from active (t=20 to 22)
relaxed_slice = eeg[5*fs : 7*fs]
active_slice  = eeg[20*fs : 22*fs]
t_short = np.linspace(0, 2, 2*fs)

ax2.plot(t_short, relaxed_slice, color="green", linewidth=1.2,
         label="Relaxed (alpha dominant)", alpha=0.9)
ax2.plot(t_short, active_slice,  color="red",   linewidth=1.2,
         label="Active (beta dominant)", alpha=0.9)
ax2.set_title("Zoomed Comparison: Relaxed vs Active (2 seconds each)", fontsize=12)
ax2.set_xlabel("Time (seconds)")
ax2.set_ylabel("Amplitude (µV)")
ax2.legend()
ax2.grid(True, alpha=0.3)

# ── Plot 3: Frequency Spectrum (FFT) — what frequencies are present? ──────────
ax3 = axes[2]

def compute_fft(signal, fs):
    n    = len(signal)
    fft  = np.abs(np.fft.rfft(signal)) / n
    freq = np.fft.rfftfreq(n, 1/fs)
    return freq, fft

freq_r, fft_r = compute_fft(relaxed_slice, fs)
freq_a, fft_a = compute_fft(active_slice,  fs)

ax3.plot(freq_r, fft_r, color="green", linewidth=1.5, label="Relaxed")
ax3.plot(freq_a, fft_a, color="red",   linewidth=1.5, label="Active")
ax3.axvspan(8,  13, alpha=0.15, color="blue",   label="Alpha band (8–13 Hz)")
ax3.axvspan(13, 30, alpha=0.15, color="orange", label="Beta band (13–30 Hz)")
ax3.set_xlim(0, 40)
ax3.set_title("Frequency Spectrum (FFT) — Alpha vs Beta Power", fontsize=12)
ax3.set_xlabel("Frequency (Hz)")
ax3.set_ylabel("Power")
ax3.legend()
ax3.grid(True, alpha=0.3)

plt.tight_layout()
os.makedirs("output", exist_ok=True)
plt.savefig("output/eeg_visualization.png", dpi=150, bbox_inches="tight")
print("✅ Plot saved to output/eeg_visualization.png")
plt.close()
print("\nNext → Run: python3 step3_extract_features.py")
