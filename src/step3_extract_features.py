"""
STEP 3 — Feature Extraction
=============================
This is the BRAIN of the system.

We take the raw EEG signal and extract two key numbers per window:
  - Alpha power  (8–13 Hz)  → how relaxed you are
  - Beta power   (13–30 Hz) → how alert/focused you are

We process the signal in 2-second windows (epochs).
Each window gives us one (alpha_power, beta_power) pair → one classification.

Run: python3 step3_extract_features.py
"""

import numpy as np
from scipy.signal import butter, filtfilt
import os

os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

# ── Auto-generate prerequisites if processed data is missing ───────────────────
if not os.path.exists("data/processed/eeg_data.npy"):
    import pickle
    _subj = pickle.load(open("data/raw/deap/s22.dat", "rb"), encoding="latin1")
    _eeg  = _subj["data"][:, :32, 384:]
    _lbl  = np.where(_subj["labels"][:, 1] >= 5, 1, 0)
    os.makedirs("data/processed", exist_ok=True)
    np.save("data/processed/eeg_data.npy", _eeg)
    np.save("data/processed/labels.npy",   _lbl)
    np.save("data/sampling_rate.npy", np.array([128.0]))
    np.save("data/eeg_session.npy",   _eeg[0, 0])

# ── Load EEG data ──────────────────────────────────────────────────────────────
eeg_data = np.load("data/processed/eeg_data.npy")
labels = np.load("data/processed/labels.npy")
fs  = int(np.load("data/sampling_rate.npy")[0])
print(f"Loaded EEG: shape={eeg_data.shape} @ {fs} Hz\n")


# ── Signal Processing Functions ────────────────────────────────────────────────

def bandpass_filter(signal, lowcut, highcut, fs, order=4):
    """
    Apply a bandpass filter to isolate a frequency range.
    
    Example: bandpass_filter(eeg, 8, 13, 256) 
             → keeps only alpha frequencies (8–13 Hz), removes everything else
    
    This is like using an equalizer to isolate one instrument in a song.
    """
    nyq  = fs / 2.0                         # Nyquist frequency
    low  = lowcut  / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype="band")
    return filtfilt(b, a, signal)           # Zero-phase filtering (no time shift)


def band_power(signal, fs, lowcut, highcut):
    """
    Compute the average power in a frequency band using FFT.
    
    Power = how much energy exists in that frequency range.
    High alpha power → relaxed brain.
    High beta power  → active/focused brain.
    """
    filtered = bandpass_filter(signal, lowcut, highcut, fs)
    power    = np.mean(filtered ** 2)       # Mean squared amplitude = power
    return power


def extract_features_windowed(eeg, fs, window_sec=2, step_sec=1):
    """
    Slide a window across the EEG and extract alpha + beta power per window.
    
    window_sec : length of each window in seconds
    step_sec   : how much to advance each step (overlap = window - step)
    
    Returns:
        times         : center time of each window
        alpha_powers  : alpha band power per window
        beta_powers   : beta band power per window
    """
    window_size = window_sec * fs
    step_size   = step_sec   * fs
    
    times        = []
    alpha_powers = []
    beta_powers  = []

    i = 0
    while i + window_size <= len(eeg):
        window = eeg[i : i + window_size]
        
        alpha = band_power(window, fs, 8,  13)   # Alpha: 8–13 Hz
        beta  = band_power(window, fs, 13, 30)   # Beta:  13–30 Hz
        
        center_time = (i + window_size / 2) / fs
        times.append(center_time)
        alpha_powers.append(alpha)
        beta_powers.append(beta)
        
        i += step_size
    
    return (np.array(times),
            np.array(alpha_powers),
            np.array(beta_powers))


# ── Run Feature Extraction ─────────────────────────────────────────────────────
print("Extracting features (alpha & beta power) with 2-second sliding windows...")
all_features = []
all_labels = []

for trial_idx in range(len(eeg_data)):

    eeg_trial = eeg_data[trial_idx]

    label = labels[trial_idx]

    # Use one EEG channel initially
    signal = eeg_trial[0]

    times, alpha_powers, beta_powers = extract_features_windowed(
        signal,
        fs,
        window_sec=2,
        step_sec=1
    )

    ratio = alpha_powers / (beta_powers + 1e-10)

    for a, b, r in zip(alpha_powers, beta_powers, ratio):

        all_features.append([a, b, r])

        all_labels.append(label)

all_features = np.array(all_features)
all_labels = np.array(all_labels)
print(f"  Windows extracted : {len(times)}")
print(f"  Time range        : {times[0]:.1f}s → {times[-1]:.1f}s\n")

# ── Print a sample ─────────────────────────────────────────────────────────────
print(f"{'Time (s)':>10} | {'Alpha Power':>12} | {'Beta Power':>12} | {'Dominant':>10}")
print("-" * 55)
for i in range(0, len(times), 5):
    dominant = "ALPHA (relax)" if alpha_powers[i] > beta_powers[i] else "BETA (active)"
    print(f"{times[i]:>10.1f} | {alpha_powers[i]:>12.4f} | {beta_powers[i]:>12.4f} | {dominant:>10}")

# ── Save features ──────────────────────────────────────────────────────────────
os.makedirs("data/processed", exist_ok=True)
np.save("data/processed/features.npy", all_features)
np.save("data/processed/feature_labels.npy", all_labels)

# Save single-trial (trial 0, channel 0) time-series for report & visualization
t0, a0, b0 = extract_features_windowed(eeg_data[0, 0], fs)
np.save("data/times.npy",        t0)
np.save("data/alpha_powers.npy", a0)
np.save("data/beta_powers.npy",  b0)

print("\n✅ Features saved to data/processed/")
print("✅ Single-trial times/powers saved to data/")
print("Next → Run: python3 step4_classify_state.py")
