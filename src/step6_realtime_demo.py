"""
STEP 6 — Real-Time Pipeline Demo
==================================
This is the COMPLETE SYSTEM running end-to-end.

It replays the real DEAP EEG session (trial 0, channel Fp1) second-by-second:
  1. Slice a 2-second window from the real EEG recording
  2. Filter signal (bandpass)
  3. Extract alpha + beta power (FFT)
  4. Classify mental state (SVM)
  5. Control music playback

In a live system, replace the pre-loaded array with a headset stream read.

Run: python3 step6_realtime_demo.py
"""

import numpy as np
from scipy.signal import butter, filtfilt
from collections import deque
import time
import random
import os

os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

# ── Pygame audio setup ────────────────────────────────────────────────────────
try:
    import pygame
    pygame.mixer.init()
    PLAY_AUDIO = True
except ImportError:
    print("pygame not found — running without audio.")
    PLAY_AUDIO = False

CALM_FOLDER      = "music/calm"
ENERGETIC_FOLDER = "music/energetic"

# ── EEG Parameters ────────────────────────────────────────────────────────────
FS          = int(np.load("data/sampling_rate.npy")[0])   # 128 Hz from DEAP
WINDOW_SEC  = 2
WINDOW_SIZE = FS * WINDOW_SEC   # 256 samples
STEP_SIZE        = FS * 1   # 1-second step (matches step3 feature extraction)
VOTE_BUFFER_SIZE = 3        # majority vote over last N windows (~3 s)

# ── Load SVM model ────────────────────────────────────────────────────────────
import pickle as _pkl_rt
_svm_bundle = _pkl_rt.load(open("data/processed/deap_svm.pkl", "rb"))
_SVM_MODEL  = _svm_bundle["model"]
_SVM_SCALER = _svm_bundle["scaler"]


# ── DSP functions ─────────────────────────────────────────────────────────────

def bandpass_filter(signal, lowcut, highcut, fs, order=4):
    nyq  = fs / 2.0
    b, a = butter(order, [lowcut/nyq, highcut/nyq], btype="band")
    return filtfilt(b, a, signal)

def band_power(signal, lowcut, highcut, fs):
    filtered = bandpass_filter(signal, lowcut, highcut, fs)
    return np.mean(filtered ** 2)

def classify_state(alpha_power, beta_power):
    ratio = alpha_power / (beta_power + 1e-10)
    feat      = _SVM_SCALER.transform([[alpha_power, beta_power, ratio]])
    svm_pred  = _SVM_MODEL.predict(feat)[0]          # 0=relaxed, 1=active
    ratio_pred = 0 if ratio > 1 else 1               # alpha>beta → relaxed, beta>alpha → active
    # If SVM and ratio agree, use that result; otherwise trust the ratio rule
    pred  = svm_pred if svm_pred == ratio_pred else ratio_pred
    state = "RELAXED" if pred == 0 else "ACTIVE"
    return state, ratio


# ── Music catalog ─────────────────────────────────────────────────────────────

MUSIC = {
    "RELAXED": ["♫  Binaural Beats - 10Hz"],
    "ACTIVE":  ["⚡ Binaural Beats - 18Hz"],
}

ICONS = {"RELAXED": "🟢", "ACTIVE": "🔴"}


# ── Real-Time Pipeline ────────────────────────────────────────────────────────

def run_realtime_demo():
    # Load the real EEG session recorded in step 1 (DEAP s01, trial 0, ch 0)
    eeg_full = np.load("data/eeg_session.npy")
    duration_s = len(eeg_full) / FS

    print("=" * 65)
    print("  BCI Music System — Real-Time Pipeline Demo")
    print("=" * 65)
    print(f"  EEG source        : data/eeg_session.npy  (DEAP s01 trial 0)")
    print(f"  Sampling rate     : {FS} Hz")
    print(f"  Session duration  : {duration_s:.0f}s")
    print(f"  Window / step     : {WINDOW_SEC}s / 1s")
    print(f"  Vote buffer       : {VOTE_BUFFER_SIZE} windows (majority of last {VOTE_BUFFER_SIZE}s)")
    print("  Classifying real EEG second-by-second...")
    print("=" * 65 + "\n")

    current_music = None
    current_state = None        # last *voted* state used for music control
    vote_buffer   = deque(maxlen=VOTE_BUFFER_SIZE)
    alpha_history = []
    beta_history  = []
    state_history = []

    window_count  = 0
    SLEEP_TIME    = 1.0   # real-time pacing: 1 second per window

    # Slide through the recording exactly as step 3 does
    for i in range(0, len(eeg_full) - WINDOW_SIZE + 1, STEP_SIZE):
        window_count += 1
        t_center = (i + WINDOW_SIZE // 2) / FS   # time in seconds

        # ── STEP 1: Slice real EEG window ─────────────────────────────────
        eeg_window = eeg_full[i : i + WINDOW_SIZE]

        # ── STEP 2: Extract features ───────────────────────────────────────
        alpha = band_power(eeg_window, 8,  13, FS)
        beta  = band_power(eeg_window, 13, 30, FS)

        alpha_history.append(alpha)
        beta_history.append(beta)

        # ── STEP 3: Classify raw state ─────────────────────────────────────
        raw_state, ratio = classify_state(alpha, beta)
        state_history.append(raw_state)

        # ── STEP 4: Majority-vote buffer ───────────────────────────────────
        vote_buffer.append(raw_state)
        r_votes = vote_buffer.count("RELAXED")
        a_votes = vote_buffer.count("ACTIVE")
        voted_state = "RELAXED" if r_votes >= a_votes else "ACTIVE"
        vote_str = f"R={r_votes} A={a_votes}/{len(vote_buffer)}"

        # ── STEP 5: Control music on voted-state flip only ─────────────────
        raw_icon   = ICONS[raw_state]
        voted_icon = ICONS[voted_state]
        print(f"  t={t_center:>5.1f}s  raw={raw_icon}{raw_state:<8}  "
              f"voted={voted_icon}{voted_state:<8}  [{vote_str}]  "
              f"α={alpha:.4f}  β={beta:.4f}  ratio={ratio:.2f}")

        if voted_state != current_state:
            current_state = voted_state
            current_music = random.choice(MUSIC[voted_state])

            if PLAY_AUDIO:
                folder = CALM_FOLDER if voted_state == "RELAXED" else ENERGETIC_FOLDER
                tracks = [f for f in os.listdir(folder) if f.endswith(".mp3")]
                if tracks:
                    pygame.mixer.music.load(os.path.join(folder, random.choice(tracks)))
                    pygame.mixer.music.play(-1)

            print(f"           ➤  {current_music}\n")

        time.sleep(SLEEP_TIME)

    # ── Session summary ────────────────────────────────────────────────────────
    total_seconds = window_count
    print("\n" + "=" * 65)
    print("  Session Complete")
    print("=" * 65)
    print(f"  Windows processed     : {window_count}")
    print(f"  Session duration      : {total_seconds}s")

    relaxed_count = state_history.count("RELAXED")
    active_count  = state_history.count("ACTIVE")
    print(f"  Vote buffer size      : {VOTE_BUFFER_SIZE} windows")
    print(f"  Time relaxed (raw)    : {relaxed_count}s ({relaxed_count/total_seconds*100:.0f}%)")
    print(f"  Time active  (raw)    : {active_count}s ({active_count/total_seconds*100:.0f}%)")
    print(f"  Avg alpha power       : {np.mean(alpha_history):.6f}")
    print(f"  Avg beta power        : {np.mean(beta_history):.6f}")

    if PLAY_AUDIO:
        pygame.mixer.music.stop()
        pygame.mixer.quit()

    print(f"\n✅ Real-time demo complete!")
    print("\nNext → Run: python3 step7_full_report.py  (generate analysis report)")


if __name__ == "__main__":
    run_realtime_demo()
