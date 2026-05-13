"""
STEP 5 — Music Controller
==========================
This script simulates music playback control based on mental state.

In a real system, this would play actual MP3 files.
Here, it demonstrates the full control logic and shows what would play.

If you have pygame and MP3 files in music/calm/ and music/energetic/,
set PLAY_AUDIO = True below to actually play music.

Run: python3 step5_music_controller.py
"""

import numpy as np
import time
import os
import random

os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

PLAY_AUDIO = True    # Requires: pip install pygame + MP3 files in music/calm/ and music/energetic/

if PLAY_AUDIO:
    try:
        import pygame
        pygame.mixer.init()
    except ImportError:
        print("pygame not available. Running in simulation mode.")
        PLAY_AUDIO = False

# ── Auto-generate prerequisites if missing ────────────────────────────────────
if not os.path.exists("data/classified_states.npy"):
    import pickle
    from scipy.signal import butter, filtfilt as _filtfilt
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC

    if not os.path.exists("data/processed/eeg_data.npy"):
        _subj = pickle.load(open("data/raw/deap/s22.dat", "rb"), encoding="latin1")
        _eeg  = _subj["data"][:, :32, 384:]
        _lbl  = np.where(_subj["labels"][:, 1] >= 5, 1, 0)
        os.makedirs("data/processed", exist_ok=True)
        np.save("data/processed/eeg_data.npy", _eeg)
        np.save("data/processed/labels.npy",   _lbl)
        np.save("data/sampling_rate.npy", np.array([128.0]))
        np.save("data/eeg_session.npy",   _eeg[0, 0])

    if not all(os.path.exists(p) for p in [
            "data/processed/features.npy", "data/processed/feature_labels.npy",
            "data/times.npy", "data/alpha_powers.npy", "data/beta_powers.npy"]):
        _eeg_data = np.load("data/processed/eeg_data.npy")
        _labels   = np.load("data/processed/labels.npy")
        _fs_pre   = int(np.load("data/sampling_rate.npy")[0])

        def _band_power(sig, lo, hi):
            nyq = _fs_pre / 2.0
            b, a = butter(4, [lo/nyq, hi/nyq], btype="band")
            return np.mean(_filtfilt(b, a, sig) ** 2)

        _feats, _flabels, _t0, _a0, _b0 = [], [], [], [], []
        _ws = 2 * _fs_pre
        for _ti, _trial in enumerate(_eeg_data):
            _sig, _i = _trial[0], 0
            while _i + _ws <= len(_sig):
                _a = _band_power(_sig[_i:_i+_ws], 8, 13)
                _b = _band_power(_sig[_i:_i+_ws], 13, 30)
                _feats.append([_a, _b, _a / (_b + 1e-10)])
                _flabels.append(_labels[_ti])
                if _ti == 0:
                    _t0.append((_i + _ws / 2) / _fs_pre)
                    _a0.append(_a)
                    _b0.append(_b)
                _i += _fs_pre
        os.makedirs("data/processed", exist_ok=True)
        np.save("data/processed/features.npy",      np.array(_feats))
        np.save("data/processed/feature_labels.npy", np.array(_flabels))
        np.save("data/times.npy",        np.array(_t0))
        np.save("data/alpha_powers.npy", np.array(_a0))
        np.save("data/beta_powers.npy",  np.array(_b0))

    _X   = np.load("data/processed/features.npy")
    _y   = np.load("data/processed/feature_labels.npy")
    _Xtr, _Xte, _ytr, _yte = train_test_split(_X, _y, test_size=0.2, random_state=42)
    _sc  = StandardScaler()
    _clf = SVC(kernel="rbf")
    _clf.fit(_sc.fit_transform(_Xtr), _ytr)
    _t_arr  = np.load("data/times.npy")
    _a_arr  = np.load("data/alpha_powers.npy")
    _b_arr  = np.load("data/beta_powers.npy")
    _ratio  = _a_arr / (_b_arr + 1e-10)
    _raw    = _clf.predict(_sc.transform(np.column_stack([_a_arr, _b_arr, _ratio])))
    _k = 2
    _smoothed = np.array([
        1 if np.mean(_raw[max(0,i-_k):min(len(_raw),i+_k+1)]) >= 0.5 else 0
        for i in range(len(_raw))
    ])
    np.save("data/classified_states.npy", _smoothed)
    np.save("data/times.npy", _t_arr)

# ── Load classified states ─────────────────────────────────────────────────────
states = np.load("data/classified_states.npy")
times  = np.load("data/times.npy")
print(f"Loaded {len(states)} classified windows\n")

# ── Music catalog ──────────────────────────────────────────────────────────────
# In a real system, fill these folders with MP3 files
CALM_FOLDER      = "music/calm"
ENERGETIC_FOLDER = "music/energetic"

# Simulate a music catalog (in real system, use os.listdir on the folders)
CALM_TRACKS = [
    "Binaural Beats - 10Hz"
]

ENERGETIC_TRACKS = [
    "Binaural Beats - 18Hz"
]


# ── Music Controller Class ─────────────────────────────────────────────────────
class MusicController:
    def __init__(self):
        self.current_state = None
        self.current_track = None
        self.play_count    = {"relaxed": 0, "active": 0}
        self.transitions   = 0
        self.history       = []

    def get_random_track(self, state):
        if state == "relaxed":
            return random.choice(CALM_TRACKS)
        else:
            return random.choice(ENERGETIC_TRACKS)

    def play_music(self, state):
        """Switch music if mental state changes."""
        state_name = "relaxed" if state == 0 else "active"
        
        if state_name == self.current_state:
            return   # Same state — continue current track

        # State has changed → switch music
        self.transitions += 1
        old_state = self.current_state
        self.current_state = state_name
        self.current_track = self.get_random_track(state_name)
        self.play_count[state_name] += 1

        if PLAY_AUDIO:
            # Real playback logic
            folder = CALM_FOLDER if state_name == "relaxed" else ENERGETIC_FOLDER
            tracks = [f for f in os.listdir(folder) if f.endswith(".mp3")]
            if tracks:
                track_path = os.path.join(folder, random.choice(tracks))
                pygame.mixer.music.load(track_path)
                pygame.mixer.music.play(-1)   # -1 = loop

        icon   = "🟢" if state_name == "relaxed" else "🔴"
        symbol = "♫ " if state_name == "relaxed" else "⚡"
        print(f"{icon} State changed: {str(old_state).upper():>7} → {state_name.upper():<7} "
              f" {symbol} Now playing: {self.current_track}")
        
        self.history.append({
            "from": old_state,
            "to":   state_name,
            "track": self.current_track
        })

    def get_summary(self):
        total = len(states)
        relaxed_pct = np.mean(states == 0) * 100
        active_pct  = np.mean(states == 1) * 100
        return {
            "total_windows":   total,
            "relaxed_pct":     relaxed_pct,
            "active_pct":      active_pct,
            "transitions":     self.transitions,
            "tracks_relaxed":  self.play_count["relaxed"],
            "tracks_active":   self.play_count["active"],
        }


# ── Run the simulation ─────────────────────────────────────────────────────────
print("=" * 65)
print("  BCI Music Controller — Simulating Real-Time Playback")
print("=" * 65)
print(f"  🟢 RELAXED state → Plays calm, ambient music")
print(f"  🔴 ACTIVE  state → Plays energetic, upbeat music")
print("=" * 65 + "\n")

controller = MusicController()

# Process each classified window
prev_state = None
for i, (t, state) in enumerate(zip(times, states)):
    if state != prev_state or i == 0:
        controller.play_music(state)
        prev_state = state
    
    time.sleep(1)   # 1 second per window — keeps pygame alive to play music

print("\n" + "=" * 65)
print("  Session Complete — Summary Report")
print("=" * 65)

summary = controller.get_summary()
print(f"  Total windows analyzed   : {summary['total_windows']}")
print(f"  Time in RELAXED state    : {summary['relaxed_pct']:.1f}%")
print(f"  Time in ACTIVE state     : {summary['active_pct']:.1f}%")
print(f"  Music transitions        : {summary['transitions']}")
print(f"  Calm tracks played       : {summary['tracks_relaxed']}")
print(f"  Energetic tracks played  : {summary['tracks_active']}")

print("\n  Playback History:")
for h in controller.history:
    arrow = f"{str(h['from']).upper():>7} → {h['to'].upper():<7}"
    print(f"    {arrow}  ♫  {h['track']}")

if PLAY_AUDIO:
    pygame.mixer.music.stop()
    pygame.mixer.quit()

print("\n✅ Music controller simulation complete!")
print("Next → Run: python3 step6_realtime_demo.py  (full pipeline demo)")
