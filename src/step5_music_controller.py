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
