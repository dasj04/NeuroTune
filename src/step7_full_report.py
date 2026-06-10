"""
STEP 7 — Final Analysis Report & Visualization
===============================================
Generates a comprehensive multi-panel report showing the entire
BCI pipeline results in one figure. Use this for your project submission.

Run: python3 step7_full_report.py
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import os
from scipy.signal import butter, filtfilt
from scipy.fft import rfft, rfftfreq

os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

# ── Load all saved data ────────────────────────────────────────────────────────
eeg          = np.load("data/eeg_session.npy")
fs           = int(np.load("data/sampling_rate.npy")[0])
times        = np.load("data/times.npy")
alpha_powers = np.load("data/alpha_powers.npy")
beta_powers  = np.load("data/beta_powers.npy")
states       = np.load("data/classified_states.npy")
time_axis    = np.linspace(0, len(eeg)/fs, len(eeg))

os.makedirs("output", exist_ok=True)

def bandpass_filter(sig, low, high, fs, order=4):
    nyq  = fs / 2
    b, a = butter(order, [low/nyq, high/nyq], btype="band")
    return filtfilt(b, a, sig)

# ── Setup figure ───────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(16, 18))
fig.patch.set_facecolor("#FAFAFA")
gs  = gridspec.GridSpec(5, 2, figure=fig, hspace=0.45, wspace=0.35)

title = fig.suptitle(
    "BCI-Controlled Music System — Complete Analysis Report\n"
    "230103002 – Jyotirmoy Das  |  230103032 – Gyan Ankur Das\n"
    "Department of Information Technology, Gauhati University",
    fontsize=13, fontweight="bold", y=0.98, linespacing=1.6
)

COLOR = {"alpha": "#4CAF50", "beta": "#F44336",
         "relaxed": "#81C784", "active": "#E57373",
         "raw": "#1E88E5", "ratio": "#9C27B0"}

# ── Panel 1: Raw EEG signal ───────────────────────────────────────────────────
ax1 = fig.add_subplot(gs[0, :])
ax1.plot(time_axis, eeg, color=COLOR["raw"], linewidth=0.5, alpha=0.8)
ax1.axvspan(0,  15, alpha=0.1, color="green")
ax1.axvspan(15, 30, alpha=0.1, color="red")
ax1.axvspan(30, 45, alpha=0.1, color="green")
ax1.axvspan(45, 60, alpha=0.1, color="red")
ax1.set_title("① Raw EEG Signal (DEAP s01.dat — 60 Seconds)", fontweight="bold")
ax1.set_xlabel("Time (s)")
ax1.set_ylabel("Amplitude (µV)")
ax1.set_xlim(0, 60)
ax1.grid(True, alpha=0.25)
for x, lbl, c in [(7.5, "RELAXED", "green"), (22.5, "ACTIVE", "red"),
                  (37.5, "RELAXED", "green"), (52.5, "ACTIVE", "red")]:
    ax1.text(x, ax1.get_ylim()[1]*0.82, lbl,
             ha="center", fontsize=8, color=c, fontweight="bold", alpha=0.8)

# ── Panel 2: Filtered alpha signal ───────────────────────────────────────────
ax2 = fig.add_subplot(gs[1, 0])
alpha_filtered = bandpass_filter(eeg[3*fs:8*fs], 8, 13, fs)
beta_filtered  = bandpass_filter(eeg[3*fs:8*fs], 13, 30, fs)
t5 = np.linspace(3, 8, 5*fs)
ax2.plot(t5, alpha_filtered, color=COLOR["alpha"], linewidth=1.2, label="Alpha (8–13 Hz)")
ax2.plot(t5, beta_filtered,  color=COLOR["beta"],  linewidth=1.2, label="Beta (13–30 Hz)")
ax2.set_title("② Bandpass Filtered — Relaxed Segment", fontweight="bold")
ax2.set_xlabel("Time (s)")
ax2.set_ylabel("Amplitude (µV)")
ax2.legend(fontsize=8)
ax2.grid(True, alpha=0.25)

# ── Panel 3: FFT frequency spectrum ──────────────────────────────────────────
ax3 = fig.add_subplot(gs[1, 1])
seg_relax = eeg[5*fs:7*fs]
seg_activ = eeg[20*fs:22*fs]
for seg, label, color in [(seg_relax, "Relaxed (0–15s)", "green"),
                           (seg_activ, "Active (15–30s)", "red")]:
    N     = len(seg)
    freqs = rfftfreq(N, 1/fs)
    power = np.abs(rfft(seg)) / N
    ax3.plot(freqs, power, color=color, linewidth=1.3, label=label)
ax3.axvspan(8,  13, alpha=0.15, color="green",  label="Alpha band")
ax3.axvspan(13, 30, alpha=0.15, color="orange", label="Beta band")
ax3.set_xlim(0, 40)
ax3.set_title("③ Frequency Spectrum (FFT)", fontweight="bold")
ax3.set_xlabel("Frequency (Hz)")
ax3.set_ylabel("Power")
ax3.legend(fontsize=8)
ax3.grid(True, alpha=0.25)

# ── Panel 4: Alpha & beta power over time ────────────────────────────────────
ax4 = fig.add_subplot(gs[2, :])
ax4.plot(times, alpha_powers, color=COLOR["alpha"], linewidth=1.8, label="Alpha power")
ax4.plot(times, beta_powers,  color=COLOR["beta"],  linewidth=1.8, label="Beta power")
ax4.fill_between(times, alpha_powers, beta_powers,
                 where=(alpha_powers > beta_powers), alpha=0.2, color="green")
ax4.fill_between(times, alpha_powers, beta_powers,
                 where=(beta_powers >= alpha_powers), alpha=0.2, color="red")
ax4.set_title("④ Alpha vs Beta Band Power — Feature Extraction", fontweight="bold")
ax4.set_xlabel("Time (s)")
ax4.set_ylabel("Power (µV²)")
ax4.legend(fontsize=9)
ax4.grid(True, alpha=0.25)
ax4.set_xlim(times[0], times[-1])

# ── Panel 5: Alpha/Beta ratio ─────────────────────────────────────────────────
ax5 = fig.add_subplot(gs[3, 0])
ratio = alpha_powers / (beta_powers + 1e-10)
ax5.plot(times, ratio, color=COLOR["ratio"], linewidth=1.5, label="α/β ratio")
ax5.axhline(1.0, color="black", linestyle="--", linewidth=1, label="Threshold")
ax5.fill_between(times, ratio, 1, where=(ratio > 1), alpha=0.2, color="green")
ax5.fill_between(times, ratio, 1, where=(ratio < 1), alpha=0.2, color="red")
ax5.set_title("⑤ Alpha/Beta Ratio", fontweight="bold")
ax5.set_xlabel("Time (s)")
ax5.set_ylabel("Ratio")
ax5.legend(fontsize=8)
ax5.grid(True, alpha=0.25)
ax5.set_xlim(times[0], times[-1])

# ── Panel 6: Classified states ────────────────────────────────────────────────
ax6 = fig.add_subplot(gs[3, 1])
colors_bar = [COLOR["relaxed"] if s == 0 else COLOR["active"] for s in states]
ax6.bar(times, [1]*len(times), width=1.0, color=colors_bar)
ax6.set_title("⑥ Mental State Classification", fontweight="bold")
ax6.set_xlabel("Time (s)")
ax6.set_yticks([])
r_patch = mpatches.Patch(color=COLOR["relaxed"], label="RELAXED")
a_patch = mpatches.Patch(color=COLOR["active"],  label="ACTIVE")
ax6.legend(handles=[r_patch, a_patch], fontsize=8, loc="upper right")
ax6.grid(True, alpha=0.25, axis="x")
ax6.set_xlim(times[0], times[-1])

# ── Panel 7: Music output summary ─────────────────────────────────────────────
ax7 = fig.add_subplot(gs[4, :])
ax7.axis("off")

relaxed_pct = np.mean(states == 0) * 100
active_pct  = np.mean(states == 1) * 100
transitions = np.sum(np.diff(states) != 0)

summary_text = (
    f"⑦  MUSIC CONTROL OUTPUT SUMMARY\n\n"
    f"  Session Duration : 60 seconds     |  "
    f"Sampling Rate : 128 Hz     |  "
    f"Window Size : 2 seconds     |  "
    f"Windows Analyzed : {len(states)}\n\n"
    f"  🟢  RELAXED state detected : {relaxed_pct:.1f}% of session  →  "
    f"Plays: Calm / Ambient / Lo-fi music\n"
    f"  🔴  ACTIVE  state detected : {active_pct:.1f}% of session  →  "
    f"Plays: Energetic / Upbeat / Focus music\n\n"
    f"  Music transitions triggered : {transitions}   |   "
    f"Classification method : SVM (RBF kernel) + majority-vote smoothing\n\n"
    f"  Real-Time Headset Support : NeuroSky MindWave (mindwave-python)  |  "
    f"Muse (muselsl)  |  OpenBCI"
)

ax7.text(0.02, 0.95, summary_text, transform=ax7.transAxes,
         fontsize=10.5, verticalalignment="top",
         bbox=dict(boxstyle="round,pad=0.6", facecolor="#E8F5E9",
                   edgecolor="#4CAF50", linewidth=1.5),
         fontfamily="monospace")

plt.savefig("output/full_report.png", dpi=150, bbox_inches="tight",
            facecolor="#FAFAFA")
print("✅ Full report saved to output/full_report.png")
plt.show()
print("\n🎉 Project complete! All output files are in the output/ folder.")
