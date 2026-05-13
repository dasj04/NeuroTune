# BCI-Controlled Music System Using EEG Signals
**230103002 – Jyotirmoy Das | 230103032 – Gyan Ankur Das**  
Department of Information Technology, Gauhati University

---

## Project Structure

```
bci_music_system/
├── step1_generate_eeg.py       # Simulate EEG data (or connect headset)
├── step2_visualize_eeg.py      # Plot raw EEG signal
├── step3_extract_features.py   # Extract alpha & beta band power
├── step4_classify_state.py     # Classify mental state (rule-based + SVM)
├── step5_music_controller.py   # Music playback controller
├── step6_realtime_demo.py      # Full real-time pipeline simulation
├── step7_full_report.py        # Generate final analysis report
├── run_all.py                  # Run everything in one command
├── music/
│   ├── calm/                   # Put .mp3 calm tracks here
│   └── energetic/              # Put .mp3 energetic tracks here
├── data/                       # Auto-generated data files
└── output/                     # Charts and reports saved here
```

---

## How to Run

### Install dependencies
```bash
pip install numpy scipy matplotlib scikit-learn pygame
```

### Run everything at once
```bash
python3 run_all.py
```

### Or run step by step
```bash
python3 step1_generate_eeg.py       # Generate EEG data
python3 step2_visualize_eeg.py      # Visualize raw signal
python3 step3_extract_features.py   # Extract features
python3 step4_classify_state.py     # Classify states
python3 step5_music_controller.py   # Music controller
python3 step6_realtime_demo.py      # Real-time demo
python3 step7_full_report.py        # Final report
```

---

## How It Works

```
EEG Headset → Raw Signal → Bandpass Filter → FFT → Band Power
                                                        │
                                              alpha power / beta power
                                                        │
                                            alpha > beta? → RELAXED → Calm Music
                                            beta > alpha? → ACTIVE  → Energetic Music
```

### Key Signal Processing Steps

| Step | Technique | Purpose |
|------|-----------|---------|
| Filtering | Butterworth Bandpass Filter | Isolate alpha (8–13 Hz) and beta (13–30 Hz) |
| Feature Extraction | Fast Fourier Transform (FFT) | Compute frequency band power |
| Classification | Alpha/Beta ratio + SVM | Determine mental state |
| Smoothing | Majority vote (5 windows) | Prevent rapid state switching |

---

## Results

- **SVM Classifier Accuracy: 98.3%** (5-fold cross-validation)
- Rule-based and ML classifiers agree **89.8%** of the time
- Correctly detects RELAXED and ACTIVE states with clear separation

---

## To Use With a Real Headset

Replace the `generate_window()` function in `step6_realtime_demo.py` with:

**NeuroSky MindWave:**
```bash
pip install mindwave-python
```
```python
import mindwave
headset = mindwave.Headset('/dev/tty.MindWave')
data = headset.raw_value   # raw EEG sample
```

**Muse Headset:**
```bash
pip install muselsl
```
```python
from muselsl import stream, list_muses
muses = list_muses()
stream(muses[0]['address'])
```

---

## Dependencies
- `numpy` — numerical computation
- `scipy` — signal filtering (Butterworth), FFT
- `matplotlib` — plotting and visualization
- `scikit-learn` — SVM classifier
- `pygame` — audio playback (optional, for real MP3 playback)

---

## References
1. Wolpaw et al., "Brain–Computer Interfaces for Communication and Control," IEEE, 2004.
2. Nicolas-Alonso & Gomez-Gil, "Brain Computer Interfaces: A Review," Sensors, 2012.
3. Teplan, "Fundamentals of EEG Measurement," Measurement Science Review, 2002.
