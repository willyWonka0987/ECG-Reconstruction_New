import numpy as np
import pickle
import matplotlib.pyplot as plt
from pathlib import Path

# --- Paths ---
DATA_DIR = Path("RichECG_Datasets")
SEGMENT_PATH = DATA_DIR / "segments_train.npy"
FEATURE_PATH = DATA_DIR / "features_train.pkl"
LEAD_NAMES = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
COLORS = {'P': 'green', 'Q': 'blue', 'R': 'red', 'S': 'orange', 'T': 'purple'}
sampling_rate = 100

# --- Load first segment ---
segments = np.load(SEGMENT_PATH, mmap_mode='r')
first_segment = segments[0]  # shape = (128, 12)
time = np.arange(128) / sampling_rate  # 128 samples @ 100Hz

# --- Load first feature dictionary ---
with open(FEATURE_PATH, 'rb') as f:
    first_feature = pickle.load(f)

# --- Plotting ---
fig, axs = plt.subplots(3, 4, figsize=(18, 10))
axs = axs.flatten()

for lead_idx, lead_name in enumerate(LEAD_NAMES):
    signal = first_segment[:, lead_idx]
    ax = axs[lead_idx]
    ax.plot(time, signal, label='ECG', color='black')

    waves = first_feature['waves'].get(lead_name, {})
    for wave in ['P', 'Q', 'R', 'S', 'T']:
        amp = waves.get(f"{wave}_amp")
        t = waves.get(f"{wave}_time")
        if amp is not None and t is not None:
            local_time = t - t + time[64]  # align center of segment
            ax.plot(local_time, amp, 'o', label=wave, color=COLORS[wave])

    ax.set_title(f"Lead: {lead_name}")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.grid(True)
    ax.legend(loc='upper right', fontsize=8)

fig.suptitle("PQRST Overlay on First Segment (Train Set)", fontsize=16)
fig.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()
