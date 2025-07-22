import pickle
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict

SAVE_DIR = Path("wave_statistics")
SAVE_DIR.mkdir(parents=True, exist_ok=True)

LEADS = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF',
         'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
WAVES = ['P', 'Q', 'R', 'S', 'T']

with open("RichECG_Datasets/features_train.pkl", "rb") as f:
    records = []
    while True:
        try:
            records.append(pickle.load(f))
        except EOFError:
            break

# Collect data per wave and lead
amp_data = {wave: defaultdict(list) for wave in WAVES}
pos_data = {wave: defaultdict(list) for wave in WAVES}

for record in records:
    waves = record.get('waves', {})
    for lead in LEADS:
        for wave in WAVES:
            amp_key = f"{wave}_amp"
            loc_key = f"{wave}_time"
            if lead in waves:
                amp = waves[lead].get(amp_key)
                pos = waves[lead].get(loc_key)
                if amp is not None and pos is not None:
                    if not np.isnan(amp) and not np.isnan(pos):
                        amp_data[wave][lead].append(amp)
                        pos_data[wave][lead].append(pos)

# Plot function
def plot_wave_stats(wave):
    fig, axs = plt.subplots(nrows=12, ncols=2, figsize=(16, 36))
    fig.suptitle(f"{wave}-Wave Distribution Analysis", fontsize=16)

    for idx, lead in enumerate(LEADS):
        row = idx

        # Position
        pos = np.array(pos_data[wave][lead])
        if len(pos) == 0:
            print(f"❌ No position data for {wave}, lead {lead}")
            continue
        mean = np.mean(pos)
        median = np.median(pos)
        std = np.std(pos)
        outliers = np.sum((pos < mean - 5 * std) | (pos > mean + 5 * std))

        ax_pos = axs[row, 0]
        ax_pos.hist(pos, bins=40, color='skyblue', edgecolor='black')
        ax_pos.axvline(mean, color='red', label=f"Mean: {mean:.3f}")
        ax_pos.axvline(median, color='green', label=f"Median: {median:.3f}")
        ax_pos.axvline(mean - std, color='orange', linestyle='--', label=f"±1σ: {std:.3f}")
        ax_pos.axvline(mean + std, color='orange', linestyle='--')
        ax_pos.axvline(mean - 3 * std, color='purple', linestyle=':', label=f"±3σ: {3 * std:.3f}")
        ax_pos.axvline(mean + 3 * std, color='purple', linestyle=':')
        ax_pos.set_title(f"Lead {lead} - {wave}-wave Position\nOutliers > 5σ: {outliers}")
        ax_pos.set_xlabel("Time (s)")
        ax_pos.set_ylabel("Frequency")
        ax_pos.legend()

        # Amplitude
        amp = np.array(amp_data[wave][lead])
        if len(amp) == 0:
            print(f"❌ No amplitude data for {wave}, lead {lead}")
            continue
        mean = np.mean(amp)
        median = np.median(amp)
        std = np.std(amp)
        outliers = np.sum((amp < mean - 5 * std) | (amp > mean + 5 * std))

        ax_amp = axs[row, 1]
        ax_amp.hist(amp, bins=40, color='skyblue', edgecolor='black')
        ax_amp.axvline(mean, color='red', label=f"Mean: {mean:.3f}")
        ax_amp.axvline(median, color='green', label=f"Median: {median:.3f}")
        ax_amp.axvline(mean - std, color='orange', linestyle='--', label=f"±1σ: {std:.3f}")
        ax_amp.axvline(mean + std, color='orange', linestyle='--')
        ax_amp.axvline(mean - 3 * std, color='purple', linestyle=':', label=f"±3σ: {3 * std:.3f}")
        ax_amp.axvline(mean + 3 * std, color='purple', linestyle=':')

        # Highlight outliers > 5σ
        outlier_indices = np.where((amp < mean - 5 * std) | (amp > mean + 5 * std))[0]
        outlier_values = amp[outlier_indices]
        ax_amp.scatter(outlier_values, np.zeros_like(outlier_values), color='red', s=5)

        ax_amp.set_title(f"Lead {lead} - {wave}-wave Amplitude\nOutliers > 5σ: {outliers}")
        ax_amp.set_xlabel("Amplitude (mV)")
        ax_amp.set_ylabel("Frequency")
        ax_amp.legend()

    fig.tight_layout(rect=[0, 0, 1, 0.98])
    fig.savefig(SAVE_DIR / f"{wave}_wave_stats.png", dpi=300)
    plt.close(fig)

# Run for all waves
for wave in WAVES:
    plot_wave_stats(wave)

print("✅ All wave statistics plotted and saved.")
