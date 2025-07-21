import os
import joblib
import matplotlib.pyplot as plt
import neurokit2 as nk
import numpy as np

# --- Parameters ---
sampling_rate = 100  # Hz
pre_window  = 0.06   # seconds before R to search for Q
post_window = 0.06   # seconds after  R to search for S
lead_names = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']

# Create output directory
os.makedirs("QRS_plots", exist_ok=True)

# Load ECG dataset
ecg_data = joblib.load("ecg_train_clean.pkl")

# --- Q and S detection function ---
def find_qs_minima(signal, r_peaks, pre_window, post_window, sampling_rate):
    pre_samples  = int(pre_window  * sampling_rate)
    post_samples = int(post_window * sampling_rate)
    q_points, s_points = [], []

    for r in r_peaks:
        start = max(0, r - pre_samples)
        seg_q = signal[start:r]
        q = start + np.argmin(seg_q) if len(seg_q) > 0 else r
        q_points.append(q)

        end = min(len(signal), r + post_samples)
        seg_s = signal[r:end]
        s = r + np.argmin(seg_s) if len(seg_s) > 0 else r
        s_points.append(s)

    return np.array(q_points), np.array(s_points)

# --- Loop through first 10 ECGs ---
for idx in range(10):
    ecg_12lead = ecg_data[idx]  # Shape: (1000, 12)

    plt.figure(figsize=(22, 30))
    for lead_idx in range(12):
        signal = ecg_12lead[:, lead_idx]
        cleaned = nk.ecg_clean(signal, sampling_rate=sampling_rate)
        r_peaks = nk.ecg_peaks(cleaned, sampling_rate=sampling_rate)[1]['ECG_R_Peaks']
        q_peaks, s_peaks = find_qs_minima(cleaned, r_peaks, pre_window, post_window, sampling_rate)

        # Plot
        plt.subplot(12, 1, lead_idx + 1)
        plt.plot(cleaned, label='ECG Signal', linewidth=1)
        plt.plot(r_peaks, cleaned[r_peaks], 'ro', markersize=3, label='R-peaks')
        plt.plot(q_peaks, cleaned[q_peaks], 'go', markersize=3, label='Q-points')
        plt.plot(s_peaks, cleaned[s_peaks], 'bo', markersize=3, label='S-points')
        plt.title(f"Lead {lead_names[lead_idx]}")
        plt.ylabel("Amplitude")
        plt.grid(True)
        if lead_idx == 0:
            plt.legend(loc='upper right', ncol=3)

    plt.xlabel("Sample Index (0 to 999)")
    plt.tight_layout()
    plt.savefig(f"QRS_plots/ecg_{idx:02d}_qrs.png")
    plt.close()

print("✅ Saved QRS plots for the first 10 ECGs to 'QRS_plots' folder.")
