import joblib
import matplotlib.pyplot as plt
import neurokit2 as nk

# Load the first cleaned ECG
ecg_data = joblib.load("ecg_train_clean.pkl")
first_ecg = ecg_data[0]  # Shape: (1000, 12)

# Leads to keep: I, II, V1–V6 → indexes: 0, 1, 6, 7, 8, 9, 10, 11
selected_leads = [0, 1, 6, 7, 8, 9, 10, 11]
selected_lead_names = ['I', 'II', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']

sampling_rate = 100  # Hz

# Plot only the selected leads with R-peaks
plt.figure(figsize=(15, 10))
for idx, lead in enumerate(selected_leads):
    signal = first_ecg[:, lead]

    # Apply Pan-Tompkins QRS detection
    try:
        processed = nk.ecg_process(signal, sampling_rate=sampling_rate)
        rpeaks = processed[1]['ECG_R_Peaks']
    except Exception as e:
        print(f"Lead {selected_lead_names[idx]} error: {e}")
        rpeaks = []

    plt.subplot(4, 2, idx + 1)
    plt.plot(signal, label='ECG', linewidth=1)
    if len(rpeaks) > 0:
        plt.plot(rpeaks, signal[rpeaks], 'ro', markersize=3, label='R-peaks')
    plt.title(f"Lead {selected_lead_names[idx]}")
    plt.xticks([])
    plt.yticks([])

plt.suptitle("First Cleaned ECG (Only Leads I, II, V1–V6) with R-peaks", fontsize=16, y=1.02)
plt.tight_layout()
plt.show()
