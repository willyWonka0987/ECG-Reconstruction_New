# 📈 Plotting the first cleaned ECG signal (12 leads, 10 seconds)
import matplotlib.pyplot as plt
import joblib
import os

pkl_path = 'ecg_train_clean.pkl'

# Check if the file exists
if not os.path.exists(pkl_path):
    raise FileNotFoundError(f"File not found: {pkl_path}")

# Load using joblib (not pickle)
ecg_train_clean = joblib.load(pkl_path)

# Extract the first ECG: shape (1000, 12)
first_ecg = ecg_train_clean[0]

# Lead names for readability
lead_names = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']

# Plotting
plt.figure(figsize=(15, 10))
for lead in range(12):
    plt.subplot(6, 2, lead + 1)
    plt.plot(first_ecg[:, lead], linewidth=1)
    plt.title(f"Lead {lead_names[lead]}")
    plt.xticks([])
    plt.yticks([])

plt.suptitle("First Cleaned ECG (10 seconds, All 12 Leads)", fontsize=16, y=1.02)
plt.tight_layout()
plt.show()
