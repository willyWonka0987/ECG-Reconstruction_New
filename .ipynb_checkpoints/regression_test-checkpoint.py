import os
import joblib
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use Agg backend for non-GUI environments
import matplotlib.pyplot as plt
from keras.models import load_model

# Ensure output folder exists
os.makedirs("regression_test_plots", exist_ok=True)

# Load ECG segments
segments = joblib.load("qrs_test_segments.pkl")

# Lead order and targets
lead_names = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
target_leads = ['II', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
target_indices = [lead_names.index(lead) for lead in target_leads]

# Cache models (optional but faster)
models = {
    lead: load_model(f"regression_models/model_predict_{lead}.keras")
    for lead in target_leads
}

# Process first 20 ECG segments
for ecg_idx in range(20):
    segment = segments[ecg_idx]
    lead_i_data = segment[0]  # Lead I input
    x_input = np.array([item for pair in lead_i_data for item in pair]).reshape(1, -1)

    plt.figure(figsize=(16, 10))
    for i, (lead, idx) in enumerate(zip(target_leads, target_indices)):
        model = models[lead]
        y_pred = model.predict(x_input, verbose=0)[0]
        lead_data = segment[idx]
        y_true = np.array([item for pair in lead_data for item in pair])

        plt.subplot(3, 3, i+1)
        # Actual (red = time, blue = amp)
        plt.scatter(y_true[::2], y_true[1::2], color='red', label='True' if i == 0 else "")
        # Predicted
        plt.scatter(y_pred[::2], y_pred[1::2], color='blue', marker='x', label='Pred' if i == 0 else "")
        # Annotate QRS
        for j, label in enumerate(['Q', 'R', 'S']):
            plt.annotate(label, (y_true[2*j], y_true[2*j+1]), color='red')
            plt.annotate(label, (y_pred[2*j], y_pred[2*j+1]), color='blue')

        plt.title(f'Lead {lead}')
        plt.xlabel("Time")
        plt.ylabel("Amplitude")
        plt.grid(True)

    plt.suptitle(f"Actual vs Predicted QRS Features - ECG #{ecg_idx}", fontsize=16)
    plt.legend(loc='upper right')
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(f"regression_test_plots/ecg_{ecg_idx:02d}.png")
    plt.close()
