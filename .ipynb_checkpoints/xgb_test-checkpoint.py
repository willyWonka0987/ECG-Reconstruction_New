import os
import joblib
import numpy as np
import matplotlib
matplotlib.use('Agg')  # For headless environments
import matplotlib.pyplot as plt
from xgboost import XGBRegressor

# Ensure output folder exists
os.makedirs("xgb_test_plots", exist_ok=True)

# Load ECG segments
segments = joblib.load("qrs_test_segments.pkl")

# Lead names and target indices
lead_names = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
target_leads = ['II', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
target_indices = [lead_names.index(lead) for lead in target_leads]

# Load XGBoost models
models = {
    lead: XGBRegressor()
    for lead in target_leads
}
for lead in target_leads:
    models[lead].load_model(f"xgboost_models/model_predict_{lead}.json")

def extract_features(lead_data):
    return np.array(lead_data).flatten().reshape(1, -1)  # shape: (1, 6)

# Process first 20 ECGs
for ecg_idx in range(20):
    segment = segments[ecg_idx]
    lead_i_data = segment[0]  # Lead I
    x_input = extract_features(lead_i_data)

    plt.figure(figsize=(16, 10))
    for i, (lead, idx) in enumerate(zip(target_leads, target_indices)):
        model = models[lead]
        y_pred = model.predict(x_input)[0]  # shape: (6,)
        lead_data = segment[idx]
        y_true = np.array(lead_data).flatten()  # shape: (6,)

        plt.subplot(3, 3, i + 1)
        plt.scatter(y_true[::2], y_true[1::2], color='red', label='True' if i == 0 else "")
        plt.scatter(y_pred[::2], y_pred[1::2], color='blue', marker='x', label='Pred' if i == 0 else "")

        for j, label in enumerate(['Q', 'R', 'S']):
            plt.annotate(label, (y_true[2 * j], y_true[2 * j + 1]), color='red')
            plt.annotate(label, (y_pred[2 * j], y_pred[2 * j + 1]), color='blue')

        plt.title(f'Lead {lead}')
        plt.xlabel("Time")
        plt.ylabel("Amplitude")
        plt.grid(True)

    plt.suptitle(f"XGBoost: True vs Predicted QRS - ECG #{ecg_idx}", fontsize=16)
    plt.legend(loc='upper right')
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(f"xgb_test_plots/ecg_{ecg_idx:02d}.png")
    plt.close()

print("Plots for first 20 ECGs saved in 'test_plots' folder.")
