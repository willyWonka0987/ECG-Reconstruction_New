import os
import joblib
import numpy as np
import matplotlib
matplotlib.use('Agg')  # For headless environments
import matplotlib.pyplot as plt
import pandas as pd
from keras.models import load_model

# Ensure output folder exists
os.makedirs("regression_test_EAH_plots", exist_ok=True)

# Load ECG segments
segments = joblib.load("qrs_test_segments.pkl")

# Load heart axis metadata
features_test = joblib.load("../data/features_test_clean.pkl")

# One-hot encode heart axis
def create_ha_one_hot(test):
    df_test = pd.DataFrame(test, columns=['ecg_id', 'superclasses', 'heart_axis'])
    ha_encoded = pd.get_dummies(df_test['heart_axis'], prefix='heart_axis')
    ha_encoded = ha_encoded.reindex(columns=[f'heart_axis_{i}' for i in range(1, 9)], fill_value=0)
    return ha_encoded.values

ha_test = create_ha_one_hot(features_test)

# Lead order and targets
lead_names = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
target_leads = ['II', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
target_indices = [lead_names.index(lead) for lead in target_leads]

# Cache models
models = {
    lead: load_model(f"regression_EAH_models/model_predict_{lead}.keras")
    for lead in target_leads
}

# Process first 20 ECG segments
for ecg_idx in range(20):
    segment = segments[ecg_idx]
    lead_i_data = segment[0]  # Lead I input
    x_lead = np.array([item for pair in lead_i_data for item in pair], dtype=np.float32)  # shape (6,)
    x_eah = np.array(ha_test[ecg_idx], dtype=np.float32)  # shape (8,)
    x_input = np.concatenate([x_lead, x_eah]).reshape(1, -1)  # shape (1, 14)


    plt.figure(figsize=(16, 10))
    for i, (lead, idx) in enumerate(zip(target_leads, target_indices)):
        model = models[lead]
        y_pred = model.predict(x_input, verbose=0)[0]
        lead_data = segment[idx]
        y_true = np.array([item for pair in lead_data for item in pair])

        plt.subplot(3, 3, i+1)
        # Actual
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

    plt.suptitle(f"Actual vs Predicted QRS Features (with EAH) - ECG #{ecg_idx}", fontsize=16)
    plt.legend(loc='upper right')
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(f"regression_test_EAH_plots/ecg_{ecg_idx:02d}.png")
    plt.close()
