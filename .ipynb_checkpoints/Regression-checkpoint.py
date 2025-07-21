import joblib
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import pearsonr
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.callbacks import EarlyStopping

# === Paths ===
QRS_TRAIN_PATH = "QRS_Segments/qrs_train_segments.pkl"
QRS_TEST_PATH = "QRS_Segments/qrs_test_segments.pkl"
SEGMENT_TRAIN_PATH = "data_no_aug/ecg_train_clean.pkl"
SEGMENT_TEST_PATH = "data_no_aug/ecg_test_clean.pkl"
SAVE_DIR = "regression_QRS_to_SEG"
os.makedirs(SAVE_DIR, exist_ok=True)

# === Load Data ===
qrs_train = joblib.load(QRS_TRAIN_PATH)
qrs_test = joblib.load(QRS_TEST_PATH)
segment_train = joblib.load(SEGMENT_TRAIN_PATH)
segment_test = joblib.load(SEGMENT_TEST_PATH)

# === ECG Lead Indices ===
lead_names = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
input_lead_idx = 0  # Lead I
output_leads_idx = [1, 6, 7, 8, 9, 10, 11]  # II, V1-V6

# === Helper ===
def flatten_qrs(lead_data):
    return np.array(lead_data).flatten()  # shape: (6,) -> (6,)

def flatten_segment(lead_data):
    return np.array(lead_data).flatten()  # shape: (128,) -> (128,)

# === Train One Model Per Output Lead ===
with open(os.path.join(SAVE_DIR, "regression_results.txt"), "w") as f:
    for target_idx in output_leads_idx:
        X_train, Y_train, X_test, Y_test = [], [], [], []

        # --- Training Set ---
        for qrs, seg in zip(qrs_train, segment_train):
            if len(qrs) <= max(input_lead_idx, target_idx):
                continue
            x = flatten_qrs(qrs[input_lead_idx])
            y = flatten_segment(seg[:, target_idx])
            X_train.append(x)
            Y_train.append(y)

        # --- Testing Set ---
        for qrs, seg in zip(qrs_test, segment_test):
            if len(qrs) <= max(input_lead_idx, target_idx):
                continue
            x = flatten_qrs(qrs[input_lead_idx])
            y = flatten_segment(seg[:, target_idx])
            X_test.append(x)
            Y_test.append(y)

        X_train = np.array(X_train)
        Y_train = np.array(Y_train)
        X_test = np.array(X_test)
        Y_test = np.array(Y_test)

        # --- Train/Val Split ---
        X_train_split, X_val, Y_train_split, Y_val = train_test_split(
            X_train, Y_train, test_size=0.2, random_state=42)

        # --- Model Architecture ---
        model = Sequential([
            Input(shape=(6,)),
            Dense(256, activation='relu'),
            Dense(256, activation='relu'),
            Dense(256, activation='relu'),
            Dense(128)  # Predicting 128 time points
        ])
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])

        # --- Train Model ---
        es = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        model.fit(X_train_split, Y_train_split,
                  validation_data=(X_val, Y_val),
                  epochs=100, batch_size=32, verbose=1, callbacks=[es])

        # --- Evaluation ---
        def evaluate(name, y_true, y_pred):
            mse = mean_squared_error(y_true, y_pred)
            rmse = np.sqrt(mse)
            r2 = r2_score(y_true, y_pred)
            pearsons = [pearsonr(y_true[:, i], y_pred[:, i])[0] for i in range(y_true.shape[1])]
            return rmse, r2, np.mean(pearsons)

        val_pred = model.predict(X_val)
        test_pred = model.predict(X_test)

        rmse_val, r2_val, pearson_val = evaluate("VAL", Y_val, val_pred)
        rmse_test, r2_test, pearson_test = evaluate("TEST", Y_test, test_pred)

        # --- Save Model and Log ---
        model_path = os.path.join(SAVE_DIR, f"model_predict_{lead_names[target_idx]}.keras")
        model.save(model_path)

        f.write(f"Model for Lead {lead_names[target_idx]}:\n")
        f.write(f"  [VAL]  RMSE: {rmse_val:.4f}, R2: {r2_val:.4f}, Pearson Corr: {pearson_val:.4f}\n")
        f.write(f"  [TEST] RMSE: {rmse_test:.4f}, R2: {r2_test:.4f}, Pearson Corr: {pearson_test:.4f}\n")
        f.write(f"  Model saved to: {model_path}\n\n")

print("✅ All models trained and saved for QRS (Lead I) ➜ ECG Segment (Target Lead).")
