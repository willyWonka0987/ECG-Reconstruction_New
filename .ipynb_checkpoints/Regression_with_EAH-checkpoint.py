import joblib
import numpy as np
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import pearsonr
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.callbacks import EarlyStopping

# Load QRS segments
train_segments = joblib.load("qrs_train_segments.pkl")
test_segments = joblib.load("qrs_test_segments.pkl")

# Load heart axis metadata (same order as segments)
features_train = joblib.load("../data/features_train_clean.pkl")
features_test = joblib.load("../data/features_test_clean.pkl")

# One-hot encode heart axis
def create_one_hot(train, test):
    f_lst = ['ecg_id', 'superclasses', 'heart_axis']
    df_train = pd.DataFrame(train, columns=f_lst)
    df_test = pd.DataFrame(test, columns=f_lst)
    n = df_train.shape[0]

    features = pd.concat([df_train, df_test])
    heart_axis = pd.get_dummies(features, columns=['heart_axis'], drop_first=True)
    ha_encoded = heart_axis.loc[:, 'heart_axis_1':'heart_axis_8']
    return ha_encoded.iloc[:n].values, ha_encoded.iloc[n:].values

ha_train, ha_test = create_one_hot(features_train, features_test)

lead_names = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
input_lead_idx = 0  # Lead I
target_leads_idx = [1, 6, 7, 8, 9, 10, 11]  # II, V1–V6

os.makedirs("regression_models", exist_ok=True)
os.makedirs("regression_EAH_models", exist_ok=True)  # Added this line to prevent directory issues

def extract_features(lead_data):
    return np.array(lead_data).flatten()  # shape: (6,)

with open("regression_results.txt", "w") as f:
    for target_idx in target_leads_idx:
        X_train, Y_train, X_test, Y_test = [], [], [], []
        ha_train_idx = 0  # Track position in ha_train separately
        ha_test_idx = 0   # Track position in ha_test separately

        # Process training data
        for beat in train_segments:
            if len(beat) <= max(input_lead_idx, target_idx):
                continue  # Skip this beat but don't increment ha_train_idx
            x = extract_features(beat[input_lead_idx])
            x_eah = ha_train[ha_train_idx]  # Use the separate counter
            X_train.append(np.concatenate([x, x_eah]))
            Y_train.append(extract_features(beat[target_idx]))
            ha_train_idx += 1  # Only increment when we actually use a sample

        # Process test data
        for beat in test_segments:
            if len(beat) <= max(input_lead_idx, target_idx):
                continue  # Skip this beat but don't increment ha_test_idx
            if ha_test_idx >= len(ha_test):  # Safety check
                break
            x = extract_features(beat[input_lead_idx])
            x_eah = ha_test[ha_test_idx]  # Use the separate counter
            X_test.append(np.concatenate([x, x_eah]))
            Y_test.append(extract_features(beat[target_idx]))
            ha_test_idx += 1  # Only increment when we actually use a sample

        X_train = np.array(X_train)
        Y_train = np.array(Y_train)
        X_test = np.array(X_test)
        Y_test = np.array(Y_test)

        # Verify we have data
        if len(X_train) == 0 or len(X_test) == 0:
            print(f"Skipping lead {lead_names[target_idx]} - no valid data")
            continue

        # Split train/validation
        X_train_split, X_val, Y_train_split, Y_val = train_test_split(
            X_train, Y_train, test_size=0.2, random_state=42)

        # Model with EAH input (14 features total: 6 + 8)
        model = Sequential([
            Input(shape=(14,)),
            Dense(256, activation='relu'),
            Dense(256, activation='relu'),
            Dense(256, activation='relu'),
            Dense(6)
        ])
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])

        # Train
        es = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        history = model.fit(X_train_split, Y_train_split, validation_data=(X_val, Y_val),
                          epochs=100, batch_size=32, verbose=1, callbacks=[es])

        # Validation eval
        Y_val_pred = model.predict(X_val)
        mse_val = mean_squared_error(Y_val, Y_val_pred)
        rmse_val = np.sqrt(mse_val)
        r2_val = r2_score(Y_val, Y_val_pred)
        pearson_avg_val = np.mean([pearsonr(Y_val[:, i], Y_val_pred[:, i])[0] for i in range(6)])

        # Test eval
        Y_test_pred = model.predict(X_test)
        mse_test = mean_squared_error(Y_test, Y_test_pred)
        rmse_test = np.sqrt(mse_test)
        r2_test = r2_score(Y_test, Y_test_pred)
        pearson_avg_test = np.mean([pearsonr(Y_test[:, i], Y_test_pred[:, i])[0] for i in range(6)])

        # Save model
        model_path = f"regression_EAH_models/model_predict_{lead_names[target_idx]}.keras"
        model.save(model_path)

        # Save results
        f.write(f"Model for Lead {lead_names[target_idx]}:\n")
        f.write(f"  [VAL]  RMSE: {rmse_val:.4f}, R2: {r2_val:.4f}, Pearson Corr: {pearson_avg_val:.4f}\n")
        f.write(f"  [TEST] RMSE: {rmse_test:.4f}, R2: {r2_test:.4f}, Pearson Corr: {pearson_avg_test:.4f}\n")
        f.write(f"  Model saved to: {model_path}\n\n")

print("✅ All models trained and evaluated with EAH.")