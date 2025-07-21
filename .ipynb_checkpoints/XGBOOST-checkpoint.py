from scipy.stats import pearsonr
import joblib
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from xgboost import XGBRegressor
from tqdm import tqdm  # ✅ Progress bar

# Load training and testing segments
train_segments = joblib.load("qrs_train_segments.pkl")
test_segments = joblib.load("qrs_test_segments.pkl")

lead_names = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
input_lead_idx = 0  # Lead I
target_leads_idx = [1, 6, 7, 8, 9, 10, 11]  # II, V1–V6

os.makedirs("xgboost_models", exist_ok=True)

def extract_features(lead_data):
    return np.array(lead_data).flatten()  # shape: (6,)

with open("xgboost_results.txt", "w") as f:
    for target_idx in tqdm(target_leads_idx, desc="Training models"):
        X_train, Y_train = [], []
        X_test, Y_test = [], []

        # Prepare training data
        for beat in train_segments:
            if len(beat) <= max(input_lead_idx, target_idx):
                continue
            x = extract_features(beat[input_lead_idx])
            y = extract_features(beat[target_idx])
            X_train.append(x)
            Y_train.append(y)

        # Prepare testing data
        for beat in test_segments:
            if len(beat) <= max(input_lead_idx, target_idx):
                continue
            x = extract_features(beat[input_lead_idx])
            y = extract_features(beat[target_idx])
            X_test.append(x)
            Y_test.append(y)

        X_train = np.array(X_train)
        Y_train = np.array(Y_train)
        X_test = np.array(X_test)
        Y_test = np.array(Y_test)

        # Train XGBoost model
        model = XGBRegressor(
            objective='reg:squarederror',
            n_estimators=100,
            learning_rate=0.3,
            max_depth=5,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42
        )
        model.fit(X_train, Y_train)

        # Evaluate on test set
        Y_pred = model.predict(X_test)
        mse = mean_squared_error(Y_test, Y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(Y_test, Y_pred)
        
        # Pearson correlation (averaged over 6 features)
        pearsons = []
        for i in range(Y_test.shape[1]):
            r, _ = pearsonr(Y_test[:, i], Y_pred[:, i])
            pearsons.append(r)
        avg_pearson = np.mean(pearsons)
        
        # Save model
        model_path = f"xgboost_models/model_predict_{lead_names[target_idx]}.json"
        model.save_model(model_path)
        
        # Save results
        f.write(f"Model for Lead {lead_names[target_idx]}:\n")
        f.write(f"  Test MSE: {mse:.4f}\n")
        f.write(f"  Test RMSE: {rmse:.4f}\n")
        f.write(f"  Test R2 Score: {r2:.4f}\n")
        f.write(f"  Pearson Corr (avg over 6 dims): {avg_pearson:.4f}\n")
        f.write(f"  Model saved to: {model_path}\n\n")

print("Training and evaluation complete. Models and results saved.")
