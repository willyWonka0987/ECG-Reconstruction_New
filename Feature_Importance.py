# Feature selection using XGBoost
import pickle
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from xgboost import XGBRegressor
import pandas as pd
from tqdm import tqdm

SAVE_DIR = Path("RichECG_Datasets")
INPUT_LEADS = ["I", "V2", "V6"]
TARGET_LEAD = "V1"
SEGMENT_LENGTH = 80

# Load features and segments
features_path = SAVE_DIR / "features_train.pkl"
segments_path = SAVE_DIR / "segments_train.npy"
segments = np.load(segments_path)[:, :SEGMENT_LENGTH, :]

X, Y = [], []

with open(features_path, "rb") as f:
    while True:
        try:
            rec = pickle.load(f)
            if rec.get("segment_index") is None:
                continue
            if not all(lead in rec["features"] for lead in INPUT_LEADS + [TARGET_LEAD]):
                continue
            seg_idx = rec["segment_index"]
            if seg_idx >= segments.shape[0]:
                continue

            full_segment_inputs = []
            for lead in INPUT_LEADS:
                lead_index = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"].index(lead)
                full_segment_inputs.append(segments[seg_idx, :, lead_index])
            full_segment_inputs = np.concatenate(full_segment_inputs)

            advanced_features_inputs = np.concatenate([rec["features"][lead] for lead in INPUT_LEADS])

            metadata = rec.get("metadata", {})
            if isinstance(metadata, dict):
                meta_values = np.array(list(metadata.values()), dtype=np.float32)
            else:
                meta_values = np.zeros(1, dtype=np.float32)

            x = np.concatenate([advanced_features_inputs, full_segment_inputs, meta_values])
            lead_index = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"].index(TARGET_LEAD)
            y = segments[seg_idx, :, lead_index]

            if np.all(np.isfinite(x)) and np.all(np.isfinite(y)):
                X.append(x)
                Y.append(y)
        except EOFError:
            break

X = np.array(X)
Y = np.array(Y)

# Fit XGBoost and get feature importances
print("\nTraining XGBoost to rank feature importance...")
xgb = XGBRegressor(n_estimators=100, max_depth=4, verbosity=0)
xgb.fit(X, Y)

importances = xgb.feature_importances_
indices = np.argsort(importances)[::-1]

# Plot top N features
TOP_K = 30
plt.figure(figsize=(12, 8))
plt.title(f"Top {TOP_K} Feature Importances for Lead {TARGET_LEAD}")
plt.bar(range(TOP_K), importances[indices[:TOP_K]], align="center")
plt.xticks(range(TOP_K), [f"feat_{i}" for i in indices[:TOP_K]], rotation=45, ha="right")
plt.tight_layout()
plt.show()

# Print ranked feature indices and their importance
print(f"\nTop {TOP_K} Features for Lead {TARGET_LEAD}:")
for i in range(TOP_K):
    idx = indices[i]
    print(f"{i+1}. Feature {idx:4d}  -> Importance: {importances[idx]:.6f}")
