# ✅ Full version with Stacked Ensemble in both stages (V5 + Others)

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pickle
import numpy as np
from pathlib import Path
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import pearsonr
from xgboost import XGBRegressor
from tqdm import tqdm
import matplotlib.pyplot as plt
import os

# --- Config ---
SAVE_DIR = Path("RichECG_Datasets")
OUTPUT_DIR = Path("Stacked_Model_Results")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR = OUTPUT_DIR / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR = OUTPUT_DIR / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
REPORT_FILE = OUTPUT_DIR / "evaluation_report.txt"
RMSE_PLOTS_DIR = PLOTS_DIR / "rmse_per_point"
RMSE_PLOTS_DIR.mkdir(parents=True, exist_ok=True)

INPUT_LEADS = ["I", "V2", "V6"]
TARGET_LEADS = ["II", "V1", "V3", "V4"]  # V5 removed for initial prediction
SEGMENT_LENGTH = 80
BATCH_SIZE = 32
EPOCHS = 300
LEARNING_RATE = 1e-3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class RichECGDataset(Dataset):
    def __init__(self, features_path, segments_path, target_lead, v5_filtered_points=None):
        self.samples = []
        segments = np.load(segments_path)[:, :SEGMENT_LENGTH, :]
        with open(features_path, "rb") as f:
            while True:
                try:
                    rec = pickle.load(f)
                    if rec.get("segment_index") is None:
                        continue
                    if not all(lead in rec["features"] for lead in INPUT_LEADS + [target_lead]):
                        continue
                    seg_idx = rec["segment_index"]
                    if seg_idx >= segments.shape[0]:
                        continue

                    full_segment_inputs = []
                    for lead in INPUT_LEADS:
                        lead_index = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"].index(lead)
                        full_segment_inputs.append(segments[seg_idx, :, lead_index])
                    full_segment_inputs = np.concatenate(full_segment_inputs)

                    if v5_filtered_points is not None:
                        v5_segment = v5_filtered_points.get(seg_idx)
                        if v5_segment is None:
                            continue
                        full_segment_inputs = np.concatenate([full_segment_inputs, v5_segment])

                    features_inputs = np.concatenate([rec["features"][lead] for lead in INPUT_LEADS])
                    x = np.concatenate([features_inputs, full_segment_inputs])

                    lead_index = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"].index(target_lead)
                    y = segments[seg_idx, :, lead_index]
                    if np.all(np.isfinite(x)) and np.all(np.isfinite(y)):
                        self.samples.append((x, y))
                except EOFError:
                    break
        print(f"{features_path.name} ({target_lead}) - num of samples : {len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x, y = self.samples[idx]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

class MLP(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512), nn.LayerNorm(512), nn.LeakyReLU(0.1), nn.Dropout(0.3),
            nn.Linear(512, 256), nn.LayerNorm(256), nn.LeakyReLU(0.1), nn.Dropout(0.3),
            nn.Linear(256, 128), nn.LayerNorm(128), nn.LeakyReLU(0.1), nn.Dropout(0.2),
            nn.Linear(128, 64), nn.LeakyReLU(0.1),
            nn.Linear(64, output_dim)
        )

    def forward(self, x):
        return self.net(x)

def collect_predictions(model, dataset):
    xs, ys, preds = [], [], []
    model.eval()
    with torch.no_grad():
        for xb, yb in DataLoader(dataset, batch_size=BATCH_SIZE):
            xs.append(xb.numpy())
            ys.append(yb.numpy())
            xb_gpu = xb.to(DEVICE)
            pred = model(xb_gpu).cpu().numpy()
            preds.append(pred)
    return np.vstack(xs), np.vstack(ys), np.vstack(preds)

# --- Stage 1: Train ensemble for V5 ---
v5_ds = RichECGDataset(SAVE_DIR / "features_train.pkl", SAVE_DIR / "segments_train.npy", "V5")
v5_loader = DataLoader(v5_ds, batch_size=BATCH_SIZE, shuffle=True)
input_dim = len(v5_ds[0][0])
mlp_v5 = MLP(input_dim, SEGMENT_LENGTH).to(DEVICE)
optimizer = torch.optim.Adam(mlp_v5.parameters(), lr=LEARNING_RATE)
loss_fn = nn.MSELoss()

best_loss = float("inf")
counter = 0
for epoch in range(EPOCHS):
    mlp_v5.train()
    total_loss = 0
    for xb, yb in tqdm(v5_loader, desc=f"[V5 Epoch {epoch+1}]", leave=False):
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        pred = mlp_v5(xb)
        loss = loss_fn(pred, yb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * xb.size(0)
    avg_loss = total_loss / len(v5_ds)
    if avg_loss < best_loss:
        best_loss = avg_loss
        best_model_v5 = mlp_v5.state_dict()
        counter = 0
    else:
        counter += 1
        if counter >= 15:
            print(f"[V5] Early stopping at epoch {epoch+1}")
            break
mlp_v5.load_state_dict(best_model_v5)

xs, ys, mlp_out = collect_predictions(mlp_v5, v5_ds)
xgb_v5 = XGBRegressor(n_estimators=100, max_depth=4, verbosity=0)
xgb_v5.fit(xs, ys)
xgb_preds = xgb_v5.predict(xs)
meta_input = np.hstack([mlp_out, xgb_preds])
meta_model_v5 = Ridge(alpha=1.0)
meta_model_v5.fit(meta_input, ys)
pred_v5_final = meta_model_v5.predict(meta_input)
rmse_v5 = np.sqrt((pred_v5_final - ys) ** 2)
mask = (rmse_v5 < 0.01).astype(np.float32)
v5_filtered_points = {i: pred_v5_final[i] * mask[i] for i in range(len(ys))}

# --- Stage 2: Train models for remaining leads ---
with open(REPORT_FILE, "w") as report:
    for lead in TARGET_LEADS:
        print(f"\n🔧 Training model for lead: {lead} using V5-filtered points...")
        train_ds = RichECGDataset(SAVE_DIR / "features_train.pkl", SAVE_DIR / "segments_train.npy", lead, v5_filtered_points)
        val_ds = RichECGDataset(SAVE_DIR / "features_val.pkl", SAVE_DIR / "segments_val.npy", lead, v5_filtered_points)
        test_ds = RichECGDataset(SAVE_DIR / "features_test.pkl", SAVE_DIR / "segments_test.npy", lead, v5_filtered_points)

        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)

        input_dim = len(train_ds[0][0])
        mlp = MLP(input_dim, SEGMENT_LENGTH).to(DEVICE)
        optimizer = torch.optim.Adam(mlp.parameters(), lr=LEARNING_RATE)

        best_val_loss = float("inf")
        counter = 0
        for epoch in range(EPOCHS):
            mlp.train()
            total_loss = 0.0
            for xb, yb in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} - Lead {lead}", leave=False):
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                pred = mlp(xb)
                loss = loss_fn(pred, yb)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * xb.size(0)
            avg_train_loss = total_loss / len(train_ds)

            mlp.eval()
            val_loss = 0.0
            with torch.no_grad():
                for xb, yb in val_loader:
                    xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                    pred = mlp(xb)
                    loss = loss_fn(pred, yb)
                    val_loss += loss.item() * xb.size(0)
            avg_val_loss = val_loss / len(val_ds)

            print(f"Epoch {epoch+1} - Train Loss: {avg_train_loss:.4f} - Val Loss: {avg_val_loss:.4f}")
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_model_state = mlp.state_dict()
                counter = 0
            else:
                counter += 1
                if counter >= 15:
                    print(f"Early stopping at epoch {epoch+1}")
                    break

        mlp.load_state_dict(best_model_state)

        xs, ys, mlp_out = collect_predictions(mlp, test_ds)
        xgb_model = XGBRegressor(n_estimators=100, max_depth=4, verbosity=0)
        xgb_model.fit(xs, ys)
        xgb_preds = xgb_model.predict(xs)
        meta_input = np.hstack([mlp_out, xgb_preds])
        meta_model = Ridge(alpha=1.0)
        meta_model.fit(meta_input, ys)
        meta_pred = meta_model.predict(meta_input)

        with open(MODELS_DIR / f"ridge_model_{lead}.pkl", "wb") as f:
            pickle.dump(meta_model, f)
        with open(MODELS_DIR / f"xgb_model_{lead}.pkl", "wb") as f:
            pickle.dump(xgb_model, f)

        rmse = np.sqrt(mean_squared_error(ys, meta_pred))
        r2 = r2_score(ys, meta_pred)
        pearson_corr = np.mean([
            pearsonr(ys[:, i], meta_pred[:, i])[0] for i in range(SEGMENT_LENGTH) if np.std(ys[:, i]) > 0
        ])

        rmse_per_point = np.sqrt(np.mean((ys - meta_pred) ** 2, axis=0)).tolist()
        report.write(f"\nEvaluation for Lead {lead} (with V5):\n")
        report.write(f"RMSE: {rmse:.4f}\n")
        report.write(f"R^2: {r2:.4f}\n")
        report.write(f"Pearson Correlation: {pearson_corr:.4f}\n")
        report.write(f"RMSE per point (length {SEGMENT_LENGTH}):\n")
        report.write(", ".join(f"{v:.6f}" for v in rmse_per_point) + "\n")

        plt.figure(figsize=(10, 4))
        plt.plot(rmse_per_point, marker='o')
        plt.title(f"RMSE per point - Lead {lead}")
        plt.xlabel("Point index")
        plt.ylabel("RMSE")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(RMSE_PLOTS_DIR / f"rmse_per_point_{lead}.png")
        plt.close()
