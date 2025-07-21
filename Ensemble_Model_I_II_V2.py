import os
import pickle
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
from tqdm import tqdm, trange

# --- Config ---
SAVE_DIR = Path("RichECG_Datasets")
OUTPUT_DIR = Path("Stacked_Model_Results")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR = OUTPUT_DIR / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR = OUTPUT_DIR / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
REPORT_FILE = OUTPUT_DIR / "evaluation_report.txt"

INPUT_LEADS = ["I", "II", "V2"]
TARGET_LEADS = ["III"]
SEGMENT_LENGTH = 80
BATCH_SIZE = 32
EPOCHS = 1
LEARNING_RATE = 1e-3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def encode_metadata(meta_dict):
    age = meta_dict.get("age", 0)
    sex = 1 if meta_dict.get("sex", "").lower() == "male" else 0
    axis = meta_dict.get("heart_axis", "").lower()
    axis_encoded = [0, 0, 0]
    if axis == "normal":
        axis_encoded[0] = 1
    elif axis == "left axis deviation":
        axis_encoded[1] = 1
    elif axis == "right axis deviation":
        axis_encoded[2] = 1
    return [age, sex] + axis_encoded

class RichECGDataset(Dataset):
    def __init__(self, features_path, segments_path, target_lead):
        self.samples = []
        segments = np.load(segments_path)[:, :SEGMENT_LENGTH, :]
        with open(features_path, "rb") as f:
            while True:
                try:
                    rec = pickle.load(f)
                    seg_idx = rec.get("segment_index")
                    if seg_idx is None or seg_idx >= segments.shape[0]:
                        continue
                    all_leads = INPUT_LEADS + [target_lead]
                    if any(lead not in rec["features"] for lead in all_leads):
                        continue
                    if any(not np.all(np.isfinite(rec["features"][lead])) for lead in all_leads):
                        continue
                    feat = np.concatenate([rec["features"][lead] for lead in INPUT_LEADS])
                    amps, times, qrs_area, qrs_dur = [], [], [], []
                    for lead in INPUT_LEADS:
                        wave = rec["waves"].get(lead, {})
                        amps.extend([wave.get(f"{w}_amp", 0) or 0 for w in ['P', 'Q', 'R', 'S', 'T']])
                        times.extend([wave.get(f"{w}time", 0) or 0 for w in ['P', 'Q', 'R', 'S', 'T']])
                        qrs_area.append(wave.get("QRS_area", 0) or 0)
                        qrs_dur.append(wave.get("QRS_duration", 0) or 0)
                    intervals = []
                    for lead in INPUT_LEADS:
                        lead_intervals = rec["intervals"].get(lead, {})
                        for key in ['PR', 'QT', 'ST', 'RR']:
                            val = lead_intervals.get(key)
                            if isinstance(val, list) and len(val) > 0:
                                mean_val = np.mean(val)
                                intervals.append(mean_val if np.isfinite(mean_val) else 0)
                            else:
                                intervals.append(0)
                    meta_features = encode_metadata(rec["metadata"])
                    x = np.concatenate([feat, amps, times, qrs_area, qrs_dur, intervals, meta_features])
                    lead_index = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"].index(target_lead)
                    y = segments[seg_idx, :, lead_index]
                    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
                        continue
                    self.samples.append((x, y))
                except EOFError:
                    break
                except Exception:
                    continue
        print(f"{features_path.name} ({target_lead}) - Final samples: {len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x, y = self.samples[idx]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

class MLP(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512), nn.LayerNorm(512), nn.LeakyReLU(0.1), nn.Dropout(0.3),
            nn.Linear(512, 256), nn.LayerNorm(256), nn.LeakyReLU(0.1), nn.Dropout(0.3),
            nn.Linear(256, 128), nn.LayerNorm(128), nn.LeakyReLU(0.1), nn.Dropout(0.2),
            nn.Linear(128, 64), nn.LeakyReLU(0.1),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.net(x)

def prepare_ar_datasets(X, Y, point_idx):
    """Prepare autoregressive input for each timepoint during training/testing."""
    n = X.shape[0]
    feat_dim = X.shape[1]
    X_ar = np.zeros((n, feat_dim + 2 * point_idx))
    if point_idx == 0:
        X_ar[:, :feat_dim] = X
        return X_ar
    X_ar[:, :feat_dim] = X
    for k in range(point_idx):
        X_ar[:, feat_dim + 2 * k] = k
        X_ar[:, feat_dim + 2 * k + 1] = Y[:, k]
    return X_ar.astype(np.float32)

def predict_ar(model_list, X, Y_initial=None, inference=False):
    """Sequential autoregressive prediction, with/without Teacher Forcing."""
    n, feat_dim = X.shape
    outputs = np.zeros((n, SEGMENT_LENGTH))
    for t in range(SEGMENT_LENGTH):
        x_ar = np.zeros((n, feat_dim + 2 * t))
        x_ar[:, :feat_dim] = X
        if t > 0:
            if inference:
                for k in range(t):
                    x_ar[:, feat_dim + 2 * k] = k
                    x_ar[:, feat_dim + 2 * k + 1] = outputs[:, k]
            else:
                for k in range(t):
                    x_ar[:, feat_dim + 2 * k] = k
                    x_ar[:, feat_dim + 2 * k + 1] = Y_initial[:, k] if Y_initial is not None else 0
        mdl = model_list[t]
        mdl.eval()
        with torch.no_grad():
            x_ar = x_ar.astype(np.float32)
            pred = mdl(torch.tensor(x_ar, dtype=torch.float32).to(DEVICE)).cpu().numpy().flatten()
        outputs[:, t] = pred
    return outputs

# --- Training & Evaluation Loop ---
with open(REPORT_FILE, "w") as report:
    for lead in TARGET_LEADS:
        print(f"\n[Teacher Forcing] Sequential/AR model training for lead: {lead}...")

        def get_numpy(ds):
            x, y = [], []
            for xb, yb in ds:
                x.append(xb.numpy())
                y.append(yb.numpy())
            return np.stack(x), np.stack(y)

        train_ds = RichECGDataset(SAVE_DIR / "features_train.pkl", SAVE_DIR / "segments_train.npy", lead)
        val_ds = RichECGDataset(SAVE_DIR / "features_val.pkl", SAVE_DIR / "segments_val.npy", lead)
        test_ds = RichECGDataset(SAVE_DIR / "features_test.pkl", SAVE_DIR / "segments_test.npy", lead)
        X_train, Y_train = get_numpy(train_ds)
        X_val, Y_val = get_numpy(val_ds)
        X_test, Y_test = get_numpy(test_ds)

        model_list = []
        best_val_losses = []
        for point_idx in tqdm(range(SEGMENT_LENGTH), desc="Train per-point models (TF)", colour='green'):
            Xtr = prepare_ar_datasets(X_train, Y_train, point_idx)
            Xvl = prepare_ar_datasets(X_val, Y_val, point_idx)
            ytr = Y_train[:, point_idx].reshape(-1, 1).astype(np.float32)
            yvl = Y_val[:, point_idx].reshape(-1, 1).astype(np.float32)

            mlp = MLP(Xtr.shape[1]).to(DEVICE)
            optimizer = torch.optim.Adam(mlp.parameters(), lr=LEARNING_RATE)
            loss_fn = nn.MSELoss()

            train_loader = DataLoader(list(zip(Xtr, ytr)), batch_size=BATCH_SIZE, shuffle=True)
            best_val_loss = float("inf")
            tr_patience = 30
            no_improve = 0
            for epoch in trange(EPOCHS, desc=f"Epochs for point {point_idx}", leave=False, colour='blue'):
                mlp.train()
                train_loss = []
                for xb, yb in train_loader:
                    xb = xb.float().to(DEVICE)
                    yb = yb.float().to(DEVICE)
                    optimizer.zero_grad()
                    out = mlp(xb)
                    loss = loss_fn(out, yb)
                    loss.backward()
                    optimizer.step()
                    train_loss.append(loss.item() * xb.size(0))
                mean_train_loss = np.sum(train_loss) / len(train_loader.dataset)

                mlp.eval()
                with torch.no_grad():
                    pred_val = mlp(torch.tensor(Xvl, dtype=torch.float32).to(DEVICE)).cpu().numpy().flatten()
                val_loss = np.mean((pred_val - yvl.flatten()) ** 2)
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    torch.save(mlp.state_dict(), MODELS_DIR / f"tf_model_point_{point_idx}_{lead}.pt")
                    no_improve = 0
                else:
                    no_improve += 1
                    if no_improve > tr_patience:
                        break
            model_list.append(mlp)
            best_val_losses.append(best_val_loss)

        # Reload best weights per timestep
        for t, mdl in enumerate(model_list):
            mdl.load_state_dict(torch.load(MODELS_DIR / f"tf_model_point_{t}_{lead}.pt"))

        # AR prediction/inference
        y_pred_test = predict_ar(model_list, X_test, inference=True)
        y_pred_val = predict_ar(model_list, X_val, Y_val, inference=False)

        # --- Evaluation ---
        rmse = np.sqrt(mean_squared_error(Y_test, y_pred_test))
        r2 = r2_score(Y_test, y_pred_test)
        pearson_corr = np.mean([
            pearsonr(Y_test[:, i], y_pred_test[:, i])[0]
            for i in range(SEGMENT_LENGTH) if np.std(Y_test[:, i]) > 0
        ])
        report.write(f"\nEvaluation for Lead {lead} (Teacher Forcing AR):\n")
        report.write(f"RMSE: {rmse:.4f}\nR^2: {r2:.4f}\nPearson Correlation: {pearson_corr:.4f}\n")

        # Metrics per timepoint
        per_point_metrics = []
        for t in range(SEGMENT_LENGTH):
            rmse_t = np.sqrt(np.mean((Y_test[:, t] - y_pred_test[:, t]) ** 2))
            r2_t = r2_score(Y_test[:, t], y_pred_test[:, t])
            p_t = pearsonr(Y_test[:, t], y_pred_test[:, t])[0] if np.std(Y_test[:, t]) > 0 else 0
            per_point_metrics.append([t, rmse_t, r2_t, p_t])
        np.savetxt(
            str(OUTPUT_DIR / f"per_point_metrics_{lead}.csv"),
            per_point_metrics, delimiter=',',
            header="point,RMSE,R2,Pearson", comments=''
        )

# --- Visualization: First 10 Prediction Comparisons ---
NUM_SAMPLES_TO_PLOT = 10
for lead in TARGET_LEADS:
    print(f"Generating side-by-side plots for {lead} (TF-AR)...")
    fig, axes = plt.subplots(2, 5, figsize=(18, 6))
    fig.suptitle(
        f"Lead {lead}: First {NUM_SAMPLES_TO_PLOT} Test Predictions [Teacher Forcing AR]", fontsize=14
    )
    for i in range(NUM_SAMPLES_TO_PLOT):
        gt = Y_test[i]
        pred = y_pred_test[i]
        ax = axes[i // 5, i % 5]
        ax.plot(gt, label="Actual", color="blue")
        ax.plot(pred, label="Predicted", color="orange", linestyle="--")
        ax.set_title(f"Sample {i+1}", fontsize=10)
        ax.set_xlim(0, SEGMENT_LENGTH)
        ax.grid(True)
        if i == 0:
            ax.legend(loc="upper right", fontsize=8)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(PLOTS_DIR / f"{lead}_comparison_first_{NUM_SAMPLES_TO_PLOT}_tfar.png")
    plt.close()
