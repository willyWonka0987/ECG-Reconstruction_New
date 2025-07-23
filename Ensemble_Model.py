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
TARGET_LEADS = ["II", "V1", "V3", "V4", "V5"]
SEGMENT_LENGTH = 80
BATCH_SIZE = 32
EPOCHS = 300
LEARNING_RATE = 1e-3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- Dataset Class ---
class RichECGDataset(Dataset):
    def __init__(self, features_path, segments_path, target_lead):
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
                    full_segment_inputs = np.stack(full_segment_inputs, axis=0)  # Shape (3, SEGMENT_LENGTH)

                    features_inputs = np.concatenate([rec["features"][lead] for lead in INPUT_LEADS])
                    # Note: We do NOT concatenate features and segment in x. Instead, keep both separately:
                    x = {
                        "features": features_inputs.astype(np.float32),
                        "segments": full_segment_inputs.astype(np.float32)
                    }

                    lead_index = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"].index(target_lead)
                    y = segments[seg_idx, :, lead_index]
                    if np.all(np.isfinite(features_inputs)) and np.all(np.isfinite(full_segment_inputs)) and np.all(np.isfinite(y)):
                        self.samples.append((x, y))
                except EOFError:
                    break
        print(f"{features_path.name} ({target_lead}) - num of samples : {len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x, y = self.samples[idx]
        features = torch.tensor(x["features"], dtype=torch.float32)
        segments = torch.tensor(x["segments"], dtype=torch.float32)
        target = torch.tensor(y, dtype=torch.float32)
        return {"features": features, "segments": segments}, target

# --- MLP Branch ---
class FeatureMLP(nn.Module):
    def __init__(self, input_dim, emb_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.LayerNorm(512),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.2),
            nn.Linear(128, emb_dim),
            nn.LeakyReLU(0.1)
        )

    def forward(self, x):
        return self.net(x)

# --- CNN Branch ---
class SegmentCNN(nn.Module):
    def __init__(self, in_channels, emb_dim=128):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.1),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.1),
            nn.AdaptiveAvgPool1d(8),  # Out: (batch, 128, 8)
            nn.Flatten(),             # Out: (batch, 128*8)
            nn.Linear(128*8, emb_dim),
            nn.LeakyReLU(0.1)
        )

    def forward(self, x):
        # x shape: (batch, channels=3, length=SEGMENT_LENGTH)
        return self.cnn(x)

# --- Fusion Model ---
class FusionModel(nn.Module):
    def __init__(self, feature_input_dim, segment_channels, segment_length, output_dim):
        super().__init__()
        emb_dim = 128  # Embedding size for each branch
        self.feature_branch = FeatureMLP(feature_input_dim, emb_dim=emb_dim)
        self.segment_branch = SegmentCNN(segment_channels, emb_dim=emb_dim)
        self.classifier = nn.Sequential(
            nn.Linear(2 * emb_dim, 128),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.2),
            nn.Linear(128, output_dim)
        )

    def forward(self, X):
        features = X["features"]
        segments = X["segments"]
        # segments: (batch, 3, SEGMENT_LENGTH)
        feature_emb = self.feature_branch(features)
        segment_emb = self.segment_branch(segments)
        concat = torch.cat([feature_emb, segment_emb], dim=1)
        out = self.classifier(concat)
        return out

# --- Training Loop ---
with open(REPORT_FILE, "w") as report:
    for lead in TARGET_LEADS:
        print(f"\n🔧 Training Stacking model for lead: {lead}...")
        train_ds = RichECGDataset(SAVE_DIR / "features_train.pkl", SAVE_DIR / "segments_train.npy", lead)
        val_ds = RichECGDataset(SAVE_DIR / "features_val.pkl", SAVE_DIR / "segments_val.npy", lead)
        test_ds = RichECGDataset(SAVE_DIR / "features_test.pkl", SAVE_DIR / "segments_test.npy", lead)

        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)

        feature_dim = train_ds[0][0]["features"].shape[0]
        segment_channels = train_ds[0][0]["segments"].shape[0]
        segment_length = train_ds[0][0]["segments"].shape[1]
        output_dim = SEGMENT_LENGTH

        model = FusionModel(feature_dim, segment_channels, segment_length, output_dim).to(DEVICE)
        loss_fn = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

        best_val_loss = float("inf")
        patience = 15
        counter = 0
        best_model_state = None

        for epoch in range(EPOCHS):
            model.train()
            total_loss = 0.0
            for batch_X, batch_y in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} - Lead {lead}", leave=False):
                batch_X = {k:v.to(DEVICE) for k, v in batch_X.items()}
                batch_y = batch_y.to(DEVICE)
                pred = model(batch_X)
                loss = loss_fn(pred, batch_y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * batch_y.size(0)
            avg_train_loss = total_loss / len(train_ds)

            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    batch_X = {k:v.to(DEVICE) for k, v in batch_X.items()}
                    batch_y = batch_y.to(DEVICE)
                    pred = model(batch_X)
                    loss = loss_fn(pred, batch_y)
                    val_loss += loss.item() * batch_y.size(0)
            avg_val_loss = val_loss / len(val_ds)

            print(f"Epoch {epoch+1} - Train Loss: {avg_train_loss:.4f} - Val Loss: {avg_val_loss:.4f}")
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_model_state = model.state_dict()
                counter = 0
            else:
                counter += 1
                if counter >= patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break

        if best_model_state is not None:
            model.load_state_dict(best_model_state)

        torch.save(model.state_dict(), MODELS_DIR / f"fusion_model_{lead}.pt")

        def collect_predictions(dataset):
            Xs, ys, model_out = [], [], []
            model.eval()
            X_xgb, Y_xgb = [], []
            with torch.no_grad():
                for batch_X, batch_y in DataLoader(dataset, batch_size=BATCH_SIZE):
                    # For xgboost, flatten features + segments for compatibility
                    feats = batch_X["features"].numpy()
                    segs = batch_X["segments"].numpy().reshape(len(feats), -1)
                    feats_and_segs = np.concatenate([feats, segs], axis=1)
                    Xs.append(batch_X)
                    ys.append(batch_y.numpy())
                    batch_X_cuda = {k: v.to(DEVICE) for k, v in batch_X.items()}
                    model_preds = model(batch_X_cuda).cpu().numpy()
                    model_out.append(model_preds)
                    X_xgb.extend(feats_and_segs)
                    Y_xgb.extend(batch_y.numpy())
            xgb_model = XGBRegressor(n_estimators=100, max_depth=4, verbosity=0)
            xgb_model.fit(np.array(X_xgb), np.array(Y_xgb))
            xgb_preds = xgb_model.predict(np.array(X_xgb))
            meta_X = np.hstack([np.vstack(model_out), xgb_preds])
            meta_y = np.vstack(ys)
            return meta_X, meta_y, xgb_model

        meta_X_train, meta_y_train, xgb_model = collect_predictions(train_ds)
        meta_X_test, meta_y_test, _ = collect_predictions(test_ds)

        meta_model = Ridge(alpha=1.0)
        meta_model.fit(meta_X_train, meta_y_train)
        meta_pred = meta_model.predict(meta_X_test)

        with open(MODELS_DIR / f"ridge_model_{lead}.pkl", "wb") as f:
            pickle.dump(meta_model, f)
        with open(MODELS_DIR / f"xgb_model_{lead}.pkl", "wb") as f:
            pickle.dump(xgb_model, f)

        rmse = np.sqrt(mean_squared_error(meta_y_test, meta_pred))
        r2 = r2_score(meta_y_test, meta_pred)
        pearson_corr = np.mean([
            pearsonr(meta_y_test[:, i], meta_pred[:, i])[0]
            for i in range(SEGMENT_LENGTH)
            if np.std(meta_y_test[:, i]) > 0
        ])

        # RMSE per point
        rmse_per_point = np.sqrt(np.mean((meta_y_test - meta_pred) ** 2, axis=0)).tolist()
        report.write(f"\nEvaluation for Lead {lead}:\n")
        report.write(f"RMSE: {rmse:.4f}\n")
        report.write(f"R^2: {r2:.4f}\n")
        report.write(f"Pearson Correlation: {pearson_corr:.4f}\n")
        report.write(f"RMSE per point (length {SEGMENT_LENGTH}):\n")
        report.write(", ".join(f"{v:.6f}" for v in rmse_per_point) + "\n")

        # Plot RMSE per point
        plt.figure(figsize=(10, 4))
        plt.plot(rmse_per_point, marker='o')
        plt.title(f"RMSE per point - Lead {lead}")
        plt.xlabel("Point index")
        plt.ylabel("RMSE")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(RMSE_PLOTS_DIR / f"rmse_per_point_{lead}.png")
        plt.close()

        # Sample predictions plot
        Xs, ys = [], []
        for i in range(10):
            x, y = train_ds[i]
            Xs.append({k: v.unsqueeze(0) for k, v in x.items()})
            ys.append(y.numpy())
        feat_X = {
            "features": torch.cat([d["features"] for d in Xs], dim=0),
            "segments": torch.cat([d["segments"] for d in Xs], dim=0)
        }
        for k in feat_X:
            feat_X[k] = feat_X[k].to(DEVICE)
        with torch.no_grad():
            model_out = model(feat_X).cpu().numpy()
            # For XGB input compatibility as above
            feats = feat_X["features"].cpu().numpy()
            segs = feat_X["segments"].cpu().numpy().reshape(len(feats), -1)
            feats_and_segs = np.concatenate([feats, segs], axis=1)
            xgb_out = xgb_model.predict(feats_and_segs)
            meta_input = np.hstack([model_out, xgb_out])
            preds = meta_model.predict(meta_input)
        for i in range(10):
            plt.figure(figsize=(8, 4))
            plt.plot(ys[i], label="True", linewidth=2)
            plt.plot(preds[i], label="Predicted", linestyle="--")
            plt.title(f"Lead {lead} - Sample {i}")
            plt.legend()
            plt.tight_layout()
            plt.savefig(PLOTS_DIR / f"{lead}_sample_{i}.png")
            plt.close()
