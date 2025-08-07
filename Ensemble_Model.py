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
from skimage.metrics import structural_similarity as ssim
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
TARGET_LEADS = ["II", "III", "aVL", "aVR", "aVF", "V1", "V3", "V4", "V5"]
SEGMENT_LENGTH = 128
BATCH_SIZE = 32
EPOCHS = 300
LEARNING_RATE = 1e-3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LAMBDA_COSINE = 0.5 # وزن الكوساين في التابع

# --- Cosine Similarity Function ---
def cosine_similarity(y_pred, y_true):
    # y_pred, y_true: [batch_size, seq_len]
    y_pred_norm = y_pred / (y_pred.norm(dim=1, keepdim=True) + 1e-8)
    y_true_norm = y_true / (y_true.norm(dim=1, keepdim=True) + 1e-8)
    sim = (y_pred_norm * y_true_norm).sum(dim=1)
    return sim.mean()

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
                    full_segment_inputs = np.concatenate(full_segment_inputs)

                    advanced_features_inputs = np.concatenate([rec["features"][lead] for lead in INPUT_LEADS])
                    metadata = rec.get("metadata", {})
                    if isinstance(metadata, dict):
                        meta_values = np.array(list(metadata.values()), dtype=np.float32)
                    else:
                        meta_values = np.zeros(1, dtype=np.float32)
                    x = np.concatenate([full_segment_inputs])
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

# --- LSTM Model ---
class LSTMNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=1):
        super().__init__()
        self.input_dim = input_dim 
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.output_dim = output_dim

        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim,
                            num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # x shape: (batch_size, input_dim) --> we need to reshape to (batch_size, seq_len, feature_dim)
        # هنا نعتبر أن لدينا سلسلة بطول واحد (seq_len = 1) ونضع كامل x كـ feature
        x = x.unsqueeze(1)  # (batch_size, 1, input_dim)
        output, (hn, cn) = self.lstm(x)  # hn shape: (num_layers, batch_size, hidden_dim)
        hn = hn[-1]  # خذ الطبقة الأخيرة فقط
        out = self.fc(hn)
        return out

def compute_ssim_batch(y_true, y_pred):
    scores = []
    for i in range(y_true.shape[0]):
        score = ssim(y_true[i], y_pred[i], data_range=y_true[i].max() - y_true[i].min())
        scores.append(score)
    return np.mean(scores)

# --- Training Loop ---
with open(REPORT_FILE, "w") as report:
    for lead in TARGET_LEADS:
        print(f"\n🔧 Training Stacking model for lead: {lead}...")
        train_ds = RichECGDataset(SAVE_DIR / "features_train.pkl", SAVE_DIR / "segments_train.npy", lead)
        val_ds = RichECGDataset(SAVE_DIR / "features_val.pkl", SAVE_DIR / "segments_val.npy", lead)
        test_ds = RichECGDataset(SAVE_DIR / "features_test.pkl", SAVE_DIR / "segments_test.npy", lead)
        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)

        input_dim = len(train_ds[0][0])
        output_dim = SEGMENT_LENGTH

        model = LSTMNet(input_dim=input_dim, hidden_dim=256, output_dim=output_dim).to(DEVICE)
        mse_loss_fn = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

        best_val_loss = float("inf")
        patience = 15
        counter = 0
        best_model_state = None

        for epoch in range(EPOCHS):
            model.train()
            total_loss = 0.0
            total_cosine = 0.0
            for xb, yb in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} - Lead {lead}", leave=False):
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                pred = model(xb)
                mse_loss = mse_loss_fn(pred, yb)
                cos_sim = cosine_similarity(pred, yb)
                loss = mse_loss + LAMBDA_COSINE * (1 - cos_sim)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * xb.size(0)
                total_cosine += cos_sim.item() * xb.size(0)
            avg_train_loss = total_loss / len(train_ds)
            avg_train_cosine = total_cosine / len(train_ds)

            model.eval()
            val_loss = 0.0
            val_cosine = 0.0
            with torch.no_grad():
                for xb, yb in val_loader:
                    xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                    pred = model(xb)
                    mse_loss = mse_loss_fn(pred, yb)
                    cos_sim = cosine_similarity(pred, yb)
                    loss = mse_loss + LAMBDA_COSINE * (1 - cos_sim)
                    val_loss += loss.item() * xb.size(0)
                    val_cosine += cos_sim.item() * xb.size(0)
            avg_val_loss = val_loss / len(val_ds)
            avg_val_cosine = val_cosine / len(val_ds)

            print(f"Epoch {epoch+1} - Train Loss: {avg_train_loss:.4f} - Val Loss: {avg_val_loss:.4f} "
                  f"- Train CosSim: {avg_train_cosine:.4f} - Val CosSim: {avg_val_cosine:.4f}")

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

        torch.save(model.state_dict(), MODELS_DIR / f"LSTM_model_{lead}.pt")

        def collect_predictions(dataset):
            xs, ys, LSTM_out = [], [], []
            model.eval()
            X_xgb, Y_xgb = [], []
            with torch.no_grad():
                for xb, yb in DataLoader(dataset, batch_size=BATCH_SIZE):
                    xs.append(xb.numpy())
                    ys.append(yb.numpy())
                    xb_gpu = xb.to(DEVICE)
                    LSTM_preds = model(xb_gpu).cpu().numpy()
                    LSTM_out.append(LSTM_preds)
                    X_xgb.extend(xb.numpy())
                    Y_xgb.extend(yb.numpy())
            xgb_model = XGBRegressor(n_estimators=100, max_depth=4, verbosity=0)
            xgb_model.fit(np.array(X_xgb), np.array(Y_xgb))
            xgb_preds = xgb_model.predict(np.array(X_xgb))
            meta_X = np.hstack([np.vstack(LSTM_out), xgb_preds])
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
        ssim_score = compute_ssim_batch(meta_y_test, meta_pred)
        # Cosine similarity على بيانات الاختبار
        cos_test = np.mean([
            np.dot(meta_y_test[i], meta_pred[i]) / (np.linalg.norm(meta_y_test[i]) * np.linalg.norm(meta_pred[i]) + 1e-8)
            for i in range(meta_y_test.shape[0])
        ])
        print(f"\nLead {lead} Evaluation Summary:")
        print(f"  RMSE             = {rmse:.4f}")
        print(f"  R^2              = {r2:.4f}")
        print(f"  Pearson Corr     = {pearson_corr:.4f}")
        print(f"  SSIM             = {ssim_score:.4f}")
        print(f"  Cosine Similarity= {cos_test:.4f}")

        # RMSE per point
        rmse_per_point = np.sqrt(np.mean((meta_y_test - meta_pred) ** 2, axis=0)).tolist()
        report.write(f"\nEvaluation for Lead {lead}:\n")
        report.write(f"RMSE: {rmse:.4f}\n")
        report.write(f"R^2: {r2:.4f}\n")
        report.write(f"Pearson Correlation: {pearson_corr:.4f}\n")
        report.write(f"SSIM: {ssim_score:.4f}\n")
        report.write(f"Cosine Similarity: {cos_test:.4f}\n")
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
        xs, ys = [], []
        for i in range(10):
            x, y = train_ds[i]
            xs.append(x.unsqueeze(0))
            ys.append(y.numpy())
        xs_tensor = torch.cat(xs).to(DEVICE)
        with torch.no_grad():
            LSTM_out = model(xs_tensor).cpu().numpy()
            xgb_out = xgb_model.predict(xs_tensor.cpu().numpy())
            meta_input = np.hstack([LSTM_out, xgb_out])
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
