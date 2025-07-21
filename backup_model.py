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
from sklearn.ensemble import RandomForestRegressor
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

INPUT_LEADS = ["I", "II", "V2"]
TARGET_LEADS = ["III"]
SEGMENT_LENGTH = 80
BATCH_SIZE = 32
EPOCHS = 300
LEARNING_RATE = 1e-3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- Metadata Encoder ---
def encode_metadata(meta_dict):
    age = meta_dict.get("age", 0)
    sex = 1 if meta_dict.get("sex", "").lower() == "male" else 0
    axis = meta_dict.get("heart_axis", "").lower()
    axis_encoded = [0, 0, 0]
    if axis == "normal": axis_encoded[0] = 1
    elif axis == "left axis deviation": axis_encoded[1] = 1
    elif axis == "right axis deviation": axis_encoded[2] = 1
    return [age, sex] + axis_encoded

# --- Dataset Class ---
class RichECGDataset(Dataset):
    def __init__(self, features_path, segments_path, target_lead):
        self.samples = []
        segments = np.load(segments_path)[:, :SEGMENT_LENGTH, :]

        with open(features_path, "rb") as f:
            while True:
                try:
                    rec = pickle.load(f)
                    record_id = rec.get("record_id", "UNKNOWN")                    

                    # تحقق من segment_index
                    seg_idx = rec.get("segment_index")
                    if seg_idx is None:
                        print(f"⛔ {record_id}: segment_index is None → skipped")
                        continue
                    if seg_idx >= segments.shape[0]:
                        print(f"⛔ {record_id}: segment_index {seg_idx} out of range → skipped")
                        continue

                    # تحقق من توفر الليدات المطلوبة
                    all_leads = INPUT_LEADS + [target_lead]
                    missing_feats = [lead for lead in all_leads if lead not in rec["features"]]
                    if missing_feats:
                        print(f"⛔ {record_id}: missing features for leads {missing_feats} → skipped")
                        continue

                    # تحقق من القيم غير الصالحة
                    bad_leads = [lead for lead in all_leads if not np.all(np.isfinite(rec["features"][lead]))]
                    if bad_leads:
                        print(f"⛔ {record_id}: non-finite values in features of leads {bad_leads} → skipped")
                        continue

                    # بناء المدخلات
                    feat = np.concatenate([rec["features"][lead] for lead in INPUT_LEADS])

                    amps = []
                    times = []
                    qrs_area = []
                    qrs_dur = []
                    for lead in INPUT_LEADS:
                        wave = rec["waves"].get(lead, {})
                        amps.extend([wave.get(f"{w}_amp", 0) or 0 for w in ['P', 'Q', 'R', 'S', 'T']])
                        times.extend([wave.get(f"{w}time", 0) or 0 for w in ['P', 'Q', 'R', 'S', 'T']])
                        qrs_area.append(wave.get("QRS_area", 0) or 0)
                        qrs_dur.append(wave.get("QRS_duration", 0) or 0)

                    # Interval features
                    intervals = []
                    for lead in INPUT_LEADS:
                        lead_intervals = rec["intervals"].get(lead, {})
                        for key in ['PR', 'QT', 'ST', 'RR']:
                            val = lead_intervals.get(key)
                            if isinstance(val, list) and len(val) > 0:
                                mean_val = np.mean(val)
                                if not np.isfinite(mean_val):
                                    intervals.append(0)
                                else:
                                    intervals.append(mean_val)
                            else:
                                intervals.append(0)


                    meta_features = encode_metadata(rec["metadata"])
                    x = np.concatenate([feat, amps, times, qrs_area, qrs_dur, intervals, meta_features])

                    # بناء المخرجات
                    lead_index = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"].index(target_lead)
                    y = segments[seg_idx, :, lead_index]

                    # تحقق نهائي من صلاحية x و y
                    if not np.all(np.isfinite(x)):
                        print(f"⛔ {record_id}: non-finite values in input vector x → skipped")
                        continue
                    if not np.all(np.isfinite(y)):
                        print(f"⛔ {record_id}: non-finite values in output y → skipped")
                        continue

                    self.samples.append((x, y))

                except EOFError:
                    break
                except Exception as e:
                    print(f"⚠️ خطأ غير متوقع في سجل {record_id}: {e}")
                    continue

        print(f"{features_path.name} ({target_lead}) - عدد العينات النهائية: {len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x, y = self.samples[idx]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)


# --- MLP Model ---
class MLP(nn.Module):
    def __init__(self, input_dim, output_dim):
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

            nn.Linear(128, 64),
            nn.LeakyReLU(0.1),

            nn.Linear(64, output_dim)
        )

    def forward(self, x):
        return self.net(x)

# --- Training and Evaluation Loop ---
with open(REPORT_FILE, "w") as report:
    for lead in TARGET_LEADS:
        print(f"\nTraining Stacking model for lead: {lead}...")

        train_ds = RichECGDataset(SAVE_DIR / "features_train.pkl", SAVE_DIR / "segments_train.npy", lead)
        val_ds = RichECGDataset(SAVE_DIR / "features_val.pkl", SAVE_DIR / "segments_val.npy", lead)
        test_ds = RichECGDataset(SAVE_DIR / "features_test.pkl", SAVE_DIR / "segments_test.npy", lead)

        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
        input_dim = len(train_ds[0][0])
        output_dim = SEGMENT_LENGTH

        mlp = MLP(input_dim, output_dim).to(DEVICE)
        loss_fn = nn.MSELoss()
        optimizer = torch.optim.Adam(mlp.parameters(), lr=LEARNING_RATE)

        PATIENCE = 80
        best_val_loss = float("inf")
        epochs_no_improve = 0

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

            # 🔍 Validation Loss
            mlp.eval()
            val_loss = 0.0
            with torch.no_grad():
                for xb, yb in DataLoader(val_ds, batch_size=BATCH_SIZE):
                    xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                    pred = mlp(xb)
                    val_loss += loss_fn(pred, yb).item() * xb.size(0)
            avg_val_loss = val_loss / len(val_ds)

            print(f"Epoch {epoch+1} - Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                epochs_no_improve = 0
                torch.save(mlp.state_dict(), MODELS_DIR / f"mlp_model_{lead}.pt")
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= PATIENCE:
                    print(f"Early stopping at epoch {epoch+1} (no improvement in {PATIENCE} epochs)")
                    break


        torch.save(mlp.state_dict(), MODELS_DIR / f"mlp_model_{lead}.pt")

        def collect_predictions(dataset):
            xs, ys, mlp_out = [], [], []
            mlp.eval()
            X_xgb, Y_xgb = [], []
            with torch.no_grad():
                for xb, yb in DataLoader(dataset, batch_size=BATCH_SIZE):
                    xs.append(xb.numpy()); ys.append(yb.numpy())
                    mlp_preds = mlp(xb.to(DEVICE)).cpu().numpy()
                    mlp_out.append(mlp_preds)
                    X_xgb.extend(xb.numpy()); Y_xgb.extend(yb.numpy())
            xgb_model = XGBRegressor(n_estimators=100, max_depth=4, verbosity=0)
            xgb_model.fit(np.array(X_xgb), np.array(Y_xgb))
            xgb_preds = xgb_model.predict(np.array(X_xgb))
            meta_X = np.hstack([np.vstack(mlp_out), xgb_preds])
            meta_y = np.vstack(ys)
            return meta_X, meta_y, xgb_model

        meta_X_train, meta_y_train, xgb_model = collect_predictions(train_ds)
        meta_X_test, meta_y_test, _ = collect_predictions(test_ds)

        meta_model = Ridge(alpha=1.0)
        meta_model.fit(meta_X_train, meta_y_train)
        meta_pred = meta_model.predict(meta_X_test)

        # Save models
        with open(MODELS_DIR / f"ridge_model_{lead}.pkl", "wb") as f:
            pickle.dump(meta_model, f)
        with open(MODELS_DIR / f"xgb_model_{lead}.pkl", "wb") as f:
            pickle.dump(xgb_model, f)

        # Evaluation
        rmse = np.sqrt(mean_squared_error(meta_y_test, meta_pred))
        r2 = r2_score(meta_y_test, meta_pred)
        pearson_corr = np.mean([pearsonr(meta_y_test[:, i], meta_pred[:, i])[0]
                                 for i in range(SEGMENT_LENGTH) if np.std(meta_y_test[:, i]) > 0])

        report.write(f"\nEvaluation for Lead {lead}:\n")
        report.write(f"RMSE: {rmse:.4f}\nR^2: {r2:.4f}\nPearson Correlation: {pearson_corr:.4f}\n")

# --- Visualization: First 10 Prediction Comparisons ---

NUM_SAMPLES_TO_PLOT = 10

for lead in TARGET_LEADS:
    print(f"Generating side-by-side plots for {lead}...")

    # Load models
    mlp = MLP(input_dim, SEGMENT_LENGTH).to(DEVICE)
    mlp.load_state_dict(torch.load(MODELS_DIR / f"mlp_model_{lead}.pt"))
    mlp.eval()

    with open(MODELS_DIR / f"xgb_model_{lead}.pkl", "rb") as f:
        xgb_model = pickle.load(f)
    with open(MODELS_DIR / f"ridge_model_{lead}.pkl", "rb") as f:
        ridge_model = pickle.load(f)

    # Load test dataset
    dataset = RichECGDataset(SAVE_DIR / "features_test.pkl", SAVE_DIR / "segments_test.npy", lead)

    # Plotting
    fig, axes = plt.subplots(2, 5, figsize=(18, 6))
    fig.suptitle(f"Lead {lead}: First {NUM_SAMPLES_TO_PLOT} Test Predictions", fontsize=14)

    for i in range(NUM_SAMPLES_TO_PLOT):
        x, y = dataset[i]
        x_tensor = x.unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            mlp_out = mlp(x_tensor).cpu().numpy()
            xgb_out = xgb_model.predict(x.unsqueeze(0).numpy())
            meta_input = np.hstack([mlp_out, xgb_out])
            pred = ridge_model.predict(meta_input)[0]

        ax = axes[i // 5, i % 5]
        ax.plot(y.numpy(), label="Actual", color="blue")
        ax.plot(pred, label="Predicted", color="orange", linestyle="--")
        ax.set_title(f"Sample {i+1}", fontsize=10)
        ax.set_xlim(0, SEGMENT_LENGTH)
        ax.grid(True)

        if i == 0:
            ax.legend(loc="upper right", fontsize=8)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(PLOTS_DIR / f"{lead}_comparison_first_{NUM_SAMPLES_TO_PLOT}.png")
    plt.close()


