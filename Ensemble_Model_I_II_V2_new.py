# ======================= IMPORTS =======================
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import pickle
import numpy as np
from pathlib import Path
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import pearsonr
from sklearn.linear_model import Ridge
from xgboost import XGBRegressor
import shap
import json
import pandas as pd
from tqdm import tqdm, trange

# ======================= CONFIG =======================
SAVE_DIR = Path("RichECG_Datasets")
OUTPUT_DIR = Path("Stacked_Model_Results_Ensemble")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR = OUTPUT_DIR / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_FILE = OUTPUT_DIR / "progressive_results.json"
SUMMARY_FILE = OUTPUT_DIR / "all_metrics.txt"
UNCERTAINTY_FILE = OUTPUT_DIR / "uncertainty_results.csv"

INPUT_LEADS = ["I", "II", "V2"]
TARGET_LEADS = ["III"]
SEGMENT_LENGTH = 80
BATCH_SIZE = 32
EPOCHS_PER_ITER = 300
PATIENCE = 20
LEARNING_RATE = 1e-3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
POINTS_PER_ITER = 10
STACK_ITERS = 8
MC_DROPOUT_SAMPLES = 30  # عدد مرات التكرار لحساب الثقة

# ======================= HELPERS =======================
def encode_metadata(meta_dict):
    age = meta_dict.get("age", 0)
    sex = 1 if meta_dict.get("sex", "").lower() == "male" else 0
    axis = meta_dict.get("heart_axis", "").lower()
    axis_encoded = [0, 0, 0]
    if axis == "normal": axis_encoded[0] = 1
    elif axis == "left axis deviation": axis_encoded[1] = 1
    elif axis == "right axis deviation": axis_encoded[2] = 1
    return [age, sex] + axis_encoded

def evaluate_pointwise_metrics(y_true, y_pred):
    rmse_list, r2_list, pearson_list = [], [], []
    for i in range(y_true.shape[1]):
        yt, yp = y_true[:, i], y_pred[:, i]
        if np.std(yt) < 1e-6: continue
        rmse_list.append(np.sqrt(mean_squared_error(yt, yp)))
        r2_list.append(r2_score(yt, yp))
        try:
            pearson_list.append(pearsonr(yt, yp)[0])
        except:
            pearson_list.append(0)
    return np.array(rmse_list), np.array(r2_list), np.array(pearson_list)

def enable_dropout(model):
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            m.train()

# ======================= DATASET =======================
class RichECGDataset(Dataset):
    def __init__(self, features_path, segments_path, target_lead):
        self.samples = []
        segments = np.load(segments_path)[:, :SEGMENT_LENGTH, :]
        with open(features_path, "rb") as f:
            pkl_data = []
            print(f"Loading data from {features_path} ...")
            while True:
                try:
                    rec = pickle.load(f)
                    pkl_data.append(rec)
                except EOFError:
                    break
            for rec in tqdm(pkl_data, desc="Processing samples", unit="sample"):
                seg_idx = rec.get("segment_index")
                if seg_idx is None or seg_idx >= segments.shape[0]: continue
                all_leads = INPUT_LEADS + [target_lead]
                if any(lead not in rec["features"] for lead in all_leads): continue
                if any(not np.all(np.isfinite(rec["features"][lead])) for lead in all_leads): continue
                feat = np.concatenate([rec["features"][lead] for lead in INPUT_LEADS])
                amps, times, qrs_area, qrs_dur = [], [], [], []
                for lead in INPUT_LEADS:
                    wave = rec["waves"].get(lead, {})
                    amps += [wave.get(f"{w}_amp", 0) or 0 for w in ['P', 'Q', 'R', 'S', 'T']]
                    times += [wave.get(f"{w}time", 0) or 0 for w in ['P', 'Q', 'R', 'S', 'T']]
                    qrs_area.append(wave.get("QRS_area", 0) or 0)
                    qrs_dur.append(wave.get("QRS_duration", 0) or 0)
                intervals = []
                for lead in INPUT_LEADS:
                    intv = rec["intervals"].get(lead, {})
                    for key in ['PR', 'QT', 'ST', 'RR']:
                        val = intv.get(key)
                        val = np.mean(val) if isinstance(val, list) and len(val) > 0 else 0
                        intervals.append(val if np.isfinite(val) else 0)
                meta = encode_metadata(rec["metadata"])
                x = np.concatenate([feat, amps, times, qrs_area, qrs_dur, intervals, meta])
                lead_index = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"].index(target_lead)
                y = segments[seg_idx, :, lead_index]
                if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)): continue
                self.samples.append((x, y))

    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        x, y = self.samples[idx]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

# ======================= MLP Backbone =======================
class MLPBackbone(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(input_dim, 512), nn.LayerNorm(512), nn.LeakyReLU(), nn.Dropout(0.3),
            nn.Linear(512, 256), nn.LayerNorm(256), nn.LeakyReLU(), nn.Dropout(0.3),
            nn.Linear(256, 128), nn.LayerNorm(128), nn.LeakyReLU(), nn.Dropout(0.2),
            nn.Linear(128, 64), nn.LeakyReLU()
        )
    def forward(self, x): return self.shared(x)

class StackedOutputHead(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.head = nn.Linear(input_dim, output_dim)
    def forward(self, x): return self.head(x)

# ======================= MAIN =======================
if __name__ == "__main__":
    for lead in TARGET_LEADS:
        print(f"\n🔧 Training for lead: {lead}")
        train_ds = RichECGDataset(SAVE_DIR / "features_train.pkl", SAVE_DIR / "segments_train.npy", lead)
        val_ds = RichECGDataset(SAVE_DIR / "features_val.pkl", SAVE_DIR / "segments_val.npy", lead)
        test_ds = RichECGDataset(SAVE_DIR / "features_test.pkl", SAVE_DIR / "segments_test.npy", lead)

        all_indices = list(range(SEGMENT_LENGTH))
        known_points = []
        results_log = []
        uncertainty_log = []
        predicted_points = {i: np.zeros(SEGMENT_LENGTH) for i in range(len(train_ds))}

        model_input_dim = len(train_ds[0][0]) + SEGMENT_LENGTH
        mlp = MLPBackbone(model_input_dim).to(DEVICE)
        loss_fn = nn.MSELoss()

        for iteration in trange(STACK_ITERS, desc="Stacked Iterations", unit="iter"):
            print(f"\n🔁 Iteration {iteration+1}/{STACK_ITERS}")
            target_mask = [i for i in all_indices if i not in known_points]

            def build_XY(dataset, preds_dict):
                X, Y = [], []
                for idx, (x, y) in enumerate(dataset):
                    dyn = preds_dict[idx][known_points] if known_points else np.zeros(0)
                    padded_dyn = np.concatenate([dyn, np.zeros(SEGMENT_LENGTH - len(known_points))])
                    x_concat = np.concatenate([x.numpy(), padded_dyn])
                    X.append(x_concat)
                    Y.append(y.numpy()[target_mask])
                return np.array(X), np.array(Y)

            X_train, Y_train = build_XY(train_ds, predicted_points)
            output_dim = Y_train.shape[1]
            head = StackedOutputHead(64, output_dim).to(DEVICE)
            model = nn.Sequential(mlp, head)
            optimizer = torch.optim.Adam(list(head.parameters()), lr=LEARNING_RATE)
            X_val, Y_val = build_XY(val_ds, predicted_points)

            best_val_loss = float('inf')
            wait = 0
            for epoch in trange(EPOCHS_PER_ITER, desc="Epochs", leave=False):
                mlp.train()
                for i in range(0, len(X_train), BATCH_SIZE):
                    xb = torch.tensor(X_train[i:i+BATCH_SIZE], dtype=torch.float32).to(DEVICE)
                    yb = torch.tensor(Y_train[i:i+BATCH_SIZE], dtype=torch.float32).to(DEVICE)
                    optimizer.zero_grad()
                    pred = model(xb)
                    loss = loss_fn(pred, yb)
                    loss.backward()
                    optimizer.step()

                mlp.eval()
                with torch.no_grad():
                    val_preds = model(torch.tensor(X_val, dtype=torch.float32).to(DEVICE)).cpu().numpy()
                val_loss = mean_squared_error(Y_val, val_preds)
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    wait = 0
                else:
                    wait += 1
                    if wait >= PATIENCE:
                        break

            mlp.eval()
            with torch.no_grad():
                feats_val = mlp(torch.tensor(X_val, dtype=torch.float32).to(DEVICE)).cpu().numpy()
                feats_train = mlp(torch.tensor(X_train, dtype=torch.float32).to(DEVICE)).cpu().numpy()
            xgb = XGBRegressor(n_estimators=100, max_depth=6, objective='reg:squarederror', n_jobs=-1)
            xgb.fit(X_train, Y_train)
            xgb_val = xgb.predict(X_val)

            ridge = Ridge(alpha=1.0)
            ridge.fit(np.concatenate([feats_val, xgb_val], axis=1), Y_val)
            final_preds = ridge.predict(np.concatenate([feats_val, xgb_val], axis=1))

            # 🔍 MC Dropout for uncertainty estimation
            enable_dropout(model)
            mc_preds = []
            with torch.no_grad():
                for _ in range(MC_DROPOUT_SAMPLES):
                    drop_preds = model(torch.tensor(X_val, dtype=torch.float32).to(DEVICE)).cpu().numpy()
                    mc_preds.append(drop_preds)
            # shape: [MC_DROPOUT_SAMPLES, batch, output_dim]
            mc_preds = np.stack(mc_preds)
            # نفترض أن كل نقطة Target هي بموقع منفصل على طول الـ output_dim
            uncertainty_std_per_target = mc_preds.std(axis=0).mean(axis=0)    # shape: [output_dim]

            # 🔎 XGBoost SHAP importance, shape [n_val, output_dim]
            explainer = shap.Explainer(xgb)
            shap_values = explainer(X_val)
            # على مدى ولليس فقط features (لأن XGB متعدد الأهداف)
            if shap_values.values.ndim == 3:
                # [n_val, output_dim, n_features]
                shap_abs = np.abs(shap_values.values) # [n_val, output_dim, n_features]
                # متوسط الشدة عبر جميع العيّنات
                shap_means = shap_abs.mean(axis=0)    # [output_dim, n_features]
                # نعتمد فقط المدخلات الديناميكية (القيم المتنبأ بها لكل نقطة, مواقعها هي الأخيرة في x_concat)
                dynamic_offset = -SEGMENT_LENGTH
                shap_dyn_importance = shap_means[:, dynamic_offset:]  # [output_dim, SEGMENT_LENGTH]
                # نجمع عبر جميع الأعمدة غير المستهدفة، ويظل لكل نقطة هدف وزن واحد
                shap_per_point = shap_dyn_importance.diagonal()       # [output_dim]
            else:
                # وضع fallback: إذا فقط [n_val, n_features]
                shap_abs = np.abs(shap_values.values)  # [n_val, n_features]
                shap_means = shap_abs.mean(axis=0)     # [n_features]
                shap_per_point = shap_means[-SEGMENT_LENGTH:]  # [SEGMENT_LENGTH]
                shap_per_point = shap_per_point[target_mask]   # فقط النقاط غير المعروفة
                uncertainty_std_per_target = uncertainty_std_per_target[target_mask]

            # في حال كانت خروج NDarray من diag غير ممكنة
            if shap_per_point.shape[0] != len(target_mask):
                shap_per_point = shap_per_point[-len(target_mask):]

            # اصطفاء النقاط: الأقل شك والأعلى أهمية
            if shap_per_point.shape != uncertainty_std_per_target.shape:
                raise ValueError(f"Shape mismatch: SHAP={shap_per_point.shape}, Uncertainty={uncertainty_std_per_target.shape}")

            point_scores = shap_per_point - uncertainty_std_per_target

            new_points = sorted(
                zip(target_mask, point_scores.tolist()),
                key=lambda x: -x[1]
            )[:POINTS_PER_ITER]
            known_points += [i for i, _ in new_points]

            for i in range(len(train_ds)):
                pred = model(torch.tensor(X_train[i:i+1], dtype=torch.float32).to(DEVICE)).cpu().detach().numpy().flatten()
                predicted_points[i][target_mask] = pred

            rmse, r2, pearson = evaluate_pointwise_metrics(Y_val, final_preds)
            results_log.append({
                "iteration": iteration + 1,
                "known_points": list(known_points),
                "rmse_mean": float(rmse.mean()),
                "r2_mean": float(r2.mean()),
                "pearson_mean": float(np.mean(pearson))
            })
            uncertainty_log.append({
                "iteration": iteration + 1,
                "uncertainty_std": list(uncertainty_std_per_target)
            })

        # Final Prediction on Test Set
        X_test, Y_test = build_XY(test_ds, predicted_points)
        with torch.no_grad():
            feats_test = mlp(torch.tensor(X_test, dtype=torch.float32).to(DEVICE)).cpu().numpy()
        xgb_test = xgb.predict(X_test)
        final_input = np.concatenate([feats_test, xgb_test], axis=1)
        preds = ridge.predict(final_input)

        rmse_t, r2_t, p_t = evaluate_pointwise_metrics(Y_test, preds)
        with open(SUMMARY_FILE, "w") as f:
            f.write(f"Final metrics for lead {lead}:\n")
            f.write(f"Known points: {known_points}\n")
            f.write(f"RMSE mean: {rmse_t.mean():.4f}\n")
            f.write(f"R2 mean: {r2_t.mean():.4f}\n")
            f.write(f"Pearson: {np.mean(p_t):.4f}\n")

        df_log = pd.DataFrame(results_log)
        df_log.to_csv(OUTPUT_DIR / f"metrics_per_iteration_{lead}.csv", index=False)
        pd.DataFrame(uncertainty_log).to_csv(UNCERTAINTY_FILE, index=False)

        print(f"✅ Done. Final results for lead {lead}:")
        print(f"  RMSE mean  : {rmse_t.mean():.4f}")
        print(f"  R² mean    : {r2_t.mean():.4f}")
        print(f"  Pearson    : {np.mean(p_t):.4f}")
        print(f"  Metrics saved in: {SUMMARY_FILE}")
        print(f"  Iteration log saved in: metrics_per_iteration_{lead}.csv")
        print(f"  Uncertainty log saved in: {UNCERTAINTY_FILE}")
