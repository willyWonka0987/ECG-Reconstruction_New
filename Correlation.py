import pickle
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
import os

# --- Configuration ---
SAVE_DIR = Path("RichECG_Datasets")
TARGET_LEADS = ["III", "V1", "V3", "V4", "V5", "V6"]
INPUT_LEADS = ["I", "II", "V2"]
SEGMENT_LENGTH = 80
PLOTS_DIR = Path("Correlation/plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

THRESHOLD = 0.0  # Set your threshold

lead_stats = ['mean', 'std', 'min', 'max', 'skew', 'kurtosis', 'rms', 'entropy',
              'total_power', 'dominant_freq', 'wavelet_energy',
              'max_slope', 'min_slope', 'mean_slope', 'zero_crossings', 'slope_early', 'slope_late']
feature_names = []
for lead in INPUT_LEADS:
    feature_names += [f"{lead}_{stat}" for stat in lead_stats]
    feature_names += [f"{lead}_{p}_amp" for p in ['P','Q','R','S','T']]
    feature_names += [f"{lead}_{p}_time" for p in ['P','Q','R','S','T']]
    feature_names += [f"{lead}_QRS_area", f"{lead}_QRS_duration"]
    feature_names += [f"{lead}_{iv}" for iv in ['PR','QT','ST','RR']]
feature_names += ['age', 'sex', 'heart_axis_normal', 'heart_axis_left_axis_deviation', 'heart_axis_right_axis_deviation']

def load_test_data(features_path, segments_path, target_lead):
    segments = np.load(segments_path)[:, :SEGMENT_LENGTH, :]
    xs, ys = [], []
    with open(features_path, "rb") as f:
        while True:
            try:
                rec = pickle.load(f)
                seg_idx = rec.get("segment_index")
                if seg_idx is None or seg_idx >= segments.shape[0]:
                    continue
                feat = np.concatenate([rec["features"][ld] for ld in INPUT_LEADS])
                amps, times, qrs_area, qrs_dur = [], [], [], []
                for ld in INPUT_LEADS:
                    wave = rec["waves"].get(ld, {})
                    amps += [wave.get(f"{w}_amp", 0) or 0 for w in ['P','Q','R','S','T']]
                    times += [wave.get(f"{w}_time", 0) or 0 for w in ['P','Q','R','S','T']]
                    qrs_area.append(wave.get("QRS_area", 0) or 0)
                    qrs_dur.append(wave.get("QRS_duration", 0) or 0)
                intervals = []
                for ld in INPUT_LEADS:
                    ival = rec["intervals"].get(ld, {})
                    for key in ['PR','QT','ST','RR']:
                        v = ival.get(key)
                        intervals.append(np.mean(v) if isinstance(v, list) and len(v) else 0)
                def encode_metadata(meta):
                    age = meta.get("age", 0)
                    sex = 1 if meta.get("sex", "").lower() == "male" else 0
                    axis = meta.get("heart_axis", "").lower()
                    axis_vec = [0, 0, 0]
                    if axis == "normal": axis_vec[0] = 1
                    elif axis == "left axis deviation": axis_vec[1] = 1
                    elif axis == "right axis deviation": axis_vec[2] = 1
                    return [age, sex] + axis_vec
                meta_features = encode_metadata(rec["metadata"])
                x = np.concatenate([feat, amps, times, qrs_area, qrs_dur, intervals, meta_features])
                lead_index = ['I','II','III','aVR','aVL','aVF','V1','V2','V3','V4','V5','V6'].index(target_lead)
                y = segments[seg_idx, :, lead_index]
                xs.append(x)
                ys.append(y)
            except EOFError:
                break
            except Exception:
                continue
    return np.stack(xs), np.stack(ys)

def compute_correlation_matrix(X, Y):
    correlations = np.zeros((X.shape[1], Y.shape[1]))
    for f in tqdm(range(X.shape[1]), desc="Feature correlations"):
        for t in range(Y.shape[1]):
            with np.errstate(invalid='ignore'):
                corr = np.corrcoef(X[:, f], Y[:, t])[0, 1]
            correlations[f, t] = corr if np.isfinite(corr) else 0
    return correlations

def plot_masked_correlation_heatmap(correlations, lead_name, feature_labels, threshold):
    masked = np.full_like(correlations, fill_value=np.nan)
    masked[correlations > threshold] = correlations[correlations > threshold]

    # Get feature names with all correlations <= threshold
    below_mask = np.all(correlations <= threshold, axis=1)
    n_below = np.sum(below_mask)
    features_below = [feature_labels[i] for i, b in enumerate(below_mask) if b]

    from matplotlib.colors import ListedColormap
    red_cmap = plt.get_cmap('Reds')
    colors = red_cmap(np.linspace(0, 1, 256))
    white = np.array([1, 1, 1, 1])
    colors[0] = white
    custom_cmap = ListedColormap(colors)

    plt.figure(figsize=(24, max(12, len(feature_labels) * 0.45)))
    im = plt.imshow(masked, aspect='auto', cmap=custom_cmap, vmin=threshold, vmax=1)
    plt.colorbar(im, label=f"Pearson r (> {threshold})")
    plt.xlabel("Output segment point (0–79)")
    plt.ylabel("Feature")
    plt.title(f"Features with r > {threshold} in red — (Lead {lead_name})", fontsize=18, weight='bold')
    plt.yticks(np.arange(len(feature_labels)), feature_labels, fontsize=7)

    # Write the number of features below threshold
    plt.gcf().text(
        0.5, 0.96,
        f"{n_below} features have all correlations ≤ {threshold}",
        fontsize=20, color='firebrick', ha='center', va='bottom', weight='bold',
        bbox=dict(facecolor='white', alpha=0.97, boxstyle='round,pad=0.6')
    )
    # Write the list of those features (split across lines if necessary)
    if features_below:
        # Group the feature list into chunks to prevent overly long lines
        grouped = [features_below[i:i+7] for i in range(0, len(features_below), 7)]
        full_text = '\n'.join([', '.join(group) for group in grouped])
        # Place the full list below the count
        plt.gcf().text(
            0.5, 0.925, full_text,
            fontsize=12, color='black', ha='center', va='top',
            bbox=dict(facecolor='white', alpha=0.95, boxstyle='round,pad=0.4')
        )

    plt.tight_layout(rect=[0, 0, 1, 0.91])
    plt.savefig(PLOTS_DIR / f"feature_output_poscorronly_{lead_name}_annotated.png", dpi=300)
    plt.close()

# --- Main execution ---
for lead in TARGET_LEADS:
    print(f"Processing {lead}")
    X, Y = load_test_data(SAVE_DIR / "features_train.pkl",
                          SAVE_DIR / "segments_train.npy", lead)
    corr_mat = compute_correlation_matrix(X, Y)
    plot_masked_correlation_heatmap(corr_mat, lead, feature_names, THRESHOLD)

print("All heatmaps with fully saved and written feature lists are now in Correlation/plots.")
