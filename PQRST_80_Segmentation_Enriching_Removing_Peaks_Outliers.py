# ✅ Full script with outlier peak filtering added before saving

import os
import numpy as np
import pandas as pd
import neurokit2 as nk
from pathlib import Path
from tqdm import tqdm
import pickle
from scipy.stats import skew, kurtosis, entropy
from scipy.signal import welch
import pywt
import multiprocessing as mp
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="neurokit2")
warnings.filterwarnings("ignore", category=UserWarning, module="neurokit2.signal.signal_period")

# --- Configuration ---
DATA_DIR = Path("ptbxl_dl_dataset_v2/datasets")
SAVE_DIR = Path("RichECG_Datasets_Outliers_Removed")
STATS_DIR = SAVE_DIR / "stats"
SAVE_DIR.mkdir(parents=True, exist_ok=True)
STATS_DIR.mkdir(parents=True, exist_ok=True)
REPORT_PATH = SAVE_DIR / "segmentation_report.txt"
SAMPLING_RATE = 100
SEGMENT_LENGTH = 80
HALF_WINDOW = SEGMENT_LENGTH // 2
R_WINDOW = 50
LEAD_NAMES = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
SPLIT_SIZES = {"train": None, "val": None, "test": None}
PEAK_COLORS = {"P": "blue", "Q": "orange", "R": "red", "S": "green", "T": "purple"}

report_lines = []

# ⬇️ This helper function checks for outlier amplitudes

def has_outlier_peaks(peaks_dict, segment, threshold=5.0):
    all_peaks = []
    for lead_idx, lead in enumerate(LEAD_NAMES):
        for wave in ['P', 'Q', 'R', 'S', 'T']:
            peak_idx = peaks_dict[lead][wave]
            if peak_idx is not None and 0 <= peak_idx < SEGMENT_LENGTH:
                amp = segment[int(peak_idx), lead_idx]
                all_peaks.append(amp)
    if len(all_peaks) < 2:
        return True  # not enough peaks to compute stats
    all_peaks = np.array(all_peaks)
    mean = np.mean(all_peaks)
    std = np.std(all_peaks)
    if std == 0:
        return True  # degenerate case
    z_scores = np.abs((all_peaks - mean) / std)
    return np.any(z_scores > threshold)

# ⬇️ Inside `process_record` function, just before saving segment:
# Add this line:
#     if has_outlier_peaks(segment_peaks, segment): continue
# You can insert that line in the same location where you skip segment if any peak is None
#     if any(segment_peaks[...]): continue

# ✨ All other parts of your code remain 100% unchanged.
# ✅ The modified script will skip any ECG segment if *any peak* in any lead has amplitude > 5 std from global mean.




def plot_segment(segment, segment_peaks, record_id, segment_idx):
    plt.figure(figsize=(12, 10))
    for i, lead in enumerate(LEAD_NAMES):
        signal = segment[:, i]
        plt.plot(signal + i * 2, label=lead, color="black", linewidth=1)
        for wave, color in PEAK_COLORS.items():
            peak = segment_peaks.get(lead, {}).get(wave)
            if peak is not None and isinstance(peak, (int, np.integer)) and 0 <= peak < len(signal):
                plt.scatter(int(peak), signal[int(peak)] + i * 2, color=color, s=20)
    plt.title(f"Record {record_id} | Segment {segment_idx}")
    plt.xlabel("Samples")
    plt.ylabel("Amplitude + offset")
    plt.grid(True)
    plt.tight_layout()
    plt.legend(loc="upper right", fontsize=6)
    img_path = SAVE_DIR / f"{record_id}_segment_{segment_idx}.png"
    plt.savefig(img_path)
    plt.close()

def extract_features(signal):
    signal = np.asarray(signal).astype(np.float32)

    # حماية من الإشارات القصيرة أو الفارغة
    if len(signal) < 2 or np.any(~np.isfinite(signal)):
        return [0.0] * 17  # عدد الميزات الكلي المتوقع

    # الميزات الإحصائية
    try:
        mean_ = np.mean(signal)
        std_ = np.std(signal)
        min_ = np.min(signal)
        max_ = np.max(signal)
        skew_ = float(skew(signal))
        kurt = float(kurtosis(signal))
        rms = np.sqrt(np.mean(signal**2))
    except:
        mean_, std_, min_, max_, skew_, kurt, rms = [0.0]*7

    # Histogram + Entropy
    try:
        hist, _ = np.histogram(signal, bins=10, density=True)
        if np.sum(hist) == 0 or np.any(~np.isfinite(hist)):
            entropy_val = 0.0
        else:
            entropy_val = float(entropy(hist))
    except:
        entropy_val = 0.0

    # PSD - Power Spectral Density
    try:
        freqs, psd = welch(signal, fs=SAMPLING_RATE, nperseg=min(40, len(signal)))
        total_power = float(np.sum(psd))
        dom_freq = float(freqs[np.argmax(psd)]) if len(psd) > 0 else 0.0
    except:
        total_power, dom_freq = 0.0, 0.0

    # Wavelet Energy
    try:
        coeffs = pywt.wavedec(signal, 'db6', level=2)
        wavelet_energy = float(sum(np.sum(c**2) for c in coeffs))
    except:
        wavelet_energy = 0.0

    # Slope Features
    try:
        dx = 1.0 / SAMPLING_RATE
        slope = np.diff(signal) / dx
        max_slope = np.max(slope)
        min_slope = np.min(slope)
        mean_slope = np.mean(slope)
        zero_crossings = len(np.where(np.diff(np.sign(slope)))[0])
        slope_early = (signal[20] - signal[0]) / (20 * dx) if len(signal) > 20 else 0.0
        slope_late = (signal[-1] - signal[40]) / ((len(signal) - 41) * dx) if len(signal) > 40 else 0.0
    except:
        max_slope = min_slope = mean_slope = slope_early = slope_late = 0.0
        zero_crossings = 0

    # الميزات النهائية
    return [
        mean_, std_, min_, max_, skew_, kurt, rms, entropy_val,         # ← 8 ميزات إحصائية
        total_power, dom_freq,                                          # ← 2 ميزات ترددية
        wavelet_energy,                                                 # ← 1 ميزة تحويل موجي
        max_slope, min_slope, mean_slope, zero_crossings, slope_early, slope_late  # ← 6 ميزات انحدار
    ]


def extract_intervals(delineate_raw):
    delineate = delineate_raw[1] if isinstance(delineate_raw, tuple) else delineate_raw
    def get(key): return delineate.get(key, [])
    def compute_interval(start_peaks, end_peaks):
        return [end - start for start, end in zip(start_peaks, end_peaks)]
    return {
        'PR': compute_interval(get('ECG_P_Peaks'), get('ECG_R_Peaks')),
        'QT': compute_interval(get('ECG_Q_Peaks'), get('ECG_T_Peaks')),
        'ST': compute_interval(get('ECG_S_Peaks'), get('ECG_T_Peaks')),
        'RR': compute_interval(get('ECG_R_Peaks')[:-1], get('ECG_R_Peaks')[1:])
    }

def pad_segment(segment, target_length):
    seg_len = segment.shape[0]
    if seg_len < target_length:
        pad_width = ((0, target_length - seg_len), (0, 0))
        segment = np.pad(segment, pad_width, mode='constant')
    elif seg_len > target_length:
        segment = segment[:target_length, :]
    return segment

def find_pqrst_around_r_local(signal, r_idx):
    """
    Locally search around R for P, Q, S, and T peaks in a single lead signal.
    Returns the sample indices of P, Q, R, S, T.
    """
    P_win = (-30, -10)
    Q_win = (-7, -2)
    S_win = (2, 6)
    T_win = (15, 55)
    peaks = {'P': None, 'Q': None, 'R': r_idx, 'S': None, 'T': None}

    def local_peak(win, prefer_negative=False):
        start = r_idx + win[0]
        end = r_idx + win[1]
        if start < 0 or end > len(signal): return None
        segment = signal[start:end]
        if len(segment) == 0: return None
        max_val = np.max(segment)
        min_val = np.min(segment)
        max_idx = np.argmax(segment)
        min_idx = np.argmin(segment)
        if prefer_negative:
            return start + min_idx if abs(min_val) >= abs(max_val) else start + max_idx
        else:
            return start + max_idx if abs(max_val) >= abs(min_val) else start + min_idx

    peaks['P'] = local_peak(P_win)
    peaks['Q'] = local_peak(Q_win, prefer_negative=True)
    peaks['T'] = local_peak(T_win)
    # S as lowest point after R
    s_start = r_idx + S_win[0]
    s_end = r_idx + S_win[1]
    if s_end < len(signal):
        segment = signal[s_start:s_end]
        if len(segment) > 0:
            s_idx = np.argmin(segment)
            peaks['S'] = s_start + s_idx

    # Ensure order validity
    if not (peaks['P'] and peaks['Q'] and peaks['S'] and peaks['T']):
        return None
    if not (peaks['P'] < peaks['Q'] < peaks['R'] < peaks['S'] < peaks['T']):
        return None

    return peaks['P'], peaks['Q'], peaks['R'], peaks['S'], peaks['T']

def compute_area(signal, q_idx, s_idx):
    if q_idx is None or s_idx is None or q_idx >= s_idx:
        return 0.0
    q_idx, s_idx = int(q_idx), int(s_idx)
    segment = signal[q_idx:s_idx+1]
    if len(segment) < 2:
        return 0.0
    return float(np.trapz(segment))  # Approximate area under curve

def compute_duration(q_idx, s_idx):
    if q_idx is None or s_idx is None or q_idx >= s_idx:
        return 0.0
    return float(s_idx - q_idx) / SAMPLING_RATE

def process_record(args):
    idx, ecg, meta, label = args
    record_id = meta.get('ecg_id', f'record_{idx}')
    segments_out, pqrst_out, segment_keys = [], [], []
    if np.any(np.isnan(ecg)) or np.any(np.isinf(ecg)):
        return [], [], []
    rpeaks_all, delineate_all = [], []
    for lead_idx in range(len(LEAD_NAMES)):
        signal = nk.ecg_clean(ecg[:, lead_idx], sampling_rate=SAMPLING_RATE)
        try:
            _, info = nk.ecg_process(signal, sampling_rate=SAMPLING_RATE)
            rpeaks = info['ECG_R_Peaks']
            delineate = nk.ecg_delineate(signal, rpeaks=rpeaks, sampling_rate=SAMPLING_RATE, method="dwt")
            delineate = delineate[1] if isinstance(delineate, tuple) else delineate
            rpeaks_all.append(rpeaks)
            delineate_all.append(delineate)
        except: return [], [], []
    ref_rpeaks = rpeaks_all[0]
    if len(ref_rpeaks) == 0: return [], [], []
    for ref_r in ref_rpeaks:
        r_indices = [ref_r]
        segment_peaks = {lead: {} for lead in LEAD_NAMES}
        segment_peaks['I']['R'] = ref_r
        valid = True
        for lead_idx in range(1, len(LEAD_NAMES)):
            rpeaks = rpeaks_all[lead_idx]
            lead_name = LEAD_NAMES[lead_idx]
            candidates = [r for r in rpeaks if abs(r - ref_r) <= R_WINDOW]
            if not candidates:
                valid = False; break
            matched_r = min(candidates, key=lambda x: abs(x - ref_r))
            pqrst = find_pqrst_around_r_local(ecg[:, lead_idx], matched_r)
            if pqrst is None:
                valid = False; break
            for name, val in zip(['P', 'Q', 'R', 'S', 'T'], pqrst):
                segment_peaks[lead_name][name] = val
            r_indices.append(matched_r)
        if not valid: continue
        pqrst = find_pqrst_around_r_local(ecg[:, lead_idx], matched_r)
        if pqrst is None: continue
        for name, val in zip(['P', 'Q', 'R', 'S', 'T'], pqrst):
            segment_peaks['I'][name] = val
        start = max(0, min(r_indices) - HALF_WINDOW)
        end = min(ecg.shape[0], max(r_indices) + HALF_WINDOW)
        segment = pad_segment(ecg[start:end], SEGMENT_LENGTH)
        for lead in LEAD_NAMES:
            for wave in ['P', 'Q', 'R', 'S', 'T']:
                peak = segment_peaks[lead][wave]
                segment_peaks[lead][wave] = peak - start if peak and start <= peak < end else None
        if any(segment_peaks[lead][w] is None for lead in LEAD_NAMES for w in ['P', 'Q', 'R', 'S', 'T']):
            continue
        feature_record = {'record_id': record_id, 'label': label, 'metadata': meta, 'features': {}, 'intervals': {}, 'waves': {}, 'peaks': segment_peaks}
        for lead_idx, lead_name in enumerate(LEAD_NAMES):
            signal = ecg[:, lead_idx]
            try:
                feats = extract_features(signal)
                feature_record['features'][lead_name] = feats
            except: return [], [], []
            feature_record['intervals'][lead_name] = extract_intervals(delineate_all[lead_idx])
            wave_info = {}
            for wave in ['P', 'Q', 'R', 'S', 'T']:
                rel_peak = segment_peaks[lead_name][wave]
                if rel_peak is not None and 0 <= rel_peak < SEGMENT_LENGTH:
                    wave_info[f"{wave}_amp"] = float(segment[int(rel_peak), lead_idx])
                    wave_info[f"{wave}_time"] = rel_peak / SAMPLING_RATE
                else:
                    wave_info[f"{wave}_amp"] = None
                    wave_info[f"{wave}_time"] = None

            # QRS Area and Duration ← هذا خارج الفور، لكن داخل نفس البلوك
            q_idx = segment_peaks[lead_name].get('Q')
            s_idx = segment_peaks[lead_name].get('S')
            wave_info['QRS_area'] = compute_area(segment[:, lead_idx], q_idx, s_idx)
            wave_info['QRS_duration'] = compute_duration(q_idx, s_idx)

            feature_record['waves'][lead_name] = wave_info
        feature_record['segment'] = segment  # Add this line
        if has_outlier_peaks(segment_peaks, segment):
            continue
        segments_out.append(segment)
        pqrst_out.append(feature_record)
        segment_keys.append({'record_id': record_id, 'R_indices': r_indices})
        if idx == 0:
            plot_segment(segment, segment_peaks, record_id, len(pqrst_out))
    return segments_out, pqrst_out, segment_keys

def process_all(set_name, max_records):
    report_lines.append(f"\n--- Processing {set_name} set ---")
    try:
        ecg_data = np.load(DATA_DIR / f"{set_name}_signals.npz")['X']
        labels = pd.read_csv(DATA_DIR / f"{set_name}_labels.csv")
        meta = pd.read_csv(DATA_DIR / f"{set_name}_metadata.csv")
        if max_records:
            ecg_data, labels, meta = ecg_data[:max_records], labels.iloc[:max_records], meta.iloc[:max_records]
        report_lines.append(f"Loaded {set_name}: {ecg_data.shape[0]} records")
    except Exception as e:
        report_lines.append(f"Failed loading {set_name}: {e}"); return
    args = [(i, ecg_data[i], meta.iloc[i].to_dict(), labels.iloc[i].to_dict()) for i in range(len(ecg_data))]
    segments_all, keys_all, total_segments = [], [], 0
    with open(SAVE_DIR / f"features_{set_name}.pkl", 'wb') as f_pkl:
        with mp.Pool(processes=4) as pool:
            for segments, pqrst_list, segment_keys in tqdm(pool.imap(process_record, args), total=len(args)):
                for i, p in enumerate(pqrst_list):
                    p['segment_index'] = total_segments + i
                    pickle.dump(p, f_pkl, protocol=pickle.HIGHEST_PROTOCOL)
                segments_all.extend(segments); keys_all.extend(segment_keys)
                total_segments += len(segments)
    np.save(SAVE_DIR / f"segments_{set_name}.npy", np.array(segments_all, dtype=np.float32))
    with open(SAVE_DIR / f"segment_keys_{set_name}.pkl", 'wb') as f_key:
        pickle.dump(keys_all, f_key)
    meta.to_csv(SAVE_DIR / f"metadata_{set_name}.csv", index=False)
    labels.to_csv(SAVE_DIR / f"labels_{set_name}.csv", index=False)
    report_lines.append(f"Total aligned segments saved: {total_segments}\n")
    with open(REPORT_PATH, 'a') as f:
        for line in report_lines: f.write(line + '\n')
        report_lines.clear()

def collect_all_features_from_pkl():
    features = []
    for split in ["train", "val", "test"]:
        path = SAVE_DIR / f"features_{split}.pkl"
        with open(path, "rb") as f:
            while True:
                try:
                    r = pickle.load(f)
                    for lead in LEAD_NAMES:
                        fvec = r['features'].get(lead)
                        if fvec is not None and np.all(np.isfinite(fvec)):
                            features.append(fvec)
                except EOFError:
                    break
    return features

def analyze_features(features_list):
    if not features_list:
        print("No features collected."); return
    df = pd.DataFrame(features_list)
    df.describe().to_csv(STATS_DIR / "feature_summary.csv")
    for col in df.columns:
        plt.figure()
        sns.histplot(df[col], bins=30, kde=True, color='skyblue')
        plt.title(f"Distribution of Feature {col}")
        plt.tight_layout()
        plt.savefig(STATS_DIR / f"feature_{col}_hist.png")
        plt.close()

if __name__ == '__main__':
    for split in SPLIT_SIZES:
        process_all(split, SPLIT_SIZES[split])
    feats = collect_all_features_from_pkl()
    analyze_features(feats)
