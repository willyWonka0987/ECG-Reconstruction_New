import joblib
import neurokit2 as nk
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

sampling_rate = 100
padding = 64
segment_length = 2 * padding
lead_names = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF',
              'V1', 'V2', 'V3', 'V4', 'V5', 'V6']

input_train_path = Path("data_no_segmentation/ecg_train_clean.pkl")
input_test_path = Path("data_no_segmentation/ecg_test_clean.pkl")
meta_train_path = Path("train_split.csv")
meta_test_path = Path("test_split.csv")
output_dir = Path("QRS_Triplet_Input_and_FullBeat_Target")
output_dir.mkdir(parents=True, exist_ok=True)

def find_qs_minima(signal, r_peaks, pre_window=0.06, post_window=0.06):
    pre_samples = int(pre_window * sampling_rate)
    post_samples = int(post_window * sampling_rate)
    q_points, s_points = [], []
    for r in r_peaks:
        start_q = max(0, r - pre_samples)
        seg_q = signal[start_q:r]
        q = start_q + np.argmin(seg_q) if len(seg_q) > 0 else r
        q_points.append(q)
        end_s = min(len(signal), r + post_samples)
        seg_s = signal[r:end_s]
        s = r + np.argmin(seg_s) if len(seg_s) > 0 else r
        s_points.append(s)
    return np.array(q_points), np.array(s_points)

def build_combined_dataset(ecg_dataset, meta_df):
    final_dataset = []
    for i, sample in enumerate(tqdm(ecg_dataset, desc="Building QRS + metadata dataset")):
        row_meta = meta_df.iloc[i]
        age = row_meta["age"]
        sex = row_meta["sex"]
        lead_i = sample[:, 0]
        cleaned = nk.ecg_clean(lead_i, sampling_rate=sampling_rate)
        r_peaks = nk.ecg_peaks(cleaned, sampling_rate=sampling_rate)[1]['ECG_R_Peaks']
        q_peaks, s_peaks = find_qs_minima(cleaned, r_peaks)

        for j, r in enumerate(r_peaks):
            q, s = q_peaks[j], s_peaks[j]
            start, end = r - padding, r + padding
            if (start >= 0 and end <= sample.shape[0]) and (q != r and s != r):
                qrs_lead_i = [
                    (q / sampling_rate, lead_i[q]),
                    (r / sampling_rate, lead_i[r]),
                    (s / sampling_rate, lead_i[s])
                ]
                other_leads_waveforms = {
                    lead_names[k]: sample[start:end, k] for k in range(12) if k != 0
                }
                final_dataset.append({
                    "qrs_lead_I": qrs_lead_i,
                    "other_leads": other_leads_waveforms,
                    "age": age,
                    "sex": sex,
                    "source_index": i
                })
    return final_dataset

train_data = joblib.load(input_train_path)
test_data = joblib.load(input_test_path)
train_meta = pd.read_csv(meta_train_path)
test_meta = pd.read_csv(meta_test_path)

train_combined = build_combined_dataset(train_data, train_meta)
test_combined = build_combined_dataset(test_data, test_meta)

joblib.dump(train_combined, output_dir / "combined_qrs_train.pkl")
joblib.dump(test_combined, output_dir / "combined_qrs_test.pkl")