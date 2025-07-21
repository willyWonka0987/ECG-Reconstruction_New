import joblib
import neurokit2 as nk
import numpy as np
from tqdm import tqdm

# --- Parameters ---
sampling_rate = 100  # Hz
pre_window = 0.06  # seconds before R to search for Q
post_window = 0.06  # seconds after R to search for S
lead_names = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']

def find_qs_minima(signal, r_peaks, pre_window, post_window, sampling_rate):
    pre_samples = int(pre_window * sampling_rate)
    post_samples = int(post_window * sampling_rate)
    q_points, s_points = [], []

    for r in r_peaks:
        start = max(0, r - pre_samples)
        seg_q = signal[start:r]
        q = start + np.argmin(seg_q) if len(seg_q) > 0 else r
        q_points.append(q)

        end = min(len(signal), r + post_samples)
        seg_s = signal[r:end]
        s = r + np.argmin(seg_s) if len(seg_s) > 0 else r
        s_points.append(s)

    return np.array(q_points), np.array(s_points)

def extract_qrs_segments(ecg_dataset):
    all_segments = []

    for sample in tqdm(ecg_dataset, desc="Processing ECG samples"):
        leads_segments = [[] for _ in range(12)]  # Per-lead QRS triples

        for lead_idx in range(12):
            signal = sample[:, lead_idx]
            cleaned = nk.ecg_clean(signal, sampling_rate=sampling_rate)
            r_peaks = nk.ecg_peaks(cleaned, sampling_rate=sampling_rate)[1]['ECG_R_Peaks']
            q_peaks, s_peaks = find_qs_minima(cleaned, r_peaks, pre_window, post_window, sampling_rate)

            for i in range(len(r_peaks)):
                try:
                    q, r, s = q_peaks[i], r_peaks[i], s_peaks[i]
                    leads_segments[lead_idx].append([
                        (q / sampling_rate, signal[q]),  # Q
                        (r / sampling_rate, signal[r]),  # R
                        (s / sampling_rate, signal[s])   # S
                    ])
                except IndexError:
                    continue

        # Transpose to beat-wise: beat_i = [QRS_lead1, ..., QRS_lead12]
        min_beats = min(len(lead) for lead in leads_segments)
        for i in range(min_beats):
            beat_segment = [leads_segments[lead_idx][i] for lead_idx in range(12)]
            all_segments.append(beat_segment)

    return all_segments

def remove_outliers_qrs(all_segments, quantile=0.99):
    lead_qrs_amplitudes = {lead_idx: [] for lead_idx in range(12)}

    # Step 1: Collect amplitudes per lead (Q, R, S)
    for segment in all_segments:
        for lead_idx, qrs in enumerate(segment):
            for _, amp in qrs:  # Each qrs = [(t_q, amp_q), (t_r, amp_r), (t_s, amp_s)]
                lead_qrs_amplitudes[lead_idx].append(abs(amp))  # Use abs to avoid sign issues

    # Step 2: Compute 99th percentile thresholds
    thresholds = {
        lead_idx: np.quantile(lead_qrs_amplitudes[lead_idx], quantile)
        for lead_idx in range(12)
    }

    # Step 3: Filter out segments with any amplitude exceeding threshold
    cleaned_segments = []
    for segment in all_segments:
        keep = True
        for lead_idx, qrs in enumerate(segment):
            for _, amp in qrs:
                if abs(amp) > thresholds[lead_idx]:
                    keep = False
                    break
            if not keep:
                break
        if keep:
            cleaned_segments.append(segment)

    return cleaned_segments

# --- Load ECG data ---
ecg_train_data = joblib.load("data_no_segmentation/ecg_train_clean.pkl")
ecg_test_data = joblib.load("data_no_segmentation/ecg_test_clean.pkl")

# --- Extract raw QRS segments ---
train_segments = extract_qrs_segments(ecg_train_data)
test_segments = extract_qrs_segments(ecg_test_data)

# --- Apply quantile-based amplitude filtering ---
train_segments_clean = remove_outliers_qrs(train_segments, quantile=0.99)
test_segments_clean = remove_outliers_qrs(test_segments, quantile=0.99)

# --- Save cleaned QRS segments ---
joblib.dump(train_segments_clean, "qrs_train_segments.pkl")
joblib.dump(test_segments_clean, "qrs_test_segments.pkl")

print(f"Cleaned and saved: {len(train_segments_clean)} train segments, {len(test_segments_clean)} test segments")
print(f"Original train: {len(train_segments)}, Cleaned: {len(train_segments_clean)}")
print(f"Original test:  {len(test_segments)}, Cleaned: {len(test_segments_clean)}")
