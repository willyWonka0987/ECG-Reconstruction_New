import os
import numpy as np
import matplotlib.pyplot as plt
import neurokit2 as nk
from scipy.signal import find_peaks

sampling_rate = 100  # Hz
lead_names = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']

output_dir = "PQRST_plots_robust_local_search"
os.makedirs(output_dir, exist_ok=True)

data_path = "../ptbxl_dl_dataset_v2/datasets/train_signals.npz"
ecg_data = np.load(data_path)['X']  # shape: (N, 1000, 12)

# نافذة القمم بالنسبة لـ R (بالعينات، حسب 100Hz)
P_win = (-30, -10)
Q_win = (-7, -2)
S_win = (2, 6)
T_win = (15, 55)

for idx in range(min(100, len(ecg_data))):
    ecg_12lead = ecg_data[idx]
    plt.figure(figsize=(22, 36))

    for lead_idx in range(12):
        signal = ecg_12lead[:, lead_idx]
        lead_name = lead_names[lead_idx]
        cleaned = nk.ecg_clean(signal, sampling_rate=sampling_rate, method="biosppy")
        rpeaks_dict = nk.ecg_peaks(cleaned, sampling_rate=sampling_rate)[1]
        r_peaks = np.array(rpeaks_dict.get('ECG_R_Peaks', []), dtype=int)
        r_peaks = r_peaks[(r_peaks >= 0) & (r_peaks < len(cleaned))]
        print(f"Lead {lead_name} - ECG {idx}: Found {len(r_peaks)} R-peaks.")

        peaks_found = {'P': [], 'Q': [], 'T': [], 'S': []}
        windows = {'P': P_win, 'Q': Q_win, 'T': T_win}
        colors = {'P': 'm', 'Q': 'g', 'S': 'b', 'T': 'c'}
        markers = {'P': 'o', 'Q': 'v', 'S': 's', 'T': '*'}

        for r in r_peaks:
            for peak_type in ['P', 'Q', 'T']:
                win = windows[peak_type]
                start = r + win[0]
                end = r + win[1]
                if 0 <= start < end < len(cleaned):
                    segment = cleaned[start:end]
                    if peak_type in ['P', 'T']:
                        # نحاول التقاط القمة الأكبر أو الأصغر (موجبة أو سالبة)
                        max_val = np.max(segment)
                        min_val = np.min(segment)
                        max_idx = np.argmax(segment)
                        min_idx = np.argmin(segment)
                        if abs(max_val) >= abs(min_val):
                            peaks_found[peak_type].append(win[0] + max_idx)
                        else:
                            peaks_found[peak_type].append(win[0] + min_idx)
                    elif peak_type == 'Q':
                        q_idx = np.argmin(segment)
                        peaks_found[peak_type].append(win[0] + q_idx)

            # قمة S: أدنى نقطة بعد R مباشرة
            s_start, s_end = r + S_win[0], r + S_win[1]
            if s_end < len(cleaned):
                segment = cleaned[s_start:s_end]
                if len(segment) > 0:
                    s_rel = np.argmin(segment)
                    peaks_found['S'].append(S_win[0] + s_rel)

        # الرسم
        plt.subplot(12, 1, lead_idx + 1)
        plt.plot(cleaned, label='ECG Signal', linewidth=1)
        plt.plot(r_peaks, cleaned[r_peaks], 'ro', markersize=7, label='R-peaks')

        for peak_type in ['P', 'Q', 'T']:
            for r, rel in zip(r_peaks, peaks_found[peak_type]):
                peak_idx = r + rel
                if 0 <= peak_idx < len(cleaned):
                    plt.plot(peak_idx, cleaned[peak_idx],
                             colors[peak_type] + markers[peak_type],
                             markersize=5, label=f'{peak_type}-peak')

        for r, rel in zip(r_peaks, peaks_found['S']):
            peak_idx = r + rel
            if 0 <= peak_idx < len(cleaned):
                plt.plot(peak_idx, cleaned[peak_idx],
                         colors['S'] + markers['S'],
                         markersize=5, label='S-peak')

        plt.title(f"Lead {lead_name}", fontsize=10)
        plt.ylabel("Amplitude", fontsize=9)
        plt.grid(True)
        if lead_idx == 0:
            plt.legend(loc='upper right', ncol=6)

    plt.xlabel("Sample Index (0 to 999)", fontsize=10)
    plt.subplots_adjust(hspace=1.3)
    plt.savefig(os.path.join(output_dir, f"ecg_{idx:02d}_local_peaks.png"))
    plt.close()

print("\u2705 تم تحديد القمم P,Q,T بطريقة محلية تدعم القمم السالبة، وقمة S كأدنى نقطة بعد R مباشرة.")