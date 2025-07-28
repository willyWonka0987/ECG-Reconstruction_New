import os
import numpy as np
import matplotlib.pyplot as plt
import neurokit2 as nk
import pandas as pd
from pathlib import Path
import wfdb
from scipy.signal import butter, filtfilt, medfilt

# إعدادات
sampling_rate = 100  # Hz
lead_names = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
output_dir = "PQRST_peaks_on_raw"
os.makedirs(output_dir, exist_ok=True)

# تحميل البيانات
PTBXL_PATH = Path("../../../ptbxl")
DATABASE_CSV = PTBXL_PATH / "ptbxl_database.csv"
df = pd.read_csv(DATABASE_CSV)

# فلترة metadata
for col in ['electrodes_problems', 'pacemaker', 'burst_noise', 'static_noise']:
    df[col] = df[col].fillna(0)
df = df[
    (df['electrodes_problems'] == 0) &
    (df['pacemaker'] == 0) &
    (df['burst_noise'] == 0) &
    (df['static_noise'] == 0)
].reset_index(drop=True)

# فلتر Bandpass
def butter_bandpass_filter(data, lowcut=1.0, highcut=45.0, fs=100.0, order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, data, axis=0)

# إزالة خط الأساس
def baseline_wander_removal(data, fs=100.0, window_sec=0.2):
    window_size = int(window_sec * fs)
    if window_size % 2 == 0:
        window_size += 1
    baseline = medfilt(data, kernel_size=(window_size, 1))
    return data - baseline

def smooth_signal(signal, window_size=3):
    kernel = np.ones(window_size) / window_size
    return np.convolve(signal, kernel, mode='same')

# نوافذ القمم
P_win = (-30, -10)
Q_win = (-7, -2)
S_win = (2, 6)
T_win = (15, 55)

# المعالجة
for idx in range(min(20, len(df))):
    row = df.iloc[idx]
    record_path = PTBXL_PATH / row["filename_lr"].replace(".wav", "")
    record = wfdb.rdrecord(str(record_path))
    raw_signals = record.p_signal[:1000, :]

    try:
        filtered = butter_bandpass_filter(raw_signals, fs=sampling_rate)
        filtered = baseline_wander_removal(filtered, fs=sampling_rate)
        filtered = np.apply_along_axis(smooth_signal, axis=0, arr=filtered)
    except Exception as e:
        print(f"⚠️ فشل الفلترة في ECG {idx}: {e}")
        continue

    plt.figure(figsize=(22, 36))

    for lead_idx in range(12):
        raw_signal = raw_signals[:, lead_idx]       # للرسم فقط
        signal = filtered[:, lead_idx]              # للاستخلاص
        lead_name = lead_names[lead_idx]

        try:
            processed = nk.ecg_process(signal, sampling_rate=sampling_rate)
            r_peaks = processed[1].get("ECG_R_Peaks", [])
        except Exception as e:
            print(f"⚠️ خطأ في ECG {idx}, lead {lead_name}: {e}")
            continue

        r_peaks = np.array(r_peaks, dtype=int)
        r_peaks = r_peaks[(r_peaks >= 0) & (r_peaks < len(signal))]

        peaks_found = {'P': [], 'Q': [], 'T': [], 'S': []}
        windows = {'P': P_win, 'Q': Q_win, 'T': T_win}
        colors = {'P': 'm', 'Q': 'g', 'S': 'b', 'T': 'c'}
        markers = {'P': 'o', 'Q': 'v', 'S': 's', 'T': '*'}

        for r in r_peaks:
            for peak_type in ['P', 'Q', 'T']:
                win = windows[peak_type]
                start = r + win[0]
                end = r + win[1]
                if 0 <= start < end < len(signal):
                    segment = signal[start:end]
                    try:
                        if peak_type in ['P', 'T']:
                            max_val = np.max(segment)
                            min_val = np.min(segment)
                            max_idx = np.argmax(segment)
                            min_idx = np.argmin(segment)
                            peak_rel = win[0] + (max_idx if abs(max_val) >= abs(min_val) else min_idx)
                        elif peak_type == 'Q':
                            q_idx = np.argmin(segment)
                            peak_rel = win[0] + q_idx
                        if not np.isnan(peak_rel):
                            peaks_found[peak_type].append(int(peak_rel))
                    except:
                        continue

            s_start, s_end = r + S_win[0], r + S_win[1]
            if s_end < len(signal):
                segment = signal[s_start:s_end]
                if len(segment) > 0:
                    try:
                        s_rel = np.argmin(segment)
                        peak_rel = S_win[0] + s_rel
                        if not np.isnan(peak_rel):
                            peaks_found['S'].append(int(peak_rel))
                    except:
                        continue

        # --- الرسم على الإشارة الأصلية ---
        plt.subplot(12, 1, lead_idx + 1)
        plt.plot(raw_signal, label='Original Raw Signal', linewidth=1, alpha=0.9)
        plt.plot(r_peaks, raw_signal[r_peaks], 'ro', markersize=6, label='R')

        for peak_type in ['P', 'Q', 'T']:
            for r, rel in zip(r_peaks, peaks_found[peak_type]):
                peak_idx = r + rel
                if 0 <= peak_idx < len(raw_signal):
                    plt.plot(peak_idx, raw_signal[peak_idx],
                             colors[peak_type] + markers[peak_type],
                             markersize=4, label=peak_type)

        for r, rel in zip(r_peaks, peaks_found['S']):
            peak_idx = r + rel
            if 0 <= peak_idx < len(raw_signal):
                plt.plot(peak_idx, raw_signal[peak_idx],
                         colors['S'] + markers['S'],
                         markersize=4, label='S')

        plt.title(f"Lead {lead_name}", fontsize=10)
        plt.ylabel("Amp", fontsize=8)
        plt.grid(True)
        if lead_idx == 0:
            plt.legend(loc='upper right', ncol=6, fontsize=6)

    plt.xlabel("Sample Index", fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"ecg_{idx:02d}_peaks_on_raw.png"))
    plt.close()

print("✅ تم استخراج القمم من الإشارة المنظفة، ورسمها فوق الإشارة الأصلية.")
