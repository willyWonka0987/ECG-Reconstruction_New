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
output_dir_comparison = "PQRST_peaks_comparison"
output_dir_raw = "PQRST_peaks_on_raw"
output_dir_filtered = "PQRST_peaks_on_filtered"
os.makedirs(output_dir_comparison, exist_ok=True)
os.makedirs(output_dir_raw, exist_ok=True)
os.makedirs(output_dir_filtered, exist_ok=True)

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

for idx in range(min(20, len(df))):
    row = df.iloc[idx]
    record_path = PTBXL_PATH / row["filename_lr"].replace(".wav", "")
    record = wfdb.rdrecord(str(record_path))
    raw_signals = record.p_signal[:1000, :]

    try:
        filtered = butter_bandpass_filter(raw_signals, fs=sampling_rate)
        filtered = baseline_wander_removal(filtered, fs=sampling_rate)
        # filtered = np.apply_along_axis(smooth_signal, axis=0, arr=filtered)
    except Exception as e:
        print(f"⚠️ فشل الفلترة في ECG {idx}: {e}")
        continue

    # 1. رسم الإشارة الأصلية مع القمم
    plt.figure(figsize=(22, 36))
    for lead_idx in range(12):
        raw_signal = raw_signals[:, lead_idx]
        lead_name = lead_names[lead_idx]
        try:
            processed_raw = nk.ecg_process(raw_signal, sampling_rate=sampling_rate)
            r_peaks_raw = processed_raw[1].get("ECG_R_Peaks", [])
        except Exception as e:
            print(f"⚠️ خطأ في ECG {idx}, lead {lead_name} (raw): {e}")
            continue
        r_peaks_raw = np.array(r_peaks_raw, dtype=int)
        r_peaks_raw = r_peaks_raw[(r_peaks_raw >= 0) & (r_peaks_raw < len(raw_signal))]
        peaks_found_raw = {'P': [], 'Q': [], 'T': [], 'S': []}
        windows = {'P': P_win, 'Q': Q_win, 'T': T_win}
        colors = {'P': 'm', 'Q': 'g', 'S': 'b', 'T': 'c'}
        markers = {'P': 'o', 'Q': 'v', 'S': 's', 'T': '*'}
        for r in r_peaks_raw:
            for peak_type in ['P', 'Q', 'T']:
                win = windows[peak_type]
                start = r + win[0]
                end = r + win[1]
                if 0 <= start < end < len(raw_signal):
                    segment = raw_signal[start:end]
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
                            peaks_found_raw[peak_type].append(int(peak_rel))
                    except:
                        continue
            s_start, s_end = r + S_win[0], r + S_win[1]
            if s_end < len(raw_signal):
                segment = raw_signal[s_start:s_end]
                if len(segment) > 0:
                    try:
                        s_rel = np.argmin(segment)
                        peak_rel = S_win[0] + s_rel
                        if not np.isnan(peak_rel):
                            peaks_found_raw['S'].append(int(peak_rel))
                    except:
                        continue
        plt.subplot(12, 1, lead_idx + 1)
        plt.plot(raw_signal, color='tab:blue', label='Raw Signal', alpha=0.7)
        plt.plot(r_peaks_raw, raw_signal[r_peaks_raw], 'ro', markersize=4, label='R')
        for peak_type in ['P', 'Q', 'T']:
            for r, rel in zip(r_peaks_raw, peaks_found_raw[peak_type]):
                peak_idx = r + rel
                if 0 <= peak_idx < len(raw_signal):
                    plt.plot(peak_idx, raw_signal[peak_idx],
                             colors[peak_type] + markers[peak_type], markersize=4, label=peak_type)
        for r, rel in zip(r_peaks_raw, peaks_found_raw['S']):
            peak_idx = r + rel
            if 0 <= peak_idx < len(raw_signal):
                plt.plot(peak_idx, raw_signal[peak_idx],
                         colors['S'] + markers['S'], markersize=4, label='S')
        plt.title(f"Lead {lead_name}", fontsize=10)
        plt.ylabel("Amp", fontsize=8)
        plt.grid(True)
        if lead_idx == 0:
            plt.legend(loc='upper right', ncol=6, fontsize=6)
    plt.xlabel("Sample Index", fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir_raw, f"ecg_{idx:02d}_peaks_on_raw.png"))
    plt.close()

    # 2. رسم الإشارة النظيفة مع القمم
    plt.figure(figsize=(22, 36))
    for lead_idx in range(12):
        filt_signal = filtered[:, lead_idx]
        lead_name = lead_names[lead_idx]
        try:
            processed_filt = nk.ecg_process(filt_signal, sampling_rate=sampling_rate)
            r_peaks_filt = processed_filt[1].get("ECG_R_Peaks", [])
        except Exception as e:
            print(f"⚠️ خطأ في ECG {idx}, lead {lead_name} (filt): {e}")
            continue
        r_peaks_filt = np.array(r_peaks_filt, dtype=int)
        r_peaks_filt = r_peaks_filt[(r_peaks_filt >= 0) & (r_peaks_filt < len(filt_signal))]
        peaks_found_filt = {'P': [], 'Q': [], 'T': [], 'S': []}
        windows = {'P': P_win, 'Q': Q_win, 'T': T_win}
        colors = {'P': 'm', 'Q': 'g', 'S': 'b', 'T': 'c'}
        markers = {'P': 'o', 'Q': 'v', 'S': 's', 'T': '*'}
        for r in r_peaks_filt:
            for peak_type in ['P', 'Q', 'T']:
                win = windows[peak_type]
                start = r + win[0]
                end = r + win[1]
                if 0 <= start < end < len(filt_signal):
                    segment = filt_signal[start:end]
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
                            peaks_found_filt[peak_type].append(int(peak_rel))
                    except:
                        continue
            s_start, s_end = r + S_win[0], r + S_win[1]
            if s_end < len(filt_signal):
                segment = filt_signal[s_start:s_end]
                if len(segment) > 0:
                    try:
                        s_rel = np.argmin(segment)
                        peak_rel = S_win[0] + s_rel
                        if not np.isnan(peak_rel):
                            peaks_found_filt['S'].append(int(peak_rel))
                    except:
                        continue
        plt.subplot(12, 1, lead_idx + 1)
        plt.plot(filt_signal, color='tab:red', label='Filtered Signal', alpha=0.7)
        plt.plot(r_peaks_filt, filt_signal[r_peaks_filt], 'ko', markersize=4, label='R')
        for peak_type in ['P', 'Q', 'T']:
            for r, rel in zip(r_peaks_filt, peaks_found_filt[peak_type]):
                peak_idx = r + rel
                if 0 <= peak_idx < len(filt_signal):
                    plt.plot(peak_idx, filt_signal[peak_idx],
                             colors[peak_type] + markers[peak_type], markersize=4, label=peak_type)
        for r, rel in zip(r_peaks_filt, peaks_found_filt['S']):
            peak_idx = r + rel
            if 0 <= peak_idx < len(filt_signal):
                plt.plot(peak_idx, filt_signal[peak_idx],
                         colors['S'] + markers['S'], markersize=4, label='S')
        plt.title(f"Lead {lead_name}", fontsize=10)
        plt.ylabel("Amp", fontsize=8)
        plt.grid(True)
        if lead_idx == 0:
            plt.legend(loc='upper right', ncol=6, fontsize=6)
    plt.xlabel("Sample Index", fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir_filtered, f"ecg_{idx:02d}_peaks_on_filtered.png"))
    plt.close()

    # 3. المقارنة بين الاشارتين معاً في نفس الرسم (كما في الكود الحالي)
    plt.figure(figsize=(22,36))
    for lead_idx in range(12):
        raw_signal = raw_signals[:, lead_idx]
        filt_signal = filtered[:, lead_idx]
        lead_name = lead_names[lead_idx]

        # إعادة نفس استخراج القمم للاثنين (يمكنك إعادة reuse من فوق لتسريع الكود لو أردت)
        # نعيد الرسم كما كان لكن الإشارتين سوياً
        # رسم الأصلي أولاً مع القمم
        plt.subplot(12, 1, lead_idx + 1)
        plt.plot(raw_signal, color='tab:blue', label='Raw Signal', alpha=0.7)
        plt.plot(r_peaks_raw, raw_signal[r_peaks_raw], 'ro', markersize=4, label='R (raw)')
        for peak_type in ['P', 'Q', 'T']:
            for r, rel in zip(r_peaks_raw, peaks_found_raw[peak_type]):
                peak_idx = r + rel
                if 0 <= peak_idx < len(raw_signal):
                    plt.plot(peak_idx, raw_signal[peak_idx],
                             colors[peak_type] + markers[peak_type], markersize=4, label=f'{peak_type} (raw)')
        for r, rel in zip(r_peaks_raw, peaks_found_raw['S']):
            peak_idx = r + rel
            if 0 <= peak_idx < len(raw_signal):
                plt.plot(peak_idx, raw_signal[peak_idx],
                         colors['S'] + markers['S'], markersize=4, label='S (raw)')

        # رسم الفلتر فوقها مع القمم
        plt.plot(filt_signal, color='tab:red', label='Filtered Signal', alpha=0.6)
        plt.plot(r_peaks_filt, filt_signal[r_peaks_filt], 'ko', markersize=6, label='R (filtered)')
        for peak_type in ['P', 'Q', 'T']:
            for r, rel in zip(r_peaks_filt, peaks_found_filt[peak_type]):
                peak_idx = r + rel
                if 0 <= peak_idx < len(filt_signal):
                    plt.plot(peak_idx, filt_signal[peak_idx],
                             colors[peak_type] + markers[peak_type], markersize=6, label=f'{peak_type} (filtered)')
        for r, rel in zip(r_peaks_filt, peaks_found_filt['S']):
            peak_idx = r + rel
            if 0 <= peak_idx < len(filt_signal):
                plt.plot(peak_idx, filt_signal[peak_idx],
                         colors['S'] + markers['S'], markersize=6, label='S (filtered)')

        plt.title(f"Lead {lead_name}", fontsize=10)
        plt.ylabel("Amp", fontsize=8)
        plt.grid(True)
        if lead_idx == 0:
            plt.legend(loc='upper right', ncol=4, fontsize=6, framealpha=0.4)
    plt.xlabel("Sample Index", fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir_comparison, f"ecg_{idx:02d}_raw_vs_filtered.png"))
    plt.close()

print("✅ تم استخراج ورسم القمم على الإشارة الأصلية والإشارة النظيفة، وتم تخزين المخططات في مجلدات منفصلة.")  
