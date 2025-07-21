import pickle
import numpy as np

# Same list as used in your model script:
FEATURES_TO_EXCLUDE = [
    "I_entropy", "I_mean_slope", "I_slope_early", "I_slope_late", "I_R_time", "I_T_time", "I_QT",
    "I_ST", "I_RR", "II_mean", "II_skew", "II_rms", "II_total_power", "II_wavelet_energy",
    "II_mean_slope", "II_zero_crossings", "II_slope_early", "II_slope_late", "II_P_amp", "II_Q_amp", "II_P_time",
    "II_QRS_duration", "II_PR", "II_QT", "II_RR", "V2_mean", "V2_std", "V2_min",
    "V2_max", "V2_kurtosis", "V2_rms", "V2_entropy", "V2_total_power", "V2_dominant_freq", "V2_wavelet_energy",
    "V2_max_slope", "V2_min_slope", "V2_mean_slope", "V2_zero_crossings", "V2_Q_amp", "V2_R_amp", "V2_S_amp",
    "V2_T_amp", "V2_P_time", "V2_Q_time", "V2_R_time", "V2_S_time", "V2_T_time", "V2_QRS_area",
    "V2_QRS_duration", "V2_PR", "V2_QT", "V2_ST", "V2_RR", "age", "sex",
    "heart_axis_normal", "heart_axis_left_axis_deviation", "heart_axis_right_axis_deviation"
]

INPUT_LEADS = ["I", "II", "V2"]
lead_stats = ['mean', 'std', 'min', 'max', 'skew', 'kurtosis', 'rms', 'entropy',
              'total_power', 'dominant_freq', 'wavelet_energy',
              'max_slope', 'min_slope', 'mean_slope', 'zero_crossings', 'slope_early', 'slope_late']
ALL_FEATURE_NAMES = []
for lead in INPUT_LEADS:
    ALL_FEATURE_NAMES += [f"{lead}_{stat}" for stat in lead_stats]
    ALL_FEATURE_NAMES += [f"{lead}_{p}_amp" for p in ['P','Q','R','S','T']]
    ALL_FEATURE_NAMES += [f"{lead}_{p}_time" for p in ['P','Q','R','S','T']]
    ALL_FEATURE_NAMES += [f"{lead}_QRS_area", f"{lead}_QRS_duration"]
    ALL_FEATURE_NAMES += [f"{lead}_{iv}" for iv in ['PR','QT','ST','RR']]
ALL_FEATURE_NAMES += ['age', 'sex', 'heart_axis_normal', 'heart_axis_left_axis_deviation', 'heart_axis_right_axis_deviation']

INCLUDED_FEATURE_INDICES = [i for i, name in enumerate(ALL_FEATURE_NAMES) if name not in FEATURES_TO_EXCLUDE]

# Load one record to show sample values and mapping:
with open("RichECG_Datasets/features_test.pkl", "rb") as f:
    rec = pickle.load(f)
    # Rebuild the input vector exactly as in your data loader:
    feat = np.concatenate([rec["features"][lead] for lead in INPUT_LEADS])
    amps, times, qrs_area, qrs_dur = [], [], [], []
    for lead in INPUT_LEADS:
        wave = rec["waves"].get(lead, {})
        amps.extend([wave.get(f"{w}_amp", 0) or 0 for w in ['P','Q','R','S','T']])
        times.extend([wave.get(f"{w}_time", 0) or 0 for w in ['P','Q','R','S','T']])
        qrs_area.append(wave.get("QRS_area", 0) or 0)
        qrs_dur.append(wave.get("QRS_duration", 0) or 0)
    intervals = []
    for lead in INPUT_LEADS:
        lead_intervals = rec["intervals"].get(lead, {})
        for key in ['PR', 'QT', 'ST', 'RR']:
            val = lead_intervals.get(key)
            if isinstance(val, list) and len(val) > 0:
                intervals.append(np.mean(val) if np.isfinite(np.mean(val)) else 0)
            else:
                intervals.append(0)
    def encode_metadata(meta_dict):
        age = meta_dict.get("age", 0)
        sex = 1 if meta_dict.get("sex", "").lower() == "male" else 0
        axis = meta_dict.get("heart_axis", "").lower()
        axis_encoded = [0,0,0]
        if axis == "normal": axis_encoded[0] = 1
        elif axis == "left axis deviation": axis_encoded[1] = 1
        elif axis == "right axis deviation": axis_encoded[2] = 1
        return [age, sex] + axis_encoded
    meta_features = encode_metadata(rec["metadata"])
    full_feature_vector = np.concatenate([feat, amps, times, qrs_area, qrs_dur, intervals, meta_features])

# Print mapping for spot-check:
print("Index | Feature Name                         | Value     | Included?")
print("-" * 60)
for i, name in enumerate(ALL_FEATURE_NAMES):
    val = full_feature_vector[i] if i < len(full_feature_vector) else "NA"
    included = "YES" if i in INCLUDED_FEATURE_INDICES else "NO"
    print(f"{i:4d} | {name:35s} | {val:10.4f} | {included}")

print(f"\nTotal features: {len(ALL_FEATURE_NAMES)}")
print(f"Included: {len(INCLUDED_FEATURE_INDICES)}")
print(f"Excluded: {len(FEATURES_TO_EXCLUDE)}")

# Optionally, assert correctness
assert len(set(FEATURES_TO_EXCLUDE).intersection([ALL_FEATURE_NAMES[i] for i in INCLUDED_FEATURE_INDICES])) == 0, "Feature exclusion failed!"
