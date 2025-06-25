import os
import numpy as np
import mne
import scipy.io
import joblib
import warnings
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE

warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*Invalid measurement date.*")

def load_annotations(annotation_path):
    mat = scipy.io.loadmat(annotation_path)
    raw_annots = mat["annotat_new"][0]
    seizure_map = {}
    for i, annot in enumerate(raw_annots):
        if annot.shape[0] < 2:
            continue
        edf_name = f"eeg{i+1}.edf"
        mask = annot[1]
        intervals = extract_intervals(mask)
        seizure_map[edf_name] = intervals
    return seizure_map

def extract_intervals(mask_row):
    intervals = []
    in_seizure = False
    start = 0
    for i, val in enumerate(mask_row):
        if val == 1 and not in_seizure:
            start = i
            in_seizure = True
        elif val == 0 and in_seizure:
            intervals.append((start, i))
            in_seizure = False
    if in_seizure:
        intervals.append((start, len(mask_row)))
    return intervals

def extract_channels(file_path, channels=[0, 1, 2, 3]):
    try:
        raw = mne.io.read_raw_edf(file_path, preload=True, verbose=False)
        data = raw.get_data(picks=channels)
        fs = int(raw.info['sfreq'])
        return data, fs
    except Exception as e:
        print(f"Skipping file {file_path} due to error: {e}")
        return None, None

def extract_features(segment):
    feats = []
    for ch in segment:
        feats.extend([
            np.mean(ch), np.std(ch), np.min(ch), np.max(ch),
            np.percentile(ch, 25), np.percentile(ch, 75)
        ])
    return feats

def segment_and_label(data, seizure_intervals, fs, window_sec=10, preseizure_margin_sec=900):
    window_size = fs * window_sec
    total_windows = data.shape[1] // window_size
    X, y = [], []
    for i in range(total_windows):
        start = i * window_size
        end = start + window_size
        global_time = start // fs
        label = 0
        for sz_start, sz_end in seizure_intervals:
            pre_start = max(sz_start - preseizure_margin_sec, 0)
            if sz_start <= global_time < sz_end:
                label = 2
                break
            elif pre_start <= global_time < sz_start:
                label = 1
                break
        segment = data[:, start:end]
        if segment.shape[1] == window_size:
            X.append(extract_features(segment))
            y.append(label)
    return np.array(X), np.array(y)

def build_dataset(edf_folder, seizure_map):
    X_all, y_all = [], []
    for edf_file, intervals in seizure_map.items():
        edf_path = os.path.join(edf_folder, edf_file)
        if not os.path.exists(edf_path):
            print(f"Missing file: {edf_path}")
            continue
        data, fs = extract_channels(edf_path)
        if data is None:
            continue
        try:
            X, y = segment_and_label(data, intervals, fs)
            if len(X) > 0:
                X_all.append(X)
                y_all.append(y)
        except Exception as e:
            print(f"Error segmenting {edf_file}: {e}")
    if not X_all:
        return None, None
    return np.concatenate(X_all), np.concatenate(y_all)

def train_stacking_model(X, y):
    print("\nOriginal label distribution:", np.bincount(y))
    smote = SMOTE(random_state=42)
    X_res, y_res = smote.fit_resample(X, y)
    print("After SMOTE label distribution:", np.bincount(y_res))

    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X_res, y_res, stratify=y_res, test_size=0.1, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, stratify=y_trainval, test_size=0.2, random_state=42)

    base_rf = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
    base_lgbm = LGBMClassifier(n_estimators=100, learning_rate=0.1,
                               max_depth=7, class_weight='balanced',
                               objective='multiclass', num_class=3, random_state=42)

    meta_model = SVC(kernel='rbf', C=1, probability=True)

    stack = StackingClassifier(
        estimators=[('rf', base_rf), ('lgbm', base_lgbm)],
        final_estimator=meta_model,
        cv=5
    )

    stack.fit(X_train, y_train)

    print("\nValidation Report:")
    print(classification_report(y_val, stack.predict(X_val), target_names=["Non-Seizure", "Pre-Seizure", "Seizure"]))

    print("\nTest Report:")
    print(classification_report(y_test, stack.predict(X_test), target_names=["Non-Seizure", "Pre-Seizure", "Seizure"]))

    joblib.dump(stack, "stacking_model_2.pkl")
    print("Model saved as stacking_model_2.pkl")

if __name__ == "__main__":
    base_folder = "C:/Users/karti/Downloads/EEG_images/zenodo"
    annotation_file = os.path.join(base_folder, "annotations_2017.mat")

    print("Loading annotations...")
    seizure_map = load_annotations(annotation_file)

    print("Building dataset...")
    X_all, y_all = build_dataset(base_folder, seizure_map)

    if X_all is not None:
        print(f"\nFinal dataset shape: {X_all.shape}")
        train_stacking_model(X_all, y_all)
    else:
        print("No valid data extracted from EDF files.")
