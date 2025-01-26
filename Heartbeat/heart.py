import pickle
import os
import pandas as pd
import numpy as np
from scipy.signal import butter, filtfilt, find_peaks
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import classification_report, accuracy_score, ConfusionMatrixDisplay
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier

def bandpass_filter(signal, low, high, fs):
    b, a = butter(4, [low / (0.5 * fs), high / (0.5 * fs)], btype='band')
    return filtfilt(b, a, signal)

def process_participant(participant_id, base_dir):
    file_path = os.path.join(base_dir, f"S{participant_id}/S{participant_id}.pkl")
    with open(file_path, "rb") as file:
        data = pickle.load(file, encoding='latin1')

    signals = data["signal"]
    wrist_signal = signals["wrist"]
    heart_rate = np.array(wrist_signal["BVP"]).flatten()  # BVP (64 Hz)
    accelerometer = np.array(wrist_signal["ACC"])  # Accelerometer (32 Hz)
    temperature = np.array(wrist_signal["TEMP"]).flatten()  # Temperature (4 Hz)
    labels = data["label"]  # Labels (1 Hz)

    fs_bvp = 64
    fs_acc = 32
    fs_temp = 4
    fs_label = 1

    max_time = len(labels) / fs_label
    common_time_index = np.arange(0, max_time, 1)

    heart_rate_filtered = bandpass_filter(heart_rate, low=0.5, high=4, fs=fs_bvp)
    peaks, _ = find_peaks(heart_rate_filtered, distance=fs_bvp / 2)
    rr_intervals = np.diff(peaks) / fs_bvp
    heart_rate_bpm = 60 / rr_intervals
    original_time_index_hr = peaks[:-1] / fs_bvp
    heart_rate_resampled = np.interp(common_time_index, original_time_index_hr, heart_rate_bpm)

    original_time_index_acc = np.arange(len(accelerometer)) / fs_acc
    accelerometer_resampled = np.zeros((len(common_time_index), 3))
    for i in range(3):
        accelerometer_resampled[:, i] = np.interp(common_time_index, original_time_index_acc, accelerometer[:, i])

    original_time_index_temp = np.arange(len(temperature)) / fs_temp
    temperature_resampled = np.interp(common_time_index, original_time_index_temp, temperature)

    labels_resampled = labels[:len(common_time_index)]

    accel_magnitude = np.sqrt(
        accelerometer_resampled[:, 0] ** 2 +
        accelerometer_resampled[:, 1] ** 2 +
        accelerometer_resampled[:, 2] ** 2
    )
    heart_rate_ma = pd.Series(heart_rate_resampled).rolling(window=5, min_periods=1).mean()

    participant_data = pd.DataFrame({
        'time': common_time_index,
        'heart_rate': heart_rate_resampled,
        'heart_rate_ma': heart_rate_ma,
        'accel_x': accelerometer_resampled[:, 0],
        'accel_y': accelerometer_resampled[:, 1],
        'accel_z': accelerometer_resampled[:, 2],
        'accel_magnitude': accel_magnitude,
        'temperature': temperature_resampled,
        'label': labels_resampled
    })

    participant_data['participant'] = f"S{participant_id}"
    return participant_data

base_dir = "WESAD"
all_data = pd.DataFrame()
for participant_id in range(2, 16):
    if participant_id != 12:
        print(f"Processing participant S{participant_id}...")
        participant_data = process_participant(participant_id, base_dir)
        all_data = pd.concat([all_data, participant_data], ignore_index=True)

all_data.to_csv("combined_wesad_data.csv", index=False)
print("Combined dataset saved to combined_wesad_data.csv")

def feature_engineering(data, window_size=60):
    data['hr_mean'] = data['heart_rate'].rolling(window=window_size).mean()
    data['hr_std'] = data['heart_rate'].rolling(window=window_size).std()
    data['accel_magnitude_mean'] = data['accel_magnitude'].rolling(window=window_size).mean()
    data['accel_magnitude_std'] = data['accel_magnitude'].rolling(window=window_size).std()
    data['temperature_mean'] = data['temperature'].rolling(window=window_size).mean()
    data['temperature_std'] = data['temperature'].rolling(window=window_size).std()
    data = data.dropna().reset_index(drop=True)
    return data

all_data_features = feature_engineering(all_data, window_size=60)
all_data_features.to_csv("feature_engineered_wesad_data.csv", index=False)
print("Feature-engineered dataset saved to feature_engineered_wesad_data.csv")

X = all_data_features.drop(columns=['time', 'label', 'participant'])
y = all_data_features['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

scaler = StandardScaler()
X_train_smote_scaled = scaler.fit_transform(X_train_smote)
X_test_scaled = scaler.transform(X_test)

xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42)
param_dist = {
    'n_estimators': [50, 100, 200],
    'max_depth': [3, 6, 10],
    'learning_rate': [0.01, 0.1, 0.2],
    'subsample': [0.7, 0.8, 1.0],
    'colsample_bytree': [0.7, 0.8, 1.0]
}

random_search = RandomizedSearchCV(
    estimator=xgb_model,
    param_distributions=param_dist,
    n_iter=20,
    scoring='f1_weighted',
    cv=3,
    n_jobs=-1,
    random_state=42,
    verbose=2
)

random_search.fit(X_train_smote_scaled, y_train_smote)
best_xgb_model = random_search.best_estimator_
print("\nBest Parameters:", random_search.best_params_)

y_pred = best_xgb_model.predict(X_test_scaled)
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
print("Accuracy:", accuracy_score(y_test, y_pred))

ConfusionMatrixDisplay.from_estimator(best_xgb_model, X_test_scaled, y_test)
