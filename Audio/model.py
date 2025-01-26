import pandas as pd
import numpy as np
import os
import librosa
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import warnings

warnings.filterwarnings("ignore")

Ravdess = "RAVDESS/"
Savee = "SAVEE/"
Tess = "TESS/"

ravdess_emotions = []
ravdess_paths = []
ravdess_directories = [
    d for d in os.listdir(Ravdess) if os.path.isdir(os.path.join(Ravdess, d))
]

for directories in ravdess_directories:
    actor = os.listdir(Ravdess + directories)
    for audio in actor:
        part = audio.split(".")[0]
        part = part.split("-")
        ravdess_emotions.append(int(part[2]))
        ravdess_paths.append(Ravdess + directories + "/" + audio)

Ravdess_df = pd.DataFrame(
    {
        "Path": ravdess_paths,
        "Emotions": ravdess_emotions,
    }
)

RAVDESS_EMOTIONS = {
    1: "neutral",
    2: "calm",
    3: "happy",
    4: "sad",
    5: "angry",
    6: "fear",
    7: "disgust",
    8: "surprise",
}

Ravdess_df["Emotions"] = Ravdess_df["Emotions"].map(RAVDESS_EMOTIONS)

savee_emotions = []
savee_paths = []
savee_files = [d for d in os.listdir(Savee) if os.path.isfile(os.path.join(Savee, d))]

for savee_file in savee_files:
    savee_paths.append(Savee + savee_file)
    part = savee_file.split("_")[1]
    savee_emotion = part[:-6]
    if savee_emotion=='a':
        savee_emotions.append('angry')
    elif savee_emotion=='d':
        savee_emotions.append('disgust')
    elif savee_emotion=='f':
        savee_emotions.append('fear')
    elif savee_emotion=='h':
        savee_emotions.append('happy')
    elif savee_emotion=='n':
        savee_emotions.append('neutral')
    elif savee_emotion=='sa':
        savee_emotions.append('sad')
    elif savee_emotion=='su':
        savee_emotions.append('surprise')
    else:
        savee_emotions.append('Unknown')
Savee_df = pd.DataFrame(
    {
        "Path": savee_paths,
        "Emotions": savee_emotions,
    }
)

tess_emotions = []
tess_paths = []
tess_files = [d for d in os.listdir(Tess) if os.path.isfile(os.path.join(Tess, d))]

for tess_file in tess_files:
    tess_paths.append(Tess + tess_file)
    part = tess_file.split("_")[2]
    tess_emotion = part[:-4]
    if tess_emotion=='angry':
        tess_emotions.append('angry')
    elif tess_emotion=='disgust':
        tess_emotions.append('disgust')
    elif tess_emotion=='fear':
        tess_emotions.append('fear')
    elif tess_emotion=='happy':
        tess_emotions.append('happy')
    elif tess_emotion=='neutral':
        tess_emotions.append('neutral')
    elif tess_emotion=='sad':
        tess_emotions.append('sad')
    elif tess_emotion=='ps':
        tess_emotions.append('surprise')
    else:
        tess_emotions.append('Unknown')
Tess_df = pd.DataFrame(
    {
        "Path": tess_paths,
        "Emotions": tess_emotions,
    }
)

audio_df = pd.concat([Ravdess_df, Savee_df, Tess_df], axis = 0)
audio_df = audio_df.drop(audio_df[audio_df['Emotions'] == "Unknown"].index, inplace=False)
audio_df = audio_df.drop_duplicates()
audio_df.to_csv("Audio Dataset.csv", index=False)

def load_audio(path, duration=2.5, offset=0.6):
    try:
        data, sample_rate = librosa.load(path, sr=None)
        start_sample = int(offset * sample_rate)
        end_sample = int((offset + duration) * sample_rate)
        trimmed_data = data[start_sample:end_sample]
        return trimmed_data, sample_rate
    except Exception as e:
        print(f"Error loading {path}: {e}")
        return None, None


def extract_features(data, sample_rate):
    if data is None or len(data) == 0:
        return None

    try:
        zcr = librosa.feature.zero_crossing_rate(y=data)
        zcr_mean = np.mean(zcr)

        stft = np.abs(librosa.stft(data))
        chroma_stft = librosa.feature.chroma_stft(S=stft, sr=sample_rate)
        chroma_stft_mean = np.mean(chroma_stft, axis=1)

        mfccs = librosa.feature.mfcc(y=data, sr=sample_rate, n_mfcc=13)
        mfccs_mean = np.mean(mfccs, axis=1)

        rms = librosa.feature.rms(y=data)
        rms_mean = np.mean(rms)

        mel_spec = librosa.feature.melspectrogram(y=data, sr=sample_rate)
        mel_mean = np.mean(mel_spec, axis=1)

        features = np.concatenate(
            [[zcr_mean], chroma_stft_mean, mfccs_mean, [rms_mean], mel_mean]
        )

        return features
    except Exception as e:
        print(f"Feature extraction error: {e}")
        return None


def get_features(path):
    data, sample_rate = load_audio(path)

    if data is None:
        print(f"Could not load audio from {path}")
        return None

    original_features = extract_features(data, sample_rate)

    if original_features is None:
        print(f"Could not extract features from {path}")
        return None

    return original_features


X, Y = [], []
for path, emotion in zip(audio_df.Path, audio_df.Emotions):
    feature = get_features(path)

    if feature is not None:
        X.append(feature)
        Y.append(emotion)

Features = pd.DataFrame(X)
Features["labels"] = Y

Features.to_csv("features.csv", index=False)

X = Features.iloc[:, :-1].values
Y = Features["labels"].values

label_encoder = LabelEncoder()
Y = label_encoder.fit_transform(Y)

x_train, x_test, y_train, y_test = train_test_split(
    X, Y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
param_grid = {
    "n_estimators": [100, 200, 300, 400, 500],
    "learning_rate": [0.01, 0.03, 0.05, 0.07, 0.1, 0.2],
    "max_depth": [3, 4, 5, 6, 7, 8],
    "min_child_weight": [1, 2, 3, 4, 5],
    "subsample": [0.6, 0.7, 0.8, 0.9, 1.0],
    "colsample_bytree": [0.6, 0.7, 0.8, 0.9, 1.0],
    "gamma": [0, 0.1, 0.2, 0.3, 0.4, 0.5],
    "reg_alpha": [0, 0.01, 0.1, 1],
    "reg_lambda": [1, 0.1, 0.01, 0],
}

model = XGBClassifier(random_state=1, verbosity=0)
search = RandomizedSearchCV(
    estimator=model,
    param_distributions=param_grid,
    n_iter=100,
    cv=5,
    scoring="accuracy",
    n_jobs=-1,
    random_state=1,
)

search.fit(x_train, y_train)

print(f"Best Parameters: {search.best_params_}")
print(f"Best Score: {search.best_score_:.2f}")

y_pred = search.predict(x_test)

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    xticklabels=label_encoder.classes_,
    yticklabels=label_encoder.classes_,
)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig("confusion_matrix_xgboost.png")
plt.close()
joblib.dump(search.best_estimator_, "speech_emotion_xgboost_model.pkl")
joblib.dump(scaler, "speech_emotion_scaler.pkl")
