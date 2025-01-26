import librosa
import numpy as np
import pandas as pd


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
        zcr = librosa.feature.zero_crossing_rate(y=data).mean()
        stft = np.abs(librosa.stft(data))
        chroma = librosa.feature.chroma_stft(S=stft, sr=sample_rate).mean(axis=1)
        mfcc = librosa.feature.mfcc(y=data, sr=sample_rate, n_mfcc=13).mean(axis=1)
        rms = librosa.feature.rms(y=data).mean()
        mel = librosa.feature.melspectrogram(y=data, sr=sample_rate).mean(axis=1)

        return np.concatenate([[zcr], chroma, mfcc, [rms], mel])
    except Exception as e:
        print(f"Feature extraction error: {e}")
        return None


def process_features(audio_df):
    X, Y = [], []
    for path, emotion in zip(audio_df.Path, audio_df.Emotions):
        data, sr = load_audio(path)
        features = extract_features(data, sr)
        if features is not None:
            X.append(features)
            Y.append(emotion)

    features_df = pd.DataFrame(X)
    features_df["labels"] = Y
    features_df.to_csv("Output/features.csv", index=False)
    return features_df


if __name__ == "__main__":
    df = pd.read_csv("Output/Audio Dataset.csv")
    process_features(df)
