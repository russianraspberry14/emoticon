import sys
import librosa
import numpy as np
import joblib

print("Python script started successfully!")
# Load the trained model and scaler
try:
    model = joblib.load("speech_emotion_xgboost_model.pkl")
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    sys.exit(1)

try:
    scaler = joblib.load("speech_emotion_scaler.pkl")
    print("Scaler loaded successfully.")
except Exception as e:
    print(f"Error loading scaler: {e}")
    sys.exit(1)

def extract_features(data, sample_rate):
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

        print("Features extracted successfully.")
        return features
    except Exception as e:
        print(f"Feature extraction error: {e}")
        return None

if __name__ == "__main__":
    audio_path = sys.argv[1]  # Get the file path from the command line argument

    try:
        print(f"Processing file: {audio_path}")
        data, sample_rate = librosa.load(audio_path, sr=None, duration=2.5, offset=0.6)
        print(f"Audio loaded successfully. Sample rate: {sample_rate}")

        features = extract_features(data, sample_rate)
        if features is None:
            print("Failed to extract features.")
            sys.exit(1)

        features_scaled = scaler.transform([features])  # Scale features
        print("Features scaled successfully.")

        emotion_prediction = model.predict(features_scaled)[0]  # Predict emotion
        print(f"Prediction made successfully: {emotion_prediction}")

        emotion_map = {
            0: "angry",
            1: "disgust",
            2: "fear",
            3: "happy",
            4: "neutral",
            5: "sad",
            6: "surprise",
        }

        print(emotion_map.get(emotion_prediction, "Unknown"))  # Output emotion
    except Exception as e:
        print(f"Error processing audio: {e}")
        sys.exit(1)
