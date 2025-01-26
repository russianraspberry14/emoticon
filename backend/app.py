from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import librosa
import numpy as np
import joblib

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load your model and scaler
model = joblib.load("speech_emotion_xgboost_model.pkl")
scaler = joblib.load("speech_emotion_scaler.pkl")

# Function to extract features
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
        return features
    except Exception as e:
        print(f"Feature extraction error: {e}")
        return None

@app.route("/upload-audio", methods=["POST"])
def upload_audio():
    if "audio" not in request.files:
        return jsonify({"error": "No audio file provided."}), 400

    file = request.files["audio"]
    filepath = os.path.join("uploads", file.filename)
    file.save(filepath)

    try:
        # Load the audio file
        data, sample_rate = librosa.load(filepath, sr=None, duration=2.5, offset=0.6)
        print(f"Audio loaded successfully. Sample rate: {sample_rate}")

        # Extract features
        features = extract_features(data, sample_rate)
        if features is None:
            return jsonify({"error": "Feature extraction failed."}), 500

        # Scale features and predict emotion
        features_scaled = scaler.transform([features])
        prediction = model.predict(features_scaled)[0]

        emotion_map = {
            0: "angry",
            1: "disgust",
            2: "fear",
            3: "happy",
            4: "neutral",
            5: "sad",
            6: "surprise",
        }

        emotion = emotion_map.get(prediction, "Unknown")
        print(f"Predicted emotion: {emotion}")

        return jsonify({"emotion": emotion})

    except Exception as e:
        print(f"Error processing audio: {e}")
        return jsonify({"error": "Error processing audio."}), 500
    finally:
        # Clean up the uploaded file
        os.remove(filepath)

if __name__ == "__main__":
    if not os.path.exists("uploads"):
        os.makedirs("uploads")
    app.run(host="0.0.0.0", port=5001)
