import joblib
from features import load_audio, extract_features

def predict_emotion(audio_path):

    model = joblib.load('Models/model.pkl')
    scaler = joblib.load('Models/scaler.pkl')
    data, sr = load_audio(audio_path)
    
    if data is None:
        print(f"Could not process audio file: {audio_path}")
        return None
    
    features = extract_features(data, sr)
    
    if features is None:
        print(f"Feature extraction failed for: {audio_path}")
        return None
    features_scaled = scaler.transform(features.reshape(1, -1))
    prediction = model.predict(features_scaled)
    label_encoder = joblib.load('Models/label_encoder.pkl')
    emotion = label_encoder.inverse_transform(prediction)[0]
    
    return emotion

if __name__ == "__main__":
    test_file = "path/to/your/test/audio.wav"
    predicted_emotion = predict_emotion(test_file)
