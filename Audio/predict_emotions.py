import os
import joblib
from features import load_audio, extract_features

def predict_emotion(audio_path, model_path='Models/model.pkl', scaler_path='Models/scaler.pkl'):

    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
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

def predict_emotions_in_directory(directory):
    results = {}
    for filename in os.listdir(directory):
        if filename.endswith('.wav'):
            filepath = os.path.join(directory, filename)
            emotion = predict_emotion(filepath)
            if emotion:
                results[filename] = emotion
    return results

if __name__ == "__main__":
    test_file = "path/to/your/test/audio.wav"
    predicted_emotion = predict_emotion(test_file)
    print(f"Predicted Emotion: {predicted_emotion}")
