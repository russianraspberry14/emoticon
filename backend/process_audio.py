import sys
from ..Audio.predict_emotions import predict_emotion

if __name__ == "__main__":
    audio_path = sys.argv[1]
    predict_emotion(audio_path)
