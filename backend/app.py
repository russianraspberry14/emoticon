from flask import Flask, request, jsonify
from flask_cors import CORS
import os
from ..Audio.predict_emotions import predict_emotion

app = Flask(__name__)
CORS(app)


@app.route("/upload-audio", methods=["POST"])
def upload_audio():
    if "audio" not in request.files:
        return jsonify({"error": "No audio file provided."}), 400

    file = request.files["audio"]
    filepath = os.path.join("uploads", file.filename)
    file.save(filepath)

    try:
        # Predict emotion using the imported function
        emotion = predict_emotion(filepath)
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
