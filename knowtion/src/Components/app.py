from emoticon.Heartbeat.testing import PatchTST
from flask import Flask, jsonify, request
from flask_cors import CORS
import torch
import random
import numpy as np


# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS

# Load the trained model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
input_dim = 5  # Number of features (e.g., heart_rate, temperature, accel_x, accel_y, accel_z)
patch_size = 10
d_model = 64
n_heads = 4
num_layers = 3
num_classes = 3  # Number of moods (e.g., Happy, Stressed, Relaxed)
dropout = 0.1

model = PatchTST(input_dim, patch_size, d_model, n_heads, num_layers, num_classes, dropout).to(device)
model.load_state_dict(torch.load("model.pth", map_location=device))  # Load trained model weights
model.eval()  # Set the model to evaluation mode

# Label mapping (moods)
label_mapping = {0: "Happy", 1: "Stressed", 2: "Relaxed"}

@app.route('/simulate-data', methods=['GET'])
def simulate_data():
    # Simulate data
    data = {
        "heart_rate": random.randint(60, 120),
        "temperature": round(random.uniform(36.5, 37.5), 2),
        "accel_x": random.uniform(-1, 1),
        "accel_y": random.uniform(-1, 1),
        "accel_z": random.uniform(-1, 1),
    }
    
    # Predict the mood using the trained model
    prediction = predict_mood(data)
    return jsonify({"data": data, "prediction": prediction})

def predict_mood(features):
    # Prepare the input for the model
    input_data = np.array([
        features["heart_rate"],
        features["temperature"],
        features["accel_x"],
        features["accel_y"],
        features["accel_z"]
    ]).astype(np.float32)

    # Reshape and convert to PyTorch tensor
    input_tensor = torch.tensor(input_data).unsqueeze(0).unsqueeze(0).to(device)  # Shape: (1, 1, 5)

    # Run the model for prediction
    with torch.no_grad():
        outputs = model(input_tensor)
        _, predicted_label = torch.max(outputs, 1)

    # Convert label index to mood string
    return label_mapping[predicted_label.item()]

if __name__ == '__main__':
    app.run(debug=True, port=5000)
