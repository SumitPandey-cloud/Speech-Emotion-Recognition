import os
import pickle
import numpy as np
import librosa
from flask import Flask, request, jsonify, send_from_directory

app = Flask(__name__, static_folder="static")

weights = np.load("model_weights.npy", allow_pickle=True)

def relu(x):
    return np.maximum(0, x)

def softmax(x):
    e = np.exp(x - np.max(x))
    return e / e.sum()

def conv1d(x, W, b, stride=1):
    kernel_size = W.shape[0]
    out_len = (x.shape[0] - kernel_size) // stride + 1
    out = np.zeros((out_len, W.shape[2]))
    for i in range(out_len):
        out[i] = np.tensordot(x[i:i+kernel_size], W, axes=([0,1],[0,1])) + b
    return out

def maxpool1d(x, pool=2):
    out_len = x.shape[0] // pool
    return np.array([x[i*pool:(i+1)*pool].max(axis=0) for i in range(out_len)])

def predict_numpy(features):
    x = features.reshape(-1, 1)
    # Conv1 + pool
    x = relu(conv1d(x, weights[0], weights[1]))
    x = maxpool1d(x)
    # Conv2 + pool
    x = relu(conv1d(x, weights[2], weights[3]))
    x = maxpool1d(x)
    # Flatten
    x = x.flatten()
    # Dense1
    x = relu(x @ weights[4] + weights[5])
    # Dense2
    x = relu(x @ weights[6] + weights[7])
    # Output
    x = softmax(x @ weights[8] + weights[9])
    return x

with open("scaler.pkl", "rb") as f: scaler = pickle.load(f)
with open("encoder.pkl", "rb") as f: encoder = pickle.load(f)

EMOTION_EMOJI = {
    "neutral": "😐", "calm": "😌", "happy": "😄", "sad": "😢",
    "angry": "😡", "fearful": "😨", "disgust": "🤢", "surprised": "😲"
}

def preprocess_audio(file_path, duration=3, sr=22050):
    audio, sr = librosa.load(file_path, sr=sr, duration=duration)
    audio, _ = librosa.effects.trim(audio)
    audio = librosa.util.normalize(audio)
    desired = sr * duration
    if len(audio) > desired:
        audio = audio[:desired]
    else:
        audio = np.pad(audio, (0, desired - len(audio)))
    return audio, sr

def extract_features(file_path):
    audio, sr = preprocess_audio(file_path)
    mfcc = np.mean(librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40).T, axis=0)
    mel = np.mean(librosa.feature.melspectrogram(y=audio, sr=sr).T, axis=0)
    chroma = np.mean(librosa.feature.chroma_stft(y=audio, sr=sr).T, axis=0)
    return np.hstack((mfcc, mel, chroma))

@app.route("/")
def index():
    return send_from_directory("static", "index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    ext = os.path.splitext(file.filename)[1].lower() or ".wav"
    tmp_path = "temp_upload" + ext
    file.save(tmp_path)

    try:
        features = extract_features(tmp_path)
        features = scaler.transform([features])[0]
        probs = predict_numpy(features)
        top_idx = int(np.argmax(probs))
        top_label = encoder.classes_[top_idx]

        all_probs = [
            {
                "emotion": encoder.classes_[i],
                "emoji": EMOTION_EMOJI.get(encoder.classes_[i], ""),
                "score": round(float(probs[i]) * 100, 1)
            }
            for i in range(len(probs))
        ]
        all_probs.sort(key=lambda x: x["score"], reverse=True)

        return jsonify({
            "emotion": top_label,
            "emoji": EMOTION_EMOJI.get(top_label, ""),
            "confidence": round(float(probs[top_idx]) * 100, 1),
            "all": all_probs
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

if __name__ == "__main__":
    app.run(debug=True)