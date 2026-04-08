import os
import pickle
import numpy as np
import librosa
from flask import Flask, request, jsonify, send_from_directory
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout, Input

app = Flask(__name__, static_folder="static")

def build_model():
    model = Sequential([
        Input(shape=(180, 1)),
        Conv1D(filters=64, kernel_size=3, activation='relu'),
        MaxPooling1D(pool_size=2),
        Conv1D(filters=128, kernel_size=3, activation='relu'),
        MaxPooling1D(pool_size=2),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(8, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

model = build_model()
model.load_weights("ser_weights.weights.h5")

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

    ext = os.path.splitext(file.filename)[1].lower()
    if not ext:
        ext = ".wav"
    tmp_path = "temp_upload" + ext
    file.save(tmp_path)

    try:
        features = extract_features(tmp_path)
        features = scaler.transform([features])
        probs = model.predict(features)[0]
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