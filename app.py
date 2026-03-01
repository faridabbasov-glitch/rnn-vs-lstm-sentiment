from flask import Flask, request, jsonify, send_from_directory
import numpy as np
import os
import json

import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences

app = Flask(__name__, static_folder=".")


MAX_FEATURES = 10000
MAX_LEN      = 200
MODEL_DIR    = os.path.dirname(os.path.abspath(__file__))


print("Loading models …")
rnn_model  = load_model(os.path.join(MODEL_DIR, "simple_rnn_model.h5"))
lstm_model = load_model(os.path.join(MODEL_DIR, "lstm_model.h5"))
print("Both models loaded.")


print("Building word index …")
word_index = imdb.get_word_index()
word_index = {w: (i + 3) for w, i in word_index.items()}
word_index["<PAD>"]     = 0
word_index["<START>"]   = 1
word_index["<UNK>"]     = 2
word_index["<UNUSED>"]  = 3
print("✔  Word index ready.")


def text_to_sequence(text: str) -> np.ndarray:
    """Tokenise → clip to MAX_FEATURES → pad to MAX_LEN."""
    tokens = text.lower().split()
    seq = [word_index.get(w, 2) for w in tokens]           
    seq = [min(idx, MAX_FEATURES - 1) for idx in seq]      
    padded = pad_sequences([seq], maxlen=MAX_LEN, padding="pre", truncating="pre")
    return padded


def score_to_label(score: float) -> str:
    return "Positive" if score >= 0.5 else "Negative"


@app.route("/")
def index():
    return send_from_directory(".", "index.html")


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True)
    if not data or "text" not in data:
        return jsonify({"error": "Missing 'text' field in request body."}), 400

    text = data["text"].strip()
    if not text:
        return jsonify({"error": "Text cannot be empty."}), 400

    try:
        seq = text_to_sequence(text)

        rnn_score  = float(rnn_model.predict(seq,  verbose=0)[0][0])
        lstm_score = float(lstm_model.predict(seq, verbose=0)[0][0])

        return jsonify({
            "rnn_result": {
                "score": round(rnn_score, 4),
                "label": score_to_label(rnn_score)
            },
            "lstm_result": {
                "score": round(lstm_score, 4),
                "label": score_to_label(lstm_score)
            }
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/health")
def health():
    return jsonify({"status": "ok", "models": ["SimpleRNN", "LSTM"]})


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)