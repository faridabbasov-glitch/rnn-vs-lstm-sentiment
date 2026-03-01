# 🎬 RNN vs LSTM Sentiment Analyzer

> A full-stack web application that compares **SimpleRNN** and **LSTM** models on movie review sentiment analysis in real time.

![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=flat&logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?style=flat&logo=tensorflow&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-3.0-000000?style=flat&logo=flask&logoColor=white)
![Dataset](https://img.shields.io/badge/Dataset-IMDB%2050K-yellow?style=flat)

---

## Overview

Users type a movie review and instantly see how two different deep learning architectures - SimpleRNN and LSTM - each interpret the sentiment, with confidence scores shown side by side.

---

## 📊 Model Performance

| Model | Test Accuracy |
|-------|--------------|
| SimpleRNN | 84.35% |
| **LSTM** | **85.44%** |

> Both models trained on the IMDB 50K dataset (25K train / 25K test).

---

## Architecture

**Model A - SimpleRNN**
```
Embedding(10000, 128) → SimpleRNN(64) → Dense(1, sigmoid)
```

**Model B - LSTM**
```
Embedding(10000, 128) → LSTM(64) → Dense(1, sigmoid)
```

---

## How It Works

1. User types a movie review in the browser
2. Frontend sends text to `/predict` endpoint
3. Backend tokenizes text using the IMDB word index
4. Pads sequence to 200 tokens
5. Runs inference on both models simultaneously
6. Returns JSON with scores and labels

**API Response:**
```json
{
  "rnn_result":  { "score": 0.84, "label": "Positive" },
  "lstm_result": { "score": 0.92, "label": "Positive" }
}
```

---

## Quick Start

```bash
# 1. Clone the repo
git clone https://github.com/faridabbasov-glitch/rnn-vs-lstm-sentiment.git
cd rnn-vs-lstm-sentiment

# 2. Create virtual environment
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # Mac/Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the server
python app.py
```

Open **http://localhost:5000** in your browser.

---

## Project Structure

```
rnn-vs-lstm-sentiment/
├── app.py               # Flask backend — /predict endpoint
├── index.html           # Frontend — side-by-side comparison UI
├── simple_rnn_model.h5
├── lstm_model.h5
├── requirements.txt
└── README.md
```



---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Frontend | Vanilla HTML / CSS / JS |
| Backend | Python, Flask |
| Models | TensorFlow / Keras |
| Dataset | IMDB 50K (via `tf.keras.datasets`) |

---

## Known Limitations

Both models struggle with **negation in short sentences** - a known limitation of RNN-based architectures.

**Example:**
```
Input:  "that movie is not that good actually"
RNN  →  Positive (96% confidence)  ← overconfident, incorrect
LSTM →  Positive (78% confidence)  ← less certain, still incorrect
```

LSTM's lower confidence reflects its gating mechanism detecting conflicting signals. True negation handling requires attention-based models (e.g. BERT).

---

## Requirements

```
flask>=3.0
tensorflow>=2.13
numpy>=1.24
```
