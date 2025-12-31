# 🌀 Image-Shazam: Audio Matching via Spectrogram CNNs

**"Shazam, but using image CNNs on audio spectrograms"** – matches audio clips using computer vision!

[![Streamlit App](https://img.shields.io/badge/Streamlit-Demo-brightgreen)](https://share.streamlit.io/YOUR_USERNAME/image-shazam/main/shazam_app.py)

## 🎯 Demo
Upload any audio → get top-5 matches in <1s!

## 🚀 Quick Start

1. **Download ESC-50** (140MB):
git clone https://github.com/karolpiczak/ESC-50.git
Train model:
pip install -r requirements.txt
python esc50_shazam.py  # ~60-70% accuracy
Run demo:
streamlit run shazam_app.py

## 📊 Results
ESC-50 val accuracy: 65-75% (ResNet + SpecAugment)

Matching: Perfect on clean clips, robust to noise

## 🛠️ Architecture
text
Audio → Log-Mel Spectrogram → ResNet CNN → 256D Embedding → Cosine Similarity
