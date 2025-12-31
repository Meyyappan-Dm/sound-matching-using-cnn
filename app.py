import streamlit as st
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import librosa
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import io
import os

# =========================
#  CONFIG (same as training)
# =========================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SAMPLE_RATE = 44100
N_MELS = 128
N_FFT = 2048
HOP_LENGTH = 512
CLIP_DURATION = 5.0
TARGET_LEN = int(SAMPLE_RATE * CLIP_DURATION)
TIME_FRAMES = 384
EMBED_DIM = 256

# =========================
#  LOAD MODEL & DB (same as training)
# =========================

@st.cache_resource
def load_model_and_db():
    """Load trained model and embedding database once."""
    
    # Load model
    model = AudioResNet(num_classes=50, embedding_dim=EMBED_DIM).to(DEVICE)
    model.load_state_dict(torch.load("esc50_resnet_best.pth", map_location=DEVICE))
    model.eval()
    
    # Load embedding database
    data = np.load("esc50_embeddings_optimized.npz")
    db_embeddings_norm = data["embeddings"]
    db_labels = data["labels"]
    db_files = data["files"]
    
    # Load label encoder from CSV (for displaying class names)
    df = pd.read_csv(os.path.join("ESC-50-master", "meta", "esc50.csv"))
    label_encoder = LabelEncoder()
    label_encoder.fit(df["category"].to_numpy())
    
    st.success(f"✅ Loaded model & {len(db_embeddings_norm):,} embeddings")
    return model, db_embeddings_norm, db_labels, db_files, label_encoder

# =========================
#  PREPROCESSING (same as training)
# =========================

def load_audio_fixed(audio_bytes, target_len=TARGET_LEN, sr=SAMPLE_RATE):
    """Load audio from bytes (uploaded file)."""
    y, _ = librosa.load(io.BytesIO(audio_bytes), sr=sr, mono=True)
    if len(y) < target_len:
        pad = target_len - len(y)
        y = np.pad(y, (0, pad))
    else:
        y = y[:target_len]
    return y

def audio_to_mel_spectrogram(y, sr=SAMPLE_RATE):
    S = librosa.feature.melspectrogram(
        y=y, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=N_MELS
    )
    S_db = librosa.power_to_db(S, ref=np.max)
    mean = S_db.mean()
    std = S_db.std() + 1e-6
    S_norm = (S_db - mean) / std
    return torch.tensor(S_norm, dtype=torch.float32).unsqueeze(0)

def fix_time_dim(mel, target_T=TIME_FRAMES):
    _, H, W = mel.shape
    if W < target_T:
        pad = target_T - W
        mel = F.pad(mel, (0, pad))
    else:
        mel = mel[:, :, :target_T]
    return mel

# =========================
#  MODEL DEFINITION (same as training)
# =========================

class BasicBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out

class AudioResNet(nn.Module):
    def __init__(self, num_classes, embedding_dim=EMBED_DIM):
        super().__init__()
        self.in_channels = 32
        self.conv1 = nn.Conv2d(1, 32, 7, 2, 3, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(3, 2, 1)

        self.layer1 = self._make_layer(32, 2, 1)
        self.layer2 = self._make_layer(64, 2, 2)
        self.layer3 = self._make_layer(128, 2, 2)

        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.embedding = nn.Linear(128, embedding_dim)
        self.classifier = nn.Linear(embedding_dim, num_classes)

    def _make_layer(self, out_channels, blocks, stride):
        downsample = None
        if stride != 1 or self.in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        layers = [BasicBlock(self.in_channels, out_channels, stride, downsample)]
        self.in_channels = out_channels
        for _ in range(1, blocks):
            layers.append(BasicBlock(self.in_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x, return_embedding=True):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.gap(x).view(x.size(0), -1)
        emb = self.embedding(x)
        return emb

# =========================
#  MATCHING FUNCTION
# =========================

def get_top_k_matches(query_audio_bytes, model, db_embeddings_norm, db_labels, db_files, label_encoder, k=5):
    """Process uploaded audio → return top-k matches."""
    model.eval()
    
    # Preprocess uploaded audio
    y = load_audio_fixed(query_audio_bytes)
    mel = audio_to_mel_spectrogram(y)
    mel = fix_time_dim(mel)
    mel = mel.unsqueeze(0).to(DEVICE)
    
    # Get embedding
    with torch.no_grad():
        q_emb = model(mel).squeeze(0).cpu().numpy()
        q_emb = q_emb / (np.linalg.norm(q_emb) + 1e-8)
    
    # Cosine similarity search
    sims = db_embeddings_norm @ q_emb
    idx_sorted = np.argsort(-sims)[:k]
    
    results = []
    for idx in idx_sorted:
        label_id = db_labels[idx]
        label_str = label_encoder.inverse_transform([label_id])[0]
        filename = os.path.basename(db_files[idx])
        results.append({
            "rank": len(results) + 1,
            "match_name": label_str,
            "file": filename,
            "similarity": float(sims[idx]),
            "confidence": f"{float(sims[idx]):.1%}"
        })
    return results

# =========================
#  STREAMLIT UI
# =========================

def main():
    st.set_page_config(page_title="🌀 Image-Shazam", layout="wide")
    st.title("🌀 Image-Shazam (ESC-50)")
    st.markdown("Upload any 5-second audio clip → get top-5 matches from 2000-clip database!")
    
    # Load model & DB
    with st.spinner("Loading model and database..."):
        model, db_embeddings_norm, db_labels, db_files, label_encoder = load_model_and_db()
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Choose an audio file (WAV/MP3)",
        type=['wav', 'mp3', 'm4a', 'flac'],
        help="Upload any audio file (will use first 5 seconds)"
    )
    
    if uploaded_file is not None:
        # Audio player
        st.audio(uploaded_file, format='audio/wav')
        
        # Process & match
        with st.spinner("🎵 Analyzing audio fingerprint..."):
            matches = get_top_k_matches(
                uploaded_file.read(),
                model, db_embeddings_norm, db_labels, db_files, label_encoder
            )
        
        # Results table
        st.subheader("🎯 Top 5 Matches")
        results_df = pd.DataFrame(matches)
        
        st.dataframe(
            results_df[['rank', 'match_name', 'confidence']],
            column_config={
                "rank": st.column_config.NumberColumn("Rank", format="%d"),
                "match_name": st.column_config.TextColumn("Sound Class"),
                "confidence": st.column_config.ProgressColumn("Confidence", format="%d%%")
            },
            hide_index=True,
            use_container_width=True
        )
        
        # Best match highlight
        best_match = matches[0]
        col1, col2 = st.columns([1, 3])
        with col1:
            st.metric("🎉 Best Match", best_match['match_name'], best_match['confidence'])
        with col2:
            st.info(f"**Similarity**: {best_match['similarity']:.4f}")
            st.caption(f"File: {best_match['file']}")

if __name__ == "__main__":
    main()
