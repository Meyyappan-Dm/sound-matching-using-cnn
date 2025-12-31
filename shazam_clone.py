import os
import numpy as np
import pandas as pd
import librosa
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

# =========================
#  CONFIG
# =========================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

SAMPLE_RATE = 44100        # native ESC-50 SR [web:46][web:85]
N_MELS = 128
N_FFT = 2048
HOP_LENGTH = 512
CLIP_DURATION = 5.0
TARGET_LEN = int(SAMPLE_RATE * CLIP_DURATION)

TIME_FRAMES = 384          # more frames: better time resolution [web:82]
BATCH_SIZE = 32
EPOCHS = 40
LR = 3e-4
WEIGHT_DECAY = 1e-4
EMBED_DIM = 256            # slightly larger embedding
DROPOUT_P = 0.3

ESC50_ROOT = "ESC-50-master"


# =========================
#  AUDIO UTILS
# =========================

def load_audio_fixed(path, target_len=TARGET_LEN, sr=SAMPLE_RATE):
    y, orig_sr = librosa.load(path, sr=sr, mono=True)  # resample & mono [web:21]
    if len(y) < target_len:
        pad = target_len - len(y)
        y = np.pad(y, (0, pad))
    else:
        y = y[:target_len]
    return y


def audio_to_mel_spectrogram(y, sr=SAMPLE_RATE):
    S = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS,
        power=2.0,
    )  # (n_mels, time) [web:21][web:77]
    S_db = librosa.power_to_db(S, ref=np.max)  # log-mel [web:34]

    # global per-clip normalization
    mean = S_db.mean()
    std = S_db.std() + 1e-6
    S_norm = (S_db - mean) / std

    mel = torch.tensor(S_norm, dtype=torch.float32).unsqueeze(0)  # (1, n_mels, time)
    return mel


def fix_time_dim(mel, target_T=TIME_FRAMES):
    _, H, W = mel.shape
    if W < target_T:
        pad = target_T - W
        mel = F.pad(mel, (0, pad))
    else:
        mel = mel[:, :, :target_T]
    return mel

#  SPEC AUGMENTATION
class SpecAugment(nn.Module):
    """
    Simple SpecAugment: time masking + frequency masking on log-mel. [web:81][web:89]
    """

    def __init__(self, freq_masks=2, time_masks=2, freq_max=20, time_max=40, p=0.5):
        super().__init__()
        self.freq_masks = freq_masks
        self.time_masks = time_masks
        self.freq_max = freq_max
        self.time_max = time_max
        self.p = p

    def forward(self, x):
        # x: (B, 1, n_mels, T)
        if not self.training or torch.rand(1).item() > self.p:
            return x

        B, C, Freq, Time = x.shape
        for b in range(B):
            # frequency masks
            for _ in range(self.freq_masks):
                f = np.random.randint(0, self.freq_max)
                f0 = np.random.randint(0, max(1, Freq - f))
                x[b, :, f0:f0 + f, :] = 0.0

            # time masks
            for _ in range(self.time_masks):
                t = np.random.randint(0, self.time_max)
                t0 = np.random.randint(0, max(1, Time - t))
                x[b, :, :, t0:t0 + t] = 0.0

        return x

spec_augment = SpecAugment()
#  ESC-50 DATASET
class ESC50Dataset(Dataset):
    def __init__(self, csv_df, audio_root, label_encoder):
        self.csv_df = csv_df.reset_index(drop=True)
        self.audio_root = audio_root
        self.le = label_encoder

    def __len__(self):
        return len(self.csv_df)

    def __getitem__(self, idx):
        row = self.csv_df.iloc[idx]
        filename = row["filename"]
        label_str = row["category"]
        label = self.le.transform([label_str])[0]

        path = os.path.join(self.audio_root, filename)
        y = load_audio_fixed(path)
        mel = audio_to_mel_spectrogram(y)
        mel = fix_time_dim(mel, TIME_FRAMES)
        return mel, label

#  RESNET-LIKE CNN

class BasicBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out


class AudioResNet(nn.Module):
    """
    Light ResNet-style model for ESC-50, with dropout and GAP. [web:69][web:88]
    """

    def __init__(self, num_classes, embedding_dim=EMBED_DIM, dropout_p=DROPOUT_P):
        super().__init__()
        self.in_channels = 32

        self.conv1 = nn.Conv2d(1, 32, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(32, 2, stride=1)
        self.layer2 = self._make_layer(64, 2, stride=2)
        self.layer3 = self._make_layer(128, 2, stride=2)

        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(dropout_p)
        self.embedding = nn.Linear(128, embedding_dim)
        self.classifier = nn.Linear(embedding_dim, num_classes)

    def _make_layer(self, out_channels, blocks, stride):
        downsample = None
        if stride != 1 or self.in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

        layers = [BasicBlock(self.in_channels, out_channels, stride, downsample)]
        self.in_channels = out_channels
        for _ in range(1, blocks):
            layers.append(BasicBlock(self.in_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, x, return_embedding=False, use_specaugment=False):
        # x: (B, 1, N_MELS, T)
        if use_specaugment:
            x = spec_augment(x)

        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.gap(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        emb = self.embedding(x)

        if return_embedding:
            return emb

        logits = self.classifier(emb)
        return logits

#  TRAIN LOOP WITH COSINE LR

def run_epoch(model, loader, criterion, optimizer=None, scheduler=None, train=True):
    model.train(train)

    total_loss = 0.0
    correct = 0
    total = 0

    for mel, labels in loader:
        mel = mel.to(DEVICE)
        labels = labels.to(DEVICE)

        if train:
            optimizer.zero_grad()

        logits = model(mel, use_specaugment=train)
        loss = criterion(logits, labels)

        if train:
            loss.backward()
            optimizer.step()
            if scheduler is not None:
                scheduler.step()

        total_loss += loss.item() * mel.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += mel.size(0)

    avg_loss = total_loss / total
    acc = correct / total
    return avg_loss, acc


def build_embedding_index(model, df, audio_root, label_encoder):
    model.eval()
    db_embeddings = []
    db_labels = []
    db_files = []

    with torch.no_grad():
        for _, row in df.iterrows():
            filename = row["filename"]
            label_str = row["category"]
            label = label_encoder.transform([label_str])[0]

            path = os.path.join(audio_root, filename)
            y = load_audio_fixed(path)
            mel = audio_to_mel_spectrogram(y)
            mel = fix_time_dim(mel, TIME_FRAMES)
            mel = mel.unsqueeze(0).to(DEVICE)

            emb = model(mel, return_embedding=True)
            emb = emb.squeeze(0).cpu().numpy()

            db_embeddings.append(emb)
            db_labels.append(label)
            db_files.append(path)

    db_embeddings = np.stack(db_embeddings, axis=0)
    db_embeddings_norm = db_embeddings / (np.linalg.norm(db_embeddings, axis=1, keepdims=True) + 1e-8)
    return db_embeddings_norm, db_labels, db_files


def embed_clip_path(model, path):
    model.eval()
    y = load_audio_fixed(path)
    mel = audio_to_mel_spectrogram(y)
    mel = fix_time_dim(mel, TIME_FRAMES)
    mel = mel.unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        emb = model(mel, return_embedding=True)
    emb = emb.squeeze(0).cpu().numpy()
    emb = emb / (np.linalg.norm(emb) + 1e-8)
    return emb


def top_k_matches(query_path, model, db_embeddings_norm, db_labels, db_files, label_encoder, k=5):
    q_emb = embed_clip_path(model, query_path)
    sims = db_embeddings_norm @ q_emb
    idx_sorted = np.argsort(-sims)[:k]

    results = []
    for idx in idx_sorted:
        label_id = db_labels[idx]
        label_str = label_encoder.inverse_transform([label_id])[0]
        results.append({
            "file": db_files[idx],
            "class": label_str,
            "similarity": float(sims[idx]),
        })
    return results


def main():
    meta_path = os.path.join(ESC50_ROOT, "meta", "esc50.csv")
    df = pd.read_csv(meta_path)  # [web:69]

    label_encoder = LabelEncoder()
    label_encoder.fit(df["category"].to_numpy())
    num_classes = len(label_encoder.classes_)
    print("Num classes:", num_classes)

    # standard: folds 1-4 train, 5 val [web:54][web:85]
    train_df = df[df["fold"] != 5].reset_index(drop=True)
    val_df = df[df["fold"] == 5].reset_index(drop=True)

    audio_root = os.path.join(ESC50_ROOT, "audio")
    train_ds = ESC50Dataset(train_df, audio_root, label_encoder)
    val_ds = ESC50Dataset(val_df, audio_root, label_encoder)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    model = AudioResNet(num_classes=num_classes, embedding_dim=EMBED_DIM, dropout_p=DROPOUT_P).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    # Cosine schedule with warmup (simple version)
    total_steps = EPOCHS * len(train_loader)
    warmup_steps = int(0.1 * total_steps)

    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        progress = (step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return 0.5 * (1.0 + np.cos(np.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    global_step = 0
    best_val_acc = 0.0

    train_losses, val_losses = [], []
    train_accs, val_accs = [], []

    for epoch in range(EPOCHS):
        # manual scheduler stepping in run_epoch
        def step_scheduler():
            nonlocal global_step
            global_step += 1
            scheduler.step()

        train_loss, train_acc = run_epoch(
            model, train_loader, criterion,
            optimizer=optimizer, scheduler=None, train=True
        )
        # step scheduler outside to keep it simple
        for _ in range(len(train_loader)):
            step_scheduler()

        val_loss, val_acc = run_epoch(
            model, val_loader, criterion,
            optimizer=None, scheduler=None, train=False
        )

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        print(
            f"Epoch {epoch+1}/{EPOCHS} "
            f"train_loss={train_loss:.4f} acc={train_acc:.3f} "
            f"val_loss={val_loss:.4f} acc={val_acc:.3f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "esc50_resnet_best.pth")
            print("Saved new best model with acc:", best_val_acc)

    # plots
    plt.figure()
    plt.plot(train_losses, label="train_loss")
    plt.plot(val_losses, label="val_loss")
    plt.legend()
    plt.title("Loss")
    plt.savefig("loss_optimized.png")

    plt.figure()
    plt.plot(train_accs, label="train_acc")
    plt.plot(val_accs, label="val_acc")
    plt.legend()
    plt.title("Accuracy")
    plt.savefig("acc_optimized.png")

    print("Best val acc:", best_val_acc)

    # build embedding index on full df using best model
    model.load_state_dict(torch.load("esc50_resnet_best.pth", map_location=DEVICE))
    db_embeddings_norm, db_labels, db_files = build_embedding_index(model, df, audio_root, label_encoder)
    np.savez(
        "esc50_embeddings_optimized.npz",
        embeddings=db_embeddings_norm,
        labels=np.array(db_labels),
        files=np.array(db_files)
    )

    example_query_path = os.path.join(audio_root, val_df.iloc[0]["filename"])
    matches = top_k_matches(
        example_query_path,
        model,
        db_embeddings_norm,
        db_labels,
        db_files,
        label_encoder,
        k=5,
    )
    print("Example query:", example_query_path)
    for m in matches:
        print(m)


if __name__ == "__main__":
    main()
