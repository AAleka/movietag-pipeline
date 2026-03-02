import os
import shutil
import argparse
import random
import csv
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from sklearn.metrics import precision_score, recall_score, f1_score, hamming_loss
import matplotlib.pyplot as plt

from transformers import AutoTokenizer, AutoModel


def split_train_val(scene_to_tags, val_fraction=0.1, seed=42):
    random.seed(seed)
    scene_ids = list(scene_to_tags.keys())
    random.shuffle(scene_ids)

    n_val = int(len(scene_ids) * val_fraction)
    val_ids = set(scene_ids[:n_val])
    train_ids = set(scene_ids[n_val:])

    return (
        {k: scene_to_tags[k] for k in train_ids},
        {k: scene_to_tags[k] for k in val_ids},
    )


def load_annotations(path, tag_list):
    scene_to_tags = {}
    tag_to_idx = {t: i for i, t in enumerate(tag_list)}
    idx_to_tag = {i: t for t, i in tag_to_idx.items()}

    with open(path, encoding="utf-8") as f:
        for line in f:
            if ":" not in line:
                continue
            sid, tags = line.split(":")
            tags = [t.strip() for t in tags.split(",") if t.strip() in tag_to_idx]
            scene_to_tags[sid.strip()] = tags

    return scene_to_tags, tag_to_idx, idx_to_tag


class SceneMultiModalDataset(Dataset):
    def __init__(self, embed_dir, text_dir, scene_to_tags, tag_to_idx, tokenizer, max_len=512):
        self.embed_dir = embed_dir
        self.text_dir = text_dir
        self.scene_to_tags = scene_to_tags
        self.tag_to_idx = tag_to_idx
        self.scenes = sorted(scene_to_tags.keys())
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.num_tags = len(tag_to_idx)

        self.scenes = [
            sid for sid in self.scenes
            if os.path.exists(os.path.join(embed_dir, f"{sid}.npy"))
            and os.path.exists(os.path.join(text_dir, f"{sid}.txt"))
        ]

    def __len__(self):
        return len(self.scenes)

    def __getitem__(self, idx):
        sid = self.scenes[idx]

        frames = torch.tensor(
            np.load(os.path.join(self.embed_dir, f"{sid}.npy")),
            dtype=torch.float32
        )

        with open(os.path.join(self.text_dir, f"{sid}.txt"), encoding="utf-8") as f:
            text = f.read()

        enc = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt",
        )

        labels = torch.zeros(self.num_tags)
        for t in self.scene_to_tags[sid]:
            labels[self.tag_to_idx[t]] = 1.0

        return (
            sid,
            frames,
            enc["input_ids"].squeeze(0),
            enc["attention_mask"].squeeze(0),
            labels,
        )


def collate_fn(batch):
    sids, frames, ids, masks, labels = zip(*batch)

    max_len = max(f.shape[0] for f in frames)
    dim = frames[0].shape[1]

    padded = torch.zeros(len(frames), max_len, dim)

    for i, f in enumerate(frames):
        padded[i, :f.shape[0]] = f

    return (
        list(sids),
        padded,
        torch.stack(ids),
        torch.stack(masks),
        torch.stack(labels),
    )


class VisualTransformer(nn.Module):
    def __init__(self, dim, depth=2, heads=8, dropout=0.1):
        super().__init__()

        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.pos_emb = nn.Parameter(torch.randn(1, 2048, dim))  # max frames

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=heads,
            batch_first=True,
            dropout=dropout
        )

        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        B, N, D = x.shape

        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)

        x = x + self.pos_emb[:, :N+1]
        x = self.encoder(x)

        return self.norm(x[:, 0])


class MultiModalSceneClassifier(nn.Module):
    def __init__(self, clip_dim, num_tags):
        super().__init__()

        self.frame_encoder = VisualTransformer(clip_dim)

        self.temperature = nn.Parameter(torch.ones(1))

        self.text_encoder = AutoModel.from_pretrained("distilbert-base-uncased")
        text_dim = self.text_encoder.config.hidden_size

        self.head = nn.Sequential(
            nn.LayerNorm(clip_dim + text_dim),
            nn.Linear(clip_dim + text_dim, 512),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_tags),
        )

    def forward(self, frames, input_ids, attention_mask):
        v = self.frame_encoder(frames)

        t = self.text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        ).last_hidden_state[:, 0]

        fused = torch.cat([v, t], dim=-1)
        logits = self.head(fused)
        return logits / self.temperature


class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=0.25):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, logits, targets):
        bce = torch.nn.functional.binary_cross_entropy_with_logits(
            logits, targets, reduction="none"
        )

        probs = torch.sigmoid(logits)
        pt = torch.where(targets == 1, probs, 1 - probs)

        loss = self.alpha * (1 - pt) ** self.gamma * bce

        return loss.mean()


def save_plot(metrics_path):
    epochs = []

    train_loss = []
    val_loss = []

    precision_micro = []
    recall_micro = []
    f1_micro = []
    f1_macro = []
    accuracy = []

    with open(metrics_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            epochs.append(int(row["epoch"]))
            train_loss.append(float(row["train_loss"]))
            val_loss.append(float(row["val_loss"]))

            precision_micro.append(float(row["val_precision_micro"]))
            recall_micro.append(float(row["val_recall_micro"]))
            f1_micro.append(float(row["val_f1_micro"]))
            accuracy.append(float(row["val_hamming_acc"]))

    out_dir = os.path.dirname(metrics_path)

    plt.figure()
    plt.plot(epochs, train_loss, label="Train Loss")
    plt.plot(epochs, val_loss, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "loss_curve.png"))
    plt.close()

    plt.figure()
    plt.plot(epochs, f1_micro, label="F1")
    plt.xlabel("Epoch")
    plt.ylabel("F1 Score")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "f1_curve.png"))
    plt.close()

    plt.figure()
    plt.plot(epochs, precision_micro, label="Precision")
    plt.plot(epochs, recall_micro, label="Recall")
    plt.xlabel("Epoch")
    plt.ylabel("Score")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "precision_recall_curve.png"))
    plt.close()

    plt.figure()
    plt.plot(epochs, accuracy, label="Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "accuracy_curve.png"))
    plt.close()


def validate(model, loader, device, criterion, idx_to_tag, save_dir, epoch):
    model.eval()
    os.makedirs(save_dir, exist_ok=True)

    out_path = os.path.join(save_dir, f"valid_{epoch}.txt")

    all_true, all_pred = [], []
    total_loss = 0.0

    with torch.no_grad(), open(out_path, "w", encoding="utf-8") as f:
        pbar = tqdm(loader, desc=f"Valid {epoch}", ncols=100)
        for num, (sids, frames, ids, masks, labels) in enumerate(pbar):
            frames = frames.to(device)
            ids = ids.to(device)
            masks = masks.to(device)
            labels = labels.to(device)

            logits = model(frames, ids, masks)
            loss = criterion(logits, labels)
            total_loss += loss.item()

            probs = torch.sigmoid(logits).cpu().numpy()
            preds = (probs > 0.5).astype(int)

            all_true.append(labels.cpu().numpy())
            all_pred.append(preds)

            for i, sid in enumerate(sids):
                f.write(f"https://www.youtube.com/watch?v={sid}\n")
                f.write("\tpredicted: ")
                hits = [
                    (idx_to_tag[j], probs[i, j])
                    for j in range(probs.shape[1]) if probs[i, j] > 0.5
                ]
                if hits:
                    for t, p in sorted(hits, key=lambda x: x[1], reverse=True):
                        f.write(f"{t} ({p:.3f}) ")
                else:
                    f.write("No label")

                f.write("\n\tannotated: ")
                for j, v in enumerate(labels[i]):
                    if v == 1:
                        f.write(idx_to_tag[j] + " ")
                f.write("\n\n")

            pbar.set_postfix({"valid_loss": total_loss / (num + 1)})

    all_true = np.vstack(all_true)
    all_pred = np.vstack(all_pred)

    precision_micro = precision_score(
        all_true, all_pred, average="micro", zero_division=0
    )
    recall_micro = recall_score(
        all_true, all_pred, average="micro", zero_division=0
    )
    f1_micro = f1_score(
        all_true, all_pred, average="micro", zero_division=0
    )
    hamming_acc = 1.0 - hamming_loss(all_true, all_pred)
    val_loss = total_loss / len(loader)

    return val_loss, precision_micro, recall_micro, f1_micro, hamming_acc


def train(args):
    TAGS = (
        pd.read_excel(args.tags)["English Label"]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )
    TAGS.insert(0, "No label")
    TAGS = sorted(TAGS)

    scene_to_tags, tag_to_idx, idx_to_tag = load_annotations(args.annotations, TAGS)
    train_s2t, val_s2t = split_train_val(scene_to_tags)

    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

    train_ds = SceneMultiModalDataset(
        args.frames, args.texts, train_s2t, tag_to_idx, tokenizer
    )
    val_ds = SceneMultiModalDataset(
        args.frames, args.texts, val_s2t, tag_to_idx, tokenizer
    )

    train_loader = DataLoader(train_ds, batch_size=1, shuffle=True, collate_fn=collate_fn, num_workers=4, persistent_workers=True, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, collate_fn=collate_fn)

    sample_scene = train_ds.scenes[0]
    sample_emb = np.load(os.path.join(args.frames, f"{sample_scene}.npy"))
    clip_dim = sample_emb.shape[1]

    print(f"CLIP shape: {sample_emb.shape}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = MultiModalSceneClassifier(clip_dim, len(tag_to_idx)).to(device)

    for p in model.text_encoder.parameters():
        p.requires_grad = False

    criterion = nn.BCEWithLogitsLoss()
    # criterion = FocalLoss(gamma=2.0, alpha=0.25)
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)

    os.makedirs(args.save_dir, exist_ok=True)
    shutil.copy("train_multimodal.py", args.save_dir)
    metrics_path = os.path.join(args.save_dir, "metrics.csv")

    with open(metrics_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "epoch",
            "train_loss",
            "val_loss",
            "val_precision_micro",
            "val_recall_micro",
            "val_f1_micro",
            "val_hamming_acc"
        ])

    best = 1e9
    early_stop_count = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        loss_sum = 0

        pbar = tqdm(train_loader, desc=f"Train {epoch}", ncols=100)
        for num, (_, frames, ids, masks, labels) in enumerate(pbar):
            frames = frames.to(device)
            ids = ids.to(device)
            masks = masks.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            logits = model(frames, ids, masks)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            loss_sum += loss.item()

            pbar.set_postfix({"train_loss": loss_sum / (num + 1), "ES counter": early_stop_count})

        val_loss, precision_micro, recall_micro, f1_micro, hamming_acc = validate(
            model, val_loader, device, criterion, idx_to_tag, args.save_dir, epoch
        )

        with open(metrics_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch,
                loss_sum / len(train_loader),
                val_loss,
                precision_micro,
                recall_micro,
                f1_micro,
                hamming_acc
            ])

        torch.save(model.state_dict(), os.path.join(args.save_dir, "epoch_latest.pt"))
        save_plot(metrics_path)

        if val_loss < best:
            best = val_loss
            torch.save(model.state_dict(), os.path.join(args.save_dir, "epoch_best.pt"))

            if args.early_stop != 0:
                early_stop_count = 0

        elif args.early_stop != 0:
            early_stop_count += 1

            if early_stop_count == args.early_stop:
                    break
        
        time.sleep(10)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames", default="./extracted_data/embeddings")
    parser.add_argument("--texts", default="./extracted_data/texts")
    parser.add_argument("--annotations", default="./data/annotations.txt")
    parser.add_argument("--tags", default="./data/label_selection.xlsx")
    parser.add_argument("--save_dir", required=True)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--early_stop", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-5)

    args = parser.parse_args()

    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)

    train(args)
