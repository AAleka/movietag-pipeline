import os
import shutil
import argparse
import random
import csv
import time
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from sklearn.metrics import precision_score, recall_score, f1_score, hamming_loss
from transformers import AutoTokenizer

from model import MultiModalSceneClassifier


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


def save_plot(metrics_path):
    epochs = []

    train_loss = []
    val_loss = []

    precision_micro = []
    recall_micro = []
    f1_micro = []
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


def tune_thresholds(all_true, all_probs, min_support=5, min_precision=0.6, min_threshold=0.3, alpha=0.7):
    thresholds = []

    for i in range(all_true.shape[1]):

        y_true = all_true[:, i]
        support = y_true.sum()

        if support < min_support:
            thresholds.append(0.5)
            continue

        best_score = 0
        best_t = 0.5

        for t in np.linspace(0.1, 0.8, 30):

            preds = (all_probs[:, i] > t).astype(int)
            precision = precision_score(y_true, preds, zero_division=0)
            if precision < min_precision:
                continue

            f1 = f1_score(y_true, preds, zero_division=0)

            if f1 > best_score:
                best_score = f1
                best_t = t

        best_t = max(best_t, min_threshold)
        best_t = alpha * 0.5 + (1 - alpha) * best_t
        thresholds.append(best_t)

    return np.array(thresholds)


def validate(model, loader, device, criterion, idx_to_tag, save_dir, epoch):
    model.eval()
    os.makedirs(save_dir, exist_ok=True)

    out_path = os.path.join(save_dir, f"valid_{epoch}.txt")

    all_true, all_probs = [], []
    total_loss = 0.0
    all_sids = []

    with torch.no_grad():
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

            all_true.append(labels.cpu().numpy())
            all_probs.append(probs)
            all_sids.extend(sids)

            pbar.set_postfix({"valid_loss": total_loss / (num + 1)})

    all_true = np.vstack(all_true)
    all_probs = np.vstack(all_probs)

    thresholds = tune_thresholds(all_true, all_probs)
    all_pred = (all_probs > thresholds).astype(int)

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

    with open(out_path, "w", encoding="utf-8") as f:
        for i, sid in enumerate(all_sids):

            f.write(f"https://www.youtube.com/watch?v={sid}\n")
            f.write("\tpredicted: ")

            hits = [
                (idx_to_tag[j], all_probs[i, j])
                for j in range(all_probs.shape[1])
                if all_probs[i, j] > thresholds[j]
            ]

            if hits:
                for t, p in sorted(hits, key=lambda x: x[1], reverse=True):
                    f.write(f"{t} ({p:.3f}) ")
            else:
                f.write("No label")

            f.write("\n\tannotated: ")
            for j, v in enumerate(all_true[i]):
                if v == 1:
                    f.write(idx_to_tag[j] + " ")

            f.write("\n\n")

    return val_loss, precision_micro, recall_micro, f1_micro, hamming_acc, thresholds


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

    # criterion = nn.BCEWithLogitsLoss()
    criterion = FocalLoss(gamma=2.0, alpha=0.25)
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)

    os.makedirs(args.save_dir, exist_ok=True)
    shutil.copy("train_multimodal.py", args.save_dir)
    shutil.copy("test_pipeline.py", args.save_dir)
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

        val_loss, precision_micro, recall_micro, f1_micro, hamming_acc, thresholds = validate(
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
            np.save(os.path.join(args.save_dir, "best_thresholds.npy"), thresholds)

            if args.early_stop != 0:
                early_stop_count = 0

        elif args.early_stop != 0:
            early_stop_count += 1
            if early_stop_count == args.early_stop:
                break
        
        time.sleep(5)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames", default="./extracted_data/embeddings")
    parser.add_argument("--texts", default="./extracted_data/texts")
    parser.add_argument("--annotations", default="./data/annotations.txt")
    parser.add_argument("--tags", default="./data/label_selection.xlsx")
    parser.add_argument("--save-dir", required=True)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--early_stop", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-5)

    args = parser.parse_args()

    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)

    train(args)
