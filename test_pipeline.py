import torch
import subprocess
import json
import re
import os
import clip
import argparse
import numpy as np
import cv2 as cv
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from tqdm import tqdm
from glob import glob
from collections import defaultdict
from transformers import AutoTokenizer

from model import MultiModalSceneClassifier

device = "cuda" if torch.cuda.is_available() else "cpu"

class SceneInferenceDataset(Dataset):
    def __init__(self, embed_dir, text_dir, tokenizer, max_len=512):
        self.embed_dir = Path(embed_dir)
        self.text_dir = Path(text_dir)
        self.tokenizer = tokenizer
        self.max_len = max_len

        self.ids = [
            p.stem
            for p in self.embed_dir.glob("segment_*.npy")
            if (self.text_dir / f"{p.stem}.txt").exists()
        ]
        
        self.ids.sort()
        assert len(self.ids) > 0, f"No matching npy/txt pairs found in {embed_dir}"

        first_emb = np.load(self.embed_dir / f"{self.ids[0]}.npy")
        self.clip_dim = first_emb.shape[1]

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        sid = self.ids[idx]
        frames = torch.tensor(np.load(self.embed_dir / f"{sid}.npy"), dtype=torch.float32)

        with open(self.text_dir / f"{sid}.txt", encoding="utf-8") as f:
            text = f.read()

        enc = self.tokenizer(
            text, truncation=True, padding="max_length",
            max_length=self.max_len, return_tensors="pt"
        )

        return sid, frames, enc["input_ids"].squeeze(0), enc["attention_mask"].squeeze(0)


def collate_fn(batch):
    sids, frames, ids, masks = zip(*batch)
    max_len = max(f.shape[0] for f in frames)
    dim = frames[0].shape[1]
    padded = torch.zeros(len(frames), max_len, dim)
    for i, f in enumerate(frames):
        padded[i, :f.shape[0]] = f
    return list(sids), padded, torch.stack(ids), torch.stack(masks)


def run(cmd):
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return result.stdout, result.stderr


def get_video_duration(file):
    cmd = ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "json", file]
    out, _ = run(cmd)
    return float(json.loads(out)["format"]["duration"])


def detect_silences(file):
    cmd = ["ffmpeg", "-i", file, "-af", f"silencedetect=noise={args.silence_threshold}:d={args.silence_min_duration}", "-f", "null", "-"]
    _, err = run(cmd)
    starts = [float(m.group(1)) for m in re.finditer(r"silence_start: (\d+\.?\d*)", err)]
    return starts


def split_video(input_file, segments):
    Path(args.output_dir).mkdir(exist_ok=True)
    for i, (start, end) in enumerate(tqdm(segments, desc="Splitting Video")):
        duration = end - start
        output_video = f"{args.output_dir}/segment_{i:03d}.mkv"
        output_sub = f"{args.output_dir}/segment_{i:03d}.srt"

        if not os.path.exists(output_video):
            run(["ffmpeg", "-ss", str(start), "-i", input_file, "-t", str(duration), "-c", "copy", output_video, "-y"])
        
        if not os.path.exists(output_sub):
            run(["ffmpeg", "-ss", str(start), "-i", input_file, "-t", str(duration), "-map", "0:s:0", output_sub, "-y"])

        if os.path.exists(output_sub):
            with open(output_sub, "r", encoding="utf-8") as f:
                content = f.read()
            content = re.sub(r'\d+\n\d{2}:\d{2}:\d{2},\d{3} --> \d{2}:\d{2}:\d{2},\d{3}', '', content)
            content = re.sub(r'<[^>]*>', '', content)
            clean_text = " ".join([line.strip() for line in content.splitlines() if line.strip() and not line.strip().isdigit()])
            with open(output_sub.replace(".srt", ".txt"), "w", encoding="utf-8") as f:
                f.write(clean_text)


def segment_video(args):
    duration = get_video_duration(args.input_file)
    silence_starts = detect_silences(args.input_file)
    segments = []
    current_start = 0
    while current_start < duration:
        target = current_start + args.segment_duration
        if target >= duration:
            segments.append((current_start, duration))
            break
        cut_point = min(silence_starts, key=lambda x: abs(x - target)) if silence_starts else target
        segments.append((current_start, cut_point))
        current_start = cut_point
    split_video(args.input_file, segments)


@torch.inference_mode()
def embed_segments(args):
    clip_model, clip_preprocess = clip.load("ViT-L/14", device=device)
    segments = sorted(glob(os.path.join(args.output_dir, "segment_*.mkv")))
    
    for video in tqdm(segments, desc="Embedding"):
        output_path = video.replace(".mkv", ".npy")
        if os.path.exists(output_path): continue

        cap = cv.VideoCapture(video)
        all_embs, batch, frame_idx = [], [], 0
        while True:
            ret, frame = cap.read()
            if not ret: break
            if frame_idx % args.frame_stride == 0:
                frame = cv.cvtColor(cv.resize(frame, (224, 224)), cv.COLOR_BGR2RGB)
                img = torch.from_numpy(frame).permute(2,0,1).float() / 255.0
                batch.append(clip_preprocess.transforms[-1](img))
            
            if len(batch) == args.clip_batch_size:
                with torch.amp.autocast("cuda"):
                    e = clip_model.encode_image(torch.stack(batch).to(device))
                all_embs.append((e / e.norm(dim=-1, keepdim=True)).cpu())
                batch.clear()
            frame_idx += 1
        
        if batch:
            with torch.amp.autocast("cuda"):
                e = clip_model.encode_image(torch.stack(batch).to(device))
            all_embs.append((e / e.norm(dim=-1, keepdim=True)).cpu())
        
        cap.release()
        if all_embs:
            np.save(output_path, torch.cat(all_embs).numpy())


@torch.no_grad()
def test(args):
    tags_df = pd.read_excel(args.tags)["English Label"].dropna().astype(str).unique().tolist()
    TAGS = sorted(["No label"] + tags_df)
    idx_to_tag = {i: t for i, t in enumerate(TAGS)}

    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    dataset = SceneInferenceDataset(args.output_dir, args.output_dir, tokenizer)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)

    model = MultiModalSceneClassifier(dataset.clip_dim, len(TAGS)).to(device)
    model.load_state_dict(torch.load(args.model, map_location=device))
    model.eval()

    os.makedirs(args.results_dir, exist_ok=True)

    tag_counts = defaultdict(int)
    tag_prob_sums = defaultdict(float)
    total_segments = 0
    with open(os.path.join(args.results_dir, "annotations.txt"), "w", encoding="utf-8") as f:
        for sids, frames, ids, masks in tqdm(loader, desc="Classifying"):
            logits = model(frames.to(device), ids.to(device), masks.to(device))
            probs = torch.sigmoid(logits)[0].cpu().numpy()
            preds = sorted(
                [(idx_to_tag[i], p)
                for i, p in enumerate(probs)
                if p >= args.prediction_threshold],
                key=lambda x: x[1],
                reverse=True
            )

            sid = sids[0]
            total_segments += 1

            labels_str = ", ".join(t for t, _ in preds) if preds else "No label"
            f.write(f"{sid}.mkv: {labels_str}\n")

            for tag, prob in preds:
                tag_counts[tag] += 1
                tag_prob_sums[tag] += prob

        f.write("\n" + "="*60 + "\n")
        f.write("DOMINANT TAGS\n")
        f.write("="*60 + "\n")

        dominant_tags = []

        for tag in tag_counts:
            count = tag_counts[tag]
            avg_prob = tag_prob_sums[tag] / count
            freq_ratio = count / total_segments

            dominance_score = freq_ratio * avg_prob

            dominant_tags.append(
                (tag, count, freq_ratio, avg_prob, dominance_score)
            )

        dominant_tags = [
            t for t in dominant_tags
            if t[1] >= args.min_tag_occurrances and t[4] >= args.min_tag_score
        ]

        dominant_tags.sort(key=lambda x: x[4], reverse=True)

        if dominant_tags:
            for tag, count, freq_ratio, avg_prob, score in dominant_tags:
                f.write(
                    f"{tag} | "
                    f"segments: {count}/{total_segments} | "
                    f"freq: {freq_ratio:.2f} | "
                    f"avg_conf: {avg_prob:.2f} | "
                    f"score: {score:.3f}\n"
                )
        else:
            f.write("No dominant tags detected.\n")


def main(args):
    segment_video(args)
    embed_segments(args)
    test(args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, required=True)
    parser.add_argument("--output_dir", default="segments")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--tags", default="data/label_selection.xlsx")
    parser.add_argument("--results_dir", default="results/pipeline")
    parser.add_argument("--segment_duration", default=300)
    parser.add_argument("--silence_threshold", default="-30dB")
    parser.add_argument("--prediction_threshold", default=0.5)
    parser.add_argument("--silence_min_duration", default=0.7)
    parser.add_argument("--frame_stride", default=10)
    parser.add_argument("--clip_batch_size", default=128)
    parser.add_argument("--min_tag_occurrances", default=2)
    parser.add_argument("--min_tag_score", default=0.05)

    args = parser.parse_args()
    main(args)