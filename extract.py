import os
import re
import whisper
import librosa
import subprocess
import torch
import clip
import cv2 as cv
import numpy as np
from tqdm import tqdm
from PIL import Image
from pathlib import Path
from transformers import pipeline

device = "cuda" if torch.cuda.is_available() else "cpu"
output_root = "data/embeddings_clip"
model_name = "ViT-L/14"  # or "ViT-B/16"
frame_stride = 4         # keep 1 every N frames

model, preprocess = clip.load(model_name, device=device)
model.eval()

DATA_DIR  = Path("extracted_data")
INPUT_DIR = Path(f"{DATA_DIR}/movies")
AUDIO_DIR = Path(f"{DATA_DIR}/audios")
TEXT_DIR  = Path(f"{DATA_DIR}/texts")
EMBED_DIR = Path(f"{DATA_DIR}/embeddings")
FRAME_DIR = Path(f"{DATA_DIR}/frames")

CINEMATIC_PROMPTS = [
    "music",
    "dramatic music",
    "applause",
    "laughter",
    "crying",
    "explosion",
    "gunshot",
    "car engine",
    "crowd noise",
    "rain",
    "thunder",
    "footsteps",
    "silence",
    "suspense"
]


def extract_frame_index(filename):
    name = Path(filename).stem
    try:
        return int(name.split("_")[-1])
    except:
        return 0


def sample_frames(scene_dir, stride):
    frames = [
        f for f in os.listdir(scene_dir)
        if f.lower().endswith((".jpg", ".png"))
    ]

    if not frames:
        return []

    frames = sorted(frames, key=extract_frame_index)
    frames = frames[::stride]

    images = []
    for f in frames:
        path = os.path.join(scene_dir, f)
        img = cv.imread(path)
        if img is None:
            continue
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        images.append((f, img))

    return images


def extract_audio(video_path, audio_path):
    if audio_path.exists():
        return

    subprocess.run(
        [
            "ffmpeg", "-y",
            "-loglevel", "error",
            "-i", str(video_path),
            "-vn",
            "-map", "a:0",
            "-ac", "1",
            "-ar", "16000",
            "-f", "wav",
            str(audio_path),
        ],
        check=True,
    )


def get_advanced_cues(classifier, audio_path):
    print(f"Analyzing {audio_path.name} with CLAP...")
    
    audio_array, _ = librosa.load(str(audio_path), sr=48000)
    
    duration = librosa.get_duration(y=audio_array, sr=48000)
    step_size = 7
    found_cues = []

    for start_sec in range(0, int(duration), step_size):
        end_sec = min(start_sec + step_size, int(duration))
        chunk = audio_array[start_sec * 48000 : end_sec * 48000]
        
        if len(chunk) < 48000: continue

        results = classifier(chunk, candidate_labels=CINEMATIC_PROMPTS)
        
        top_result = results[0]
        if top_result['score'] > 0.5:
            tag = f"<{top_result['label'].upper().replace(' ', '_')}>"
            found_cues.append((start_sec, tag))
            print(f"  [{start_sec}s] Detected: {tag} ({top_result['score']:.2f})")

    return found_cues


def extract_text_audio():
    stt_model = whisper.load_model("medium")
    cue_classifier = pipeline("zero-shot-audio-classification", model="laion/clap-htsat-fused")

    for video_path in INPUT_DIR.glob("*.mkv"):
        file_stem = video_path.stem
        audio_path = AUDIO_DIR / f"{file_stem}.mp3"
        text_path = TEXT_DIR / f"{file_stem}.txt"

        if not audio_path.exists():
            extract_audio(video_path, audio_path)

        if text_path.exists():
            continue

        stt_result = stt_model.transcribe(str(audio_path))
        
        cues = get_advanced_cues(cue_classifier, audio_path)

        final_output = []
        for segment in stt_result['segments']:
            final_output.append((segment['start'], segment['text'].strip()))
        
        for timestamp, cue_tag in cues:
            final_output.append((timestamp, cue_tag))

        final_output.sort(key=lambda x: x[0])

        with open(text_path, "w", encoding="utf-8") as f:
            for _, content in final_output:
                content = re.sub(r'\[(.*?)\]', lambda m: f"<{m.group(1).upper()}>", content)
                f.write(content + " ")
        
        print(f"Finished: {text_path}")


@torch.no_grad()
def extract_frame_embed():
    videos = [
        INPUT_DIR / f
        for f in os.listdir(INPUT_DIR)
        if f.lower().endswith((".mp4", ".mkv", ".avi"))
    ]

    print(f"Found {len(videos)} videos for CLIP embedding.")
    print(f"device: {device}")

    for video_path in tqdm(videos, desc="Processing videos"):
        video_id = video_path.stem
        output_path = EMBED_DIR / f"{video_id}.npy"
        frame_save_dir = FRAME_DIR / video_id
        frame_save_dir.mkdir(parents=True, exist_ok=True)

        if output_path.exists():
            continue

        cap = cv.VideoCapture(str(video_path))
        if not cap.isOpened():
            print(f"[Skip] cannot open {video_id}")
            continue

        frame_idx = 0
        saved_idx = 0
        batch = []
        all_embeddings = []

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % frame_stride == 0:
                rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
                pil_img = Image.fromarray(rgb)

                frame_file = frame_save_dir / f"{saved_idx:06d}.jpg"
                pil_img.save(frame_file)

                img_tensor = preprocess(pil_img)
                batch.append(img_tensor)
                saved_idx += 1

                if len(batch) == 64:
                    batch_tensor = torch.stack(batch).to(device)
                    embs = model.encode_image(batch_tensor)
                    embs = embs / embs.norm(dim=-1, keepdim=True)
                    all_embeddings.append(embs.cpu())
                    batch.clear()

            frame_idx += 1

        cap.release()

        if batch:
            batch_tensor = torch.stack(batch).to(device)
            embs = model.encode_image(batch_tensor)
            embs = embs / embs.norm(dim=-1, keepdim=True)
            all_embeddings.append(embs.cpu())

        if not all_embeddings:
            print(f"[Skip] failed embedding {video_id}")
            continue

        final_embs = torch.cat(all_embeddings, dim=0).numpy()
        np.save(output_path, final_embs)


def process_workflow():
    extract_text_audio()
    extract_frame_embed()

if __name__ == "__main__":
    AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    TEXT_DIR.mkdir(parents=True, exist_ok=True)
    EMBED_DIR.mkdir(parents=True, exist_ok=True)
    FRAME_DIR.mkdir(parents=True, exist_ok=True)

    process_workflow()