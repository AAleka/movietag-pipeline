import os
import subprocess

ANNOTATIONS_FILE = "data/annotations.txt"
DOWNLOAD_DIR = "data/movies"
YTDLP_PATH = "yt-dlp.exe"

os.makedirs(DOWNLOAD_DIR, exist_ok=True)

def video_already_downloaded(video_id: str) -> bool:
    for f in os.listdir(DOWNLOAD_DIR):
        if video_id in f:
            return True
    return False

def main():
    with open(ANNOTATIONS_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or ":" not in line:
                continue

            video_id = line.split(":")[0].strip()

            youtube_url = f"https://www.youtube.com/watch?v={video_id}"

            if video_already_downloaded(video_id):
                print(f"Skipping {video_id} (already downloaded)")
                continue

            print(f"Downloading {youtube_url}")
            subprocess.run(
                [
                    YTDLP_PATH,
                    youtube_url,
                    "-f", "bestvideo+bestaudio/best",
                    "--merge-output-format", "mkv",
                    "--sleep-interval", str(5),
                    "--remux-video", "mkv",
                    "--user-agent", "Mozilla/5.0",
                    "-o", os.path.join(DOWNLOAD_DIR, "%(id)s.%(ext)s"),
                ],
                check=False
            )

if __name__ == "__main__":
    main()