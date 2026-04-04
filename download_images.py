import os
import pandas as pd
import requests
from urllib.parse import urlparse
import time

INPUT_CSV = r"D:\ARGGGG\Semester 6\DATA MINING\Pheme_CLIP\pheme_clip_ready.csv"
OUTPUT_CSV = r"D:\ARGGGG\Semester 6\DATA MINING\Pheme_CLIP\pheme_clip_ready_downloaded.csv"
IMAGE_DIR = r"D:\ARGGGG\Semester 6\DATA MINING\Pheme_CLIP\pheme_images"

os.makedirs(IMAGE_DIR, exist_ok=True)

import re


def get_extension_from_url(url):
    path = urlparse(url).path.lower()
    if path.endswith(".png"):
        return ".png"
    if path.endswith(".jpeg"):
        return ".jpeg"
    if path.endswith(".jpg"):
        return ".jpg"
    if path.endswith(".webp"):
        return ".webp"
    return ".jpg"


def safe_text(x):
    if pd.isna(x):
        return ""
    return str(x).strip()


def safe_filename(x):
    if not x:
        return ""
    s = str(x)
    s = re.sub(r"[^0-9A-Za-z_.-]", "_", s)
    return s.strip("._-") or "image"

df = pd.read_csv(INPUT_CSV)

if "id" not in df.columns or "image_link" not in df.columns:
    raise ValueError("CSV phải có ít nhất 2 cột: id và image_link")

local_paths = []
download_ok_list = []
status_list = []

total = len(df)
success = 0
failed = 0

headers = {
    "User-Agent": "Mozilla/5.0"
}

for idx, row in df.iterrows():
    tweet_id = safe_text(row["id"])
    image_url = safe_text(row["image_link"])

    if not image_url:
        local_paths.append("")
        download_ok_list.append(0)
        status_list.append("empty_url")
        failed += 1
        continue

    ext = get_extension_from_url(image_url)
    safe_id = safe_filename(tweet_id)
    save_path = os.path.join(IMAGE_DIR, f"{safe_id}{ext}")

    response = None
    attempt = 0
    max_attempts = 3
    while attempt < max_attempts:
        attempt += 1
        try:
            response = requests.get(image_url, headers=headers, timeout=20)
            if response.status_code == 200:
                ct = response.headers.get("Content-Type", "").lower()
                if "image" not in ct:
                    local_paths.append("")
                    download_ok_list.append(0)
                    status_list.append(f"wrong_content_type:{ct}")
                    failed += 1
                    break
                if not response.content:
                    local_paths.append("")
                    download_ok_list.append(0)
                    status_list.append("empty_content")
                    failed += 1
                    break
                with open(save_path, "wb") as f:
                    f.write(response.content)
                local_paths.append(save_path)
                download_ok_list.append(1)
                status_list.append("ok")
                success += 1
                break
            else:
                if attempt == max_attempts:
                    local_paths.append("")
                    download_ok_list.append(0)
                    status_list.append(f"http_{response.status_code}")
                    failed += 1
                else:
                    time.sleep(1)
                continue
        except Exception as e:
            if attempt == max_attempts:
                local_paths.append("")
                download_ok_list.append(0)
                status_list.append(f"error:{type(e).__name__}:{str(e)}")
                failed += 1
            else:
                time.sleep(1)
            continue

    if (idx + 1) % 50 == 0:
        print(f"Da xu ly {idx + 1}/{total} dong")

    time.sleep(0.2)

df["local_image_path"] = local_paths
df["download_ok"] = download_ok_list
df["download_status"] = status_list

df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")

print("=" * 50)
print("TAI ANH XONG")
print("Tong so dong:", total)
print("Tai thanh cong:", success)
print("Tai that bai:", failed)
print("Folder anh:", IMAGE_DIR)
print("File output:", OUTPUT_CSV)
print("=" * 50)