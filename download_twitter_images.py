import csv
import os
import re
import sys
from urllib.parse import urlparse

import requests

# CONFIG
INPUT_CSV_PATH = "twitter_clip_ready.csv"
OUTPUT_CSV_PATH = "twitter_clip_ready_downloaded.csv"
IMAGE_FOLDER = "twitter_images"
REQUEST_TIMEOUT = 15
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
}
ALLOWED_EXTENSIONS = [".jpg", ".jpeg", ".png", ".webp"]
DEFAULT_EXTENSION = ".jpg"

# END CONFIG


def log(message: str) -> None:
    print(message, flush=True)


def sanitize_filename(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_.-]", "_", value)
    return value[:200]


def get_extension(url, content_type=None):
    parsed = urlparse(url)
    root, ext = os.path.splitext(parsed.path)
    ext = ext.lower()
    if ext in ALLOWED_EXTENSIONS:
        return ext
    if content_type:
        if "jpeg" in content_type:
            return ".jpg"
        if "png" in content_type:
            return ".png"
        if "webp" in content_type:
            return ".webp"
    return DEFAULT_EXTENSION


def load_csv(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Input CSV not found: {path}")
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        return reader.fieldnames or [], list(reader)


def save_csv(path, rows, fieldnames):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def download_image(url, dest_path):
    try:
        response = requests.get(url, headers=HEADERS, timeout=REQUEST_TIMEOUT, stream=True)
        response.raise_for_status()
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)
        with open(dest_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        return True, "ok"
    except Exception as exc:
        return False, str(exc)


def main():
    log("[download] Loading CSV...")
    fieldnames, rows = load_csv(INPUT_CSV_PATH)
    if "image_url" not in fieldnames:
        log("ERROR: image_url column is required in input CSV.")
        sys.exit(1)

    has_id = "id" in fieldnames
    output_rows = []

    for idx, row in enumerate(rows, start=1):
        row_id = str(row.get("id", "")).strip() if has_id else str(idx)
        image_url = str(row.get("image_url", "")).strip()
        local_image_path = ""
        download_ok = 0
        status = ""

        if not image_url:
            status = "missing_image_url"
        else:
            file_name_base = row_id if row_id else f"row_{idx}"
            file_name_base = sanitize_filename(file_name_base)
            extension = get_extension(image_url)
            sanitized_name = sanitize_filename(file_name_base) + extension
            local_image_path = os.path.join(IMAGE_FOLDER, sanitized_name)
            ok, status_text = download_image(image_url, local_image_path)
            download_ok = 1 if ok else 0
            status = status_text
            if ok and not os.path.exists(local_image_path):
                download_ok = 0
                status = "downloaded_but_missing"

        output_row = dict(row)
        output_row["local_image_path"] = local_image_path
        output_row["download_ok"] = download_ok
        output_row["download_status"] = status
        output_rows.append(output_row)

        if idx % 20 == 0:
            log(f"[download] Processed {idx}/{len(rows)} rows...")

    fieldnames_out = list(fieldnames)
    if "local_image_path" not in fieldnames_out:
        fieldnames_out.append("local_image_path")
    if "download_ok" not in fieldnames_out:
        fieldnames_out.append("download_ok")
    if "download_status" not in fieldnames_out:
        fieldnames_out.append("download_status")

    save_csv(OUTPUT_CSV_PATH, output_rows, fieldnames_out)
    log(f"[download] Saved download report to: {OUTPUT_CSV_PATH}")
    log(f"[download] Images saved under: {os.path.abspath(IMAGE_FOLDER)}")


if __name__ == "__main__":
    main()
