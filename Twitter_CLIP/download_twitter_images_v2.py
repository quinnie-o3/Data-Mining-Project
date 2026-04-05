import csv
import os
import re
import sys
import time
from urllib.parse import urlparse

try:
    import requests
except ImportError:
    print("ERROR: requests not installed. Run: pip install requests")
    sys.exit(1)

# CONFIG
INPUT_CSV_PATH = "twitter_clip_ready.csv"
OUTPUT_CSV_PATH = "twitter_clip_ready_downloaded.csv"
IMAGE_FOLDER = "twitter_images"
REQUEST_TIMEOUT = 10
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
}
ALLOWED_EXTENSIONS = [".jpg", ".jpeg", ".png", ".webp"]
DEFAULT_EXTENSION = ".jpg"
RETRY_ATTEMPTS = 2

# END CONFIG


def log(message: str) -> None:
    print(message, flush=True)


def sanitize_filename(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_.-]", "_", value)
    return value[:200]


def get_extension(url, content_type=None):
    try:
        parsed = urlparse(url)
        root, ext = os.path.splitext(parsed.path)
        ext = ext.lower()
        if ext in ALLOWED_EXTENSIONS:
            return ext
    except:
        pass
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
    for attempt in range(RETRY_ATTEMPTS):
        try:
            response = requests.get(url, headers=HEADERS, timeout=REQUEST_TIMEOUT, stream=True)
            response.raise_for_status()
            os.makedirs(os.path.dirname(dest_path), exist_ok=True)
            with open(dest_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
            return True, "ok"
        except requests.exceptions.Timeout:
            if attempt < RETRY_ATTEMPTS - 1:
                time.sleep(2)
            else:
                return False, "timeout"
        except requests.exceptions.ConnectionError as e:
            return False, f"connection_error: {str(e)[:30]}"
        except Exception as exc:
            return False, f"{type(exc).__name__}: {str(exc)[:30]}"
    return False, "max_retries_exceeded"


def main():
    log("[download] Loading CSV...")
    fieldnames, rows = load_csv(INPUT_CSV_PATH)
    if "image_url" not in fieldnames:
        log("ERROR: image_url column is required in input CSV.")
        sys.exit(1)

    output_rows = []
    success_count = 0
    failed_count = 0

    for idx, row in enumerate(rows, start=1):
        row_id = str(row.get("id", f"row_{idx}"))
        image_url = str(row.get("image_url", "")).strip()
        local_image_path = ""
        download_ok = 0
        status = ""

        if not image_url:
            status = "missing_image_url"
            failed_count += 1
        else:
            file_name_base = sanitize_filename(row_id)
            extension = get_extension(image_url)
            sanitized_name = file_name_base + extension
            local_image_path = os.path.join(IMAGE_FOLDER, sanitized_name)
            
            # Skip if already exists
            if os.path.exists(local_image_path):
                download_ok = 1
                status = "already_exists"
                success_count += 1
            else:
                ok, status_text = download_image(image_url, local_image_path)
                download_ok = 1 if ok else 0
                status = status_text
                
                # Verify file actually exists
                if download_ok and not os.path.exists(local_image_path):
                    download_ok = 0
                    status = "download_failed_verify"
                
                if download_ok:
                    success_count += 1
                else:
                    failed_count += 1

        output_row = dict(row)
        output_row["local_image_path"] = local_image_path
        output_row["download_ok"] = download_ok
        output_row["download_status"] = status
        output_rows.append(output_row)

        if idx % 50 == 0:
            log(f"[download] Processed {idx}/{len(rows)} | Success: {success_count} | Failed: {failed_count}")

    fieldnames_out = list(fieldnames)
    if "local_image_path" not in fieldnames_out:
        fieldnames_out.append("local_image_path")
    if "download_ok" not in fieldnames_out:
        fieldnames_out.append("download_ok")
    if "download_status" not in fieldnames_out:
        fieldnames_out.append("download_status")

    save_csv(OUTPUT_CSV_PATH, output_rows, fieldnames_out)
    log(f"[download] Completed")
    log(f"[download] Saved to: {OUTPUT_CSV_PATH}")
    log(f"[download] Total: {len(rows)} | Success: {success_count} | Failed: {failed_count}")
    log(f"[download] Images folder: {os.path.abspath(IMAGE_FOLDER)}")


if __name__ == "__main__":
    main()
