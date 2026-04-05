import csv
import os
import sys

# CONFIG
INPUT_CSV_PATH = "twitter_clip_ready_downloaded.csv"
OUTPUT_CSV_PATH = "twitter_clip_final.csv"

# END CONFIG


def log(message: str) -> None:
    print(message, flush=True)


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


def has_text(row):
    text = str(row.get("tweetText", "")).strip()
    return bool(text)


def main():
    log("[filter] Loading downloaded CSV...")
    fieldnames, rows = load_csv(INPUT_CSV_PATH)
    required = ["local_image_path", "download_ok"]
    if not all(col in fieldnames for col in required):
        missing = [col for col in required if col not in fieldnames]
        log(f"ERROR: Missing required columns: {missing}")
        sys.exit(1)

    filtered = []
    removed_count = 0

    for row in rows:
        local_path = str(row.get("local_image_path", "")).strip()
        download_ok = str(row.get("download_ok", "")).strip()
        if download_ok not in {"1", "True", "true", "TRUE", "yes", "Yes"}:
            removed_count += 1
            continue
        if not local_path or not os.path.exists(local_path):
            removed_count += 1
            continue
        if not has_text(row):
            removed_count += 1
            continue
        filtered.append(row)

    save_csv(OUTPUT_CSV_PATH, filtered, fieldnames)
    log(f"[filter] Kept {len(filtered)} rows from {len(rows)}")
    log(f"[filter] Removed {removed_count} invalid rows")
    log(f"[filter] Saved final dataset to: {OUTPUT_CSV_PATH}")


if __name__ == "__main__":
    main()
