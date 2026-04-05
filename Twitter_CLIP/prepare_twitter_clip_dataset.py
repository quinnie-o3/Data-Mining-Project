import csv
import os
import sys
from collections import OrderedDict

# CONFIG
INPUT_CSV_PATH = "[TWITTER] tweet_with_image_url_checked.xlsx - Sheet1.csv"
OUTPUT_CSV_PATH = "twitter_clip_ready.csv"

# END CONFIG

REQUIRED_TEXT_COLUMNS = ["tweetText"]
EXPECTED_OUTPUT_COLUMNS = ["id", "tweetText", "image_url", "label"]


def log(message: str) -> None:
    print(message, flush=True)


def error(message: str) -> None:
    print(f"ERROR: {message}", flush=True)


def normalize_label(value):
    if value is None:
        return ""
    value = str(value).strip().lower()
    if value in {"fake", "real"}:
        return value
    return value


def get_text(row):
    for col in REQUIRED_TEXT_COLUMNS:
        if col in row and str(row[col]).strip():
            return str(row[col]).strip()
    return ""


def load_csv(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Input CSV not found: {path}")
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        return reader.fieldnames or [], rows


def save_csv(path, rows, fieldnames):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main():
    log("[prepare] Reading input CSV...")
    fieldnames, rows = load_csv(INPUT_CSV_PATH)

    if "tweetText" not in fieldnames:
        error("Input CSV must contain the tweetText column.")
        sys.exit(1)

    if "image_url" not in fieldnames:
        error("Input CSV must contain the image_url column.")
        sys.exit(1)

    has_id = "id" in fieldnames
    if not has_id:
        log("[prepare] Warning: id column not found. Output will keep row order and warnings will be printed.")

    cleaned_rows = []
    seen_ids = set()
    dropped_empty_text = 0
    dropped_empty_image = 0
    dropped_duplicate_id = 0

    for index, row in enumerate(rows, start=1):
        text = str(row.get("tweetText", "")).strip()
        image_url = str(row.get("image_url", "")).strip()
        row_id = str(index) if not has_id else ""

        if not text:
            dropped_empty_text += 1
            continue

        if not image_url:
            dropped_empty_image += 1
            continue

        if has_id and row_id:
            if row_id in seen_ids:
                dropped_duplicate_id += 1
                continue
            seen_ids.add(row_id)

        output_row = OrderedDict()
        output_row["id"] = row_id
        output_row["tweetText"] = text
        output_row["image_url"] = image_url
        output_row["label"] = normalize_label(row.get("label", "")) if "label" in fieldnames else ""

        cleaned_rows.append(output_row)

    log(f"[prepare] Total rows read: {len(rows)}")
    log(f"[prepare] Rows kept: {len(cleaned_rows)}")
    log(f"[prepare] Dropped empty text: {dropped_empty_text}")
    log(f"[prepare] Dropped empty image_link: {dropped_empty_image}")
    if has_id:
        log(f"[prepare] Dropped duplicate ids: {dropped_duplicate_id}")

    save_csv(OUTPUT_CSV_PATH, cleaned_rows, EXPECTED_OUTPUT_COLUMNS)
    log(f"[prepare] Saved cleaned dataset to: {OUTPUT_CSV_PATH}")


if __name__ == "__main__":
    main()
