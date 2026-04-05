import csv
import os
import sys
from typing import List

import numpy as np
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

# CONFIG
INPUT_CSV_PATH = "twitter_clip_final.csv"
METADATA_OUTPUT_CSV = "clip_metadata.csv"
TEXT_FEATURES_NPY = "clip_text_features.npy"
IMAGE_FEATURES_NPY = "clip_image_features.npy"
MODEL_NAME = "openai/clip-vit-base-patch32"
MAX_ROWS = None  # set to integer for quick testing

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


def get_text_source(row):
    return str(row.get("tweetText", "")).strip()


def normalize_vector(vector: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vector)
    if norm == 0 or np.isnan(norm):
        return vector.astype(np.float32)
    return (vector / norm).astype(np.float32)


def load_image(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Image not found: {path}")
    return Image.open(path).convert("RGB")


def main():
    log("[extract] Reading final CSV...")
    fieldnames, rows = load_csv(INPUT_CSV_PATH)
    if not rows:
        log("ERROR: No rows found in final CSV.")
        sys.exit(1)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    log(f"[extract] Using device: {device}")

    model = CLIPModel.from_pretrained(MODEL_NAME).to(device)
    processor = CLIPProcessor.from_pretrained(MODEL_NAME)

    metadata_rows: List[dict] = []
    text_features: List[np.ndarray] = []
    image_features: List[np.ndarray] = []
    skipped = 0

    max_count = len(rows) if MAX_ROWS is None else min(MAX_ROWS, len(rows))
    for idx, row in enumerate(rows[:max_count], start=1):
        try:
            text = get_text_source(row)
            if not text:
                raise ValueError("Missing text for encoding")

            image_path = str(row.get("local_image_path", "")).strip()
            if not image_path:
                raise ValueError("Missing local_image_path")

            image = load_image(image_path)
            inputs = processor(text=[text], images=image, return_tensors="pt", padding=True)
            inputs = {k: v.to(device) for k, v in inputs.items()}

            with torch.no_grad():
                text_output = model.get_text_features(**{k: v for k, v in inputs.items() if k.startswith("input_ids") or k == "attention_mask"})
                image_output = model.get_image_features(**{k: v for k, v in inputs.items() if k == "pixel_values"})

            text_vec = normalize_vector(text_output.cpu().numpy()[0])
            image_vec = normalize_vector(image_output.cpu().numpy()[0])

            text_features.append(text_vec)
            image_features.append(image_vec)

            metadata_rows.append({
                "id": row.get("id", ""),
                "tweet_text": row.get("tweet_text", ""),
                "tweet_text_clean": row.get("tweet_text_clean", ""),
                "local_image_path": image_path,
                "label": row.get("label", ""),
                "encode_status": "ok",
            })
        except Exception as exc:
            skipped += 1
            metadata_rows.append({
                "id": row.get("id", ""),
                "tweet_text": row.get("tweet_text", ""),
                "tweet_text_clean": row.get("tweet_text_clean", ""),
                "local_image_path": str(row.get("local_image_path", "")).strip(),
                "label": row.get("label", ""),
                "encode_status": f"error: {exc}",
            })
            log(f"[extract] Skipped row {idx} due to error: {exc}")
            continue

        if idx % 20 == 0:
            log(f"[extract] Processed {idx}/{max_count} rows...")

    if text_features:
        np.save(TEXT_FEATURES_NPY, np.vstack(text_features))
        log(f"[extract] Saved text vectors to: {TEXT_FEATURES_NPY}")
    else:
        log("[extract] No text vectors generated.")

    if image_features:
        np.save(IMAGE_FEATURES_NPY, np.vstack(image_features))
        log(f"[extract] Saved image vectors to: {IMAGE_FEATURES_NPY}")
    else:
        log("[extract] No image vectors generated.")

    save_csv(METADATA_OUTPUT_CSV, metadata_rows, ["id", "tweet_text", "tweet_text_clean", "local_image_path", "label", "encode_status"])
    log(f"[extract] Saved metadata to: {METADATA_OUTPUT_CSV}")
    log(f"[extract] Total rows processed: {max_count}")
    log(f"[extract] Rows skipped: {skipped}")


if __name__ == "__main__":
    main()
