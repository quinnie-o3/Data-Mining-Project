import csv
import os
import numpy as np
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

# CONFIG
INPUT_CSV_PATH = "twitter_clip_ready_downloaded_test.csv"
OUTPUT_CSV = "clip_test_output.csv"
TEXT_NPY = "clip_text_test.npy"
IMAGE_NPY = "clip_image_test.npy"
MODEL_NAME = "openai/clip-vit-base-patch32"

# END CONFIG


def log(msg):
    print(msg, flush=True)


def load_csv(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Not found: {path}")
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def normalize_vector(vec):
    norm = np.linalg.norm(vec)
    if norm == 0 or np.isnan(norm):
        return vec.astype(np.float32)
    return (vec / norm).astype(np.float32)


def main():
    log("[clip_test] Loading CSV...")
    rows = load_csv(INPUT_CSV_PATH)
    if not rows:
        log("No data found")
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    log(f"[clip_test] Device: {device}")

    log(f"[clip_test] Loading CLIP model...")
    model = CLIPModel.from_pretrained(MODEL_NAME).to(device)
    processor = CLIPProcessor.from_pretrained(MODEL_NAME)

    text_vecs = []
    image_vecs = []
    metadata = []

    for idx, row in enumerate(rows, 1):
        try:
            text = str(row.get("tweetText", "")).strip()
            img_path = str(row.get("local_image_path", "")).strip()

            if not text:
                raise ValueError("No text")
            if not img_path or not os.path.exists(img_path):
                raise ValueError(f"No image: {img_path}")

            image = Image.open(img_path).convert("RGB")
            inputs = processor(text=[text], images=image, return_tensors="pt", padding=True)
            inputs = {k: v.to(device) for k, v in inputs.items()}

            with torch.no_grad():
                text_feat = model.get_text_features(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"])
                image_feat = model.get_image_features(pixel_values=inputs["pixel_values"])

            text_vecs.append(normalize_vector(text_feat.cpu().numpy()[0]))
            image_vecs.append(normalize_vector(image_feat.cpu().numpy()[0]))

            metadata.append({
                "id": row.get("id", ""),
                "text": text[:50] + "..." if len(text) > 50 else text,
                "image": os.path.basename(img_path),
                "status": "ok"
            })

            log(f"[clip_test] Processed {idx}/{len(rows)}...")

        except Exception as e:
            metadata.append({
                "id": row.get("id", ""),
                "text": str(row.get("tweetText", ""))[:20],
                "image": str(row.get("local_image_path", ""))[-20:],
                "status": str(e)[:50]
            })
            log(f"[clip_test] Error at {idx}: {e}")

    if text_vecs:
        np.save(TEXT_NPY, np.vstack(text_vecs))
        log(f"[clip_test] Saved {len(text_vecs)} text vectors to {TEXT_NPY}")
    if image_vecs:
        np.save(IMAGE_NPY, np.vstack(image_vecs))
        log(f"[clip_test] Saved {len(image_vecs)} image vectors to {IMAGE_NPY}")

    with open(OUTPUT_CSV, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, ["id", "text", "image", "status"])
        w.writeheader()
        w.writerows(metadata)
    log(f"[clip_test] Saved metadata to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
