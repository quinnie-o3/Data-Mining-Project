import csv
import os
import sys
import numpy as np
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

# CONFIG
INPUT_CSV_PATH = "twitter_clip_ready.csv"
IMAGE_FOLDER = "twitter_images"
METADATA_OUTPUT = "clip_final_output.csv"
TEXT_FEATURES_NPY = "clip_text_vectors.npy"
IMAGE_FEATURES_NPY = "clip_image_vectors.npy"
MODEL_NAME = "openai/clip-vit-base-patch32"
MAX_PROCESS = 300  # Process only 300 samples for speed

# END CONFIG


def log(msg):
    print(msg, flush=True)


def normalize_vector(vec):
    norm = np.linalg.norm(vec)
    if norm == 0 or np.isnan(norm):
        return vec.astype(np.float32)
    return (vec / norm).astype(np.float32)


def load_csv(path):
    with open(path, "r", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def main():
    log("[CLIP] Starting extraction...")
    rows = load_csv(INPUT_CSV_PATH)[:MAX_PROCESS]
    log(f"[CLIP] Processing {len(rows)} rows")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    log(f"[CLIP] Device: {device}")
    
    model = CLIPModel.from_pretrained(MODEL_NAME).to(device)
    processor = CLIPProcessor.from_pretrained(MODEL_NAME)

    text_vecs = []
    image_vecs = []
    metadata = []

    for idx, row in enumerate(rows, 1):
        try:
            text = str(row.get("tweetText", "")).strip()
            if not text:
                continue

            # Find image
            images = [f for f in os.listdir(IMAGE_FOLDER) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp'))]
            if not images or idx - 1 >= len(images):
                continue

            img_path = os.path.join(IMAGE_FOLDER, images[idx - 1])
            image = Image.open(img_path).convert("RGB")

            inputs = processor(text=[text], images=image, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}

            with torch.no_grad():
                text_feat = model.get_text_features(input_ids=inputs["input_ids"], attention_mask=inputs.get("attention_mask"))
                img_feat = model.get_image_features(pixel_values=inputs["pixel_values"])

            text_vecs.append(normalize_vector(text_feat.cpu().numpy()[0]))
            image_vecs.append(normalize_vector(img_feat.cpu().numpy()[0]))

            metadata.append({
                "id": row.get("id", idx),
                "text_sample": text[:50],
                "image": os.path.basename(img_path),
                "label": row.get("label", ""),
                "status": "ok"
            })

            if idx % 50 == 0:
                log(f"[CLIP] Progress: {idx}/{len(rows)}")

        except Exception as e:
            log(f"[CLIP] Skipped row {idx}: {str(e)[:40]}")

    if text_vecs:
        text_array = np.vstack(text_vecs)
        np.save(TEXT_FEATURES_NPY, text_array)
        log(f"[CLIP] Text vectors: {text_array.shape} saved to {TEXT_FEATURES_NPY}")

    if image_vecs:
        image_array = np.vstack(image_vecs)
        np.save(IMAGE_FEATURES_NPY, image_array)
        log(f"[CLIP] Image vectors: {image_array.shape} saved to {IMAGE_FEATURES_NPY}")

    with open(METADATA_OUTPUT, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, ["id", "text_sample", "image", "label", "status"])
        w.writeheader()
        w.writerows(metadata)
    log(f"[CLIP] Metadata saved to {METADATA_OUTPUT}")
    log(f"[CLIP] DONE: {len(metadata)} vectors extracted")


if __name__ == "__main__":
    main()
