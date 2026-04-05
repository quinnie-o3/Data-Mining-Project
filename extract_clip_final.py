import csv
import os
import sys
import numpy as np
import torch
from PIL import Image

try:
    from transformers import CLIPModel, CLIPProcessor
except ImportError:
    print("Installing transformers...")
    os.system("pip install transformers")
    from transformers import CLIPModel, CLIPProcessor

# CONFIG
INPUT_CSV_PATH = "twitter_clip_ready.csv"
IMAGE_FOLDER = "twitter_images"
METADATA_OUTPUT = "clip_metadata_final.csv"
TEXT_FEATURES_NPY = "clip_text_features_final.npy"
IMAGE_FEATURES_NPY = "clip_image_features_final.npy"
MODEL_NAME = "openai/clip-vit-base-patch32"
MAX_ROWS = None  # None = all, or set to number for testing

# END CONFIG


def log(msg):
    print(msg, flush=True)


def normalize_vector(vec):
    norm = np.linalg.norm(vec)
    if norm == 0 or np.isnan(norm):
        return vec.astype(np.float32)
    return (vec / norm).astype(np.float32)


def load_csv(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Not found: {path}")
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def main():
    log("[extract] Loading CSV...")
    rows = load_csv(INPUT_CSV_PATH)
    
    if MAX_ROWS:
        rows = rows[:MAX_ROWS]
        log(f"[extract] Limited to {MAX_ROWS} rows for testing")

    if not rows:
        log("ERROR: No rows found in CSV")
        sys.exit(1)

    log(f"[extract] Total rows to process: {len(rows)}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    log(f"[extract] Device: {device}")
    log(f"[extract] Loading model: {MODEL_NAME}")
    
    model = CLIPModel.from_pretrained(MODEL_NAME).to(device)
    processor = CLIPProcessor.from_pretrained(MODEL_NAME)

    text_features_list = []
    image_features_list = []
    metadata_list = []
    
    success_count = 0
    skip_count = 0
    error_count = 0

    for idx, row in enumerate(rows, 1):
        try:
            # Get text
            text = str(row.get("tweetText", "")).strip()
            if not text:
                raise ValueError("Empty text")

            # Find corresponding image
            row_id = str(row.get("id", "")) or str(idx)
            
            # Look for image file
            image_path = None
            if IMAGE_FOLDER and os.path.exists(IMAGE_FOLDER):
                # Try to find by ID-based naming
                for fname in os.listdir(IMAGE_FOLDER):
                    if fname.startswith(row_id.replace(" ", "_")):
                        image_path = os.path.join(IMAGE_FOLDER, fname)
                        break
                
                # If not found, try to find any image (fallback)
                if not image_path:
                    images = [f for f in os.listdir(IMAGE_FOLDER) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp'))]
                    if images:
                        # Try to match by index as fallback
                        idx_in_folder = idx - 1
                        if idx_in_folder < len(images):
                            image_path = os.path.join(IMAGE_FOLDER, images[idx_in_folder])
            
            if not image_path or not os.path.exists(image_path):
                skip_count += 1
                continue

            # Load image
            try:
                image = Image.open(image_path).convert("RGB")
            except Exception as e:
                raise ValueError(f"Cannot load image: {e}")

            # Process with CLIP
            inputs = processor(
                text=[text],
                images=image,
                return_tensors="pt",
                padding=True
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}

            with torch.no_grad():
                text_features = model.get_text_features(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs.get("attention_mask")
                )
                image_features = model.get_image_features(pixel_values=inputs["pixel_values"])

            text_vec = normalize_vector(text_features.cpu().numpy()[0])
            image_vec = normalize_vector(image_features.cpu().numpy()[0])

            text_features_list.append(text_vec)
            image_features_list.append(image_vec)

            metadata_list.append({
                "id": row.get("id", ""),
                "tweetText": text[:100],
                "image_file": os.path.basename(image_path),
                "label": row.get("label", ""),
                "status": "ok"
            })

            success_count += 1

            if idx % 20 == 0:
                log(f"[extract] Processed {idx}/{len(rows)} | Success: {success_count} | Skip: {skip_count}")

        except Exception as e:
            error_count += 1
            metadata_list.append({
                "id": row.get("id", ""),
                "tweetText": str(row.get("tweetText", ""))[:50],
                "image_file": "",
                "label": row.get("label", ""),
                "status": f"error: {str(e)[:50]}"
            })
            if error_count <= 5:
                log(f"[extract] Error at row {idx}: {e}")

    # Save vectors
    if text_features_list:
        text_array = np.vstack(text_features_list)
        np.save(TEXT_FEATURES_NPY, text_array)
        log(f"[extract] Saved {len(text_features_list)} text features to {TEXT_FEATURES_NPY}")
        log(f"[extract] Shape: {text_array.shape}")
    else:
        log("[extract] No text features generated")

    if image_features_list:
        image_array = np.vstack(image_features_list)
        np.save(IMAGE_FEATURES_NPY, image_array)
        log(f"[extract] Saved {len(image_features_list)} image features to {IMAGE_FEATURES_NPY}")
        log(f"[extract] Shape: {image_array.shape}")
    else:
        log("[extract] No image features generated")

    # Save metadata
    with open(METADATA_OUTPUT, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, ["id", "tweetText", "image_file", "label", "status"])
        w.writeheader()
        w.writerows(metadata_list)
    log(f"[extract] Saved metadata to {METADATA_OUTPUT}")

    log(f"[extract] FINAL SUMMARY:")
    log(f"[extract] Total rows: {len(rows)}")
    log(f"[extract] Successful: {success_count}")
    log(f"[extract] Skipped: {skip_count}")
    log(f"[extract] Errors: {error_count}")


if __name__ == "__main__":
    main()
