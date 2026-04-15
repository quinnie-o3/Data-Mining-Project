from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor


INPUT_CSV = Path(r"D:\ARGGGG\Semester 6\DATA MINING\Pheme_CLIP\pheme_clip_final.csv")
OUTPUT_METADATA = Path(r"D:\ARGGGG\Semester 6\DATA MINING\Pheme_CLIP\clip_metadata.csv")
OUTPUT_TEXT_FEATURES = Path(r"D:\ARGGGG\Semester 6\DATA MINING\Pheme_CLIP\clip_text_features.npy")
OUTPUT_IMAGE_FEATURES = Path(r"D:\ARGGGG\Semester 6\DATA MINING\Pheme_CLIP\clip_image_features.npy")

MODEL_NAME = "openai/clip-vit-base-patch32"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Set to a small integer such as 5 for a quick test. Use None to process all rows.
MAX_ROWS = None
PROGRESS_EVERY = 20


def log(message: str) -> None:
    print(message, flush=True)


def load_clip_model() -> tuple[CLIPModel, CLIPProcessor]:
    log(f"Loading CLIP model: {MODEL_NAME}")
    log(f"Using device: {DEVICE}")

    try:
        model = CLIPModel.from_pretrained(MODEL_NAME).to(DEVICE)
        processor = CLIPProcessor.from_pretrained(MODEL_NAME)
    except Exception as error:
        log(f"Online load failed, retrying from local cache: {error}")
        model = CLIPModel.from_pretrained(
            MODEL_NAME,
            local_files_only=True,
        ).to(DEVICE)
        processor = CLIPProcessor.from_pretrained(
            MODEL_NAME,
            local_files_only=True,
        )

    model.eval()
    return model, processor


def read_input_csv(csv_path: Path) -> pd.DataFrame:
    log(f"Reading CSV: {csv_path}")

    if not csv_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    required_columns = ["id", "tweet_text", "local_image_path", "label"]
    missing_columns = [column for column in required_columns if column not in df.columns]
    if missing_columns:
        raise ValueError(
            "Missing required column(s): " + ", ".join(missing_columns)
        )

    if MAX_ROWS is not None:
        df = df.head(MAX_ROWS).copy()
        log(f"MAX_ROWS is set to {MAX_ROWS}. Running in quick-test mode.")

    log(f"Total input rows: {len(df)}")
    return df


def choose_text_column(df: pd.DataFrame) -> str:
    text_column = "tweet_text_clean" if "tweet_text_clean" in df.columns else "tweet_text"
    log(f"Text column in use: {text_column}")
    return text_column


def normalize_embedding(embedding: torch.Tensor) -> np.ndarray:
    normalized = embedding / embedding.norm(p=2, dim=-1, keepdim=True).clamp(min=1e-12)
    return normalized.squeeze(0).detach().cpu().numpy().astype(np.float32)


def encode_row(
    model: CLIPModel,
    processor: CLIPProcessor,
    text: str,
    image_path: Path,
) -> tuple[np.ndarray, np.ndarray]:
    with Image.open(image_path) as image:
        image_rgb = image.convert("RGB")

    text_inputs = processor(
        text=[text],
        return_tensors="pt",
        padding=True,
        truncation=True,
    )
    image_inputs = processor(images=image_rgb, return_tensors="pt")

    text_inputs = {key: value.to(DEVICE) for key, value in text_inputs.items()}
    image_inputs = {key: value.to(DEVICE) for key, value in image_inputs.items()}

    with torch.no_grad():
        text_outputs = model.text_model(**text_inputs)
        image_outputs = model.vision_model(**image_inputs)

        text_features = model.text_projection(text_outputs.pooler_output)
        image_features = model.visual_projection(image_outputs.pooler_output)

    return normalize_embedding(text_features), normalize_embedding(image_features)


def main() -> None:
    model, processor = load_clip_model()
    df = read_input_csv(INPUT_CSV)
    text_column = choose_text_column(df)

    valid_rows: list[dict] = []
    text_feature_list: list[np.ndarray] = []
    image_feature_list: list[np.ndarray] = []

    for row_number, (_, row) in enumerate(df.iterrows(), start=1):
        if row_number % PROGRESS_EVERY == 0 or row_number == 1:
            log(f"Progress: {row_number}/{len(df)} rows checked")

        row_id = row.get("id", "")
        text_value = row.get(text_column, "")
        image_value = row.get("local_image_path", "")
        label_value = row.get("label", "")

        text = "" if pd.isna(text_value) else str(text_value).strip()
        image_path_str = "" if pd.isna(image_value) else str(image_value).strip()
        image_path = Path(image_path_str)

        if not text:
            log(f"Skipping row {row_number} (id={row_id}): empty text")
            continue

        if not image_path_str:
            log(f"Skipping row {row_number} (id={row_id}): empty local_image_path")
            continue

        if not image_path.exists():
            log(f"Skipping row {row_number} (id={row_id}): image not found -> {image_path}")
            continue

        try:
            text_features, image_features = encode_row(
                model=model,
                processor=processor,
                text=text,
                image_path=image_path,
            )
        except Exception as error:
            log(f"Skipping row {row_number} (id={row_id}): encode error -> {error}")
            continue

        valid_rows.append(
            {
                "id": row_id,
                "tweet_text": row.get("tweet_text", ""),
                "local_image_path": str(image_path),
                "label": label_value,
            }
        )
        text_feature_list.append(text_features)
        image_feature_list.append(image_features)

    valid_count = len(valid_rows)
    log(f"Final valid rows: {valid_count}")

    if valid_count == 0:
        raise RuntimeError("No valid rows were encoded. Check CSV content and image paths.")

    text_feature_array = np.vstack(text_feature_list).astype(np.float32)
    image_feature_array = np.vstack(image_feature_list).astype(np.float32)
    metadata_df = pd.DataFrame(valid_rows)

    metadata_df.to_csv(OUTPUT_METADATA, index=False, encoding="utf-8-sig")
    np.save(OUTPUT_TEXT_FEATURES, text_feature_array)
    np.save(OUTPUT_IMAGE_FEATURES, image_feature_array)

    log(f"Text feature shape: {text_feature_array.shape}")
    log(f"Image feature shape: {image_feature_array.shape}")
    log("Output files:")
    log(f"  Metadata CSV: {OUTPUT_METADATA}")
    log(f"  Text features NPY: {OUTPUT_TEXT_FEATURES}")
    log(f"  Image features NPY: {OUTPUT_IMAGE_FEATURES}")


if __name__ == "__main__":
    main()
