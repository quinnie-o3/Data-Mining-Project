import json
import logging
import os
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from PIL import Image, ImageFile
from torchvision import transforms
from torchvision.models.detection import (
    FasterRCNN_ResNet50_FPN_Weights,
    fasterrcnn_resnet50_fpn,
)
from transformers import CLIPModel, CLIPProcessor


ImageFile.LOAD_TRUNCATED_IMAGES = True


# Configuration
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
OUTPUT_DIR = SCRIPT_DIR
INPUT_CSV = PROJECT_ROOT / "pheme_clip_final.csv"
IMAGE_ROOT = PROJECT_ROOT / "pheme_images"
OUTPUT_JSONL = OUTPUT_DIR / "clip_test_500_final.jsonl"
OUTPUT_PT = OUTPUT_DIR / "clip_test_500_final.pt"
OUTPUT_LOG = OUTPUT_DIR / "clip_test_500_final.log"
CACHE_DIR = OUTPUT_DIR / "model_cache"
TORCH_CACHE_DIR = CACHE_DIR / "torch"
HF_CACHE_DIR = CACHE_DIR / "huggingface"
MAX_ROWS = 500
SCORE_THRESHOLD = 0.5
TOPK_OBJECTS = 10
CLIP_MODEL_NAME = "openai/clip-vit-base-patch16"
PROGRESS_EVERY = 10


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TO_TENSOR = transforms.ToTensor()


def setup_logger() -> logging.Logger:
    logger = logging.getLogger("clip_object_pipeline")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    file_handler = logging.FileHandler(OUTPUT_LOG, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    return logger


LOGGER = setup_logger()


def log(message: str) -> None:
    LOGGER.info(message)


def infer_columns(df: pd.DataFrame) -> tuple[str, str, str, str]:
    id_candidates = ["id", "sample_id", "tweet_id", "post_id"]
    image_candidates = [
        "local_image_path",
        "image_path",
        "image_file",
        "image_filename",
        "filename",
        "image",
    ]
    text_candidates = ["tweet_text_clean", "matched_text", "tweet_text", "text", "caption"]
    label_candidates = ["label", "class", "target", "rumor_label", "veracity"]

    id_column = next((column for column in id_candidates if column in df.columns), None)
    image_column = next((column for column in image_candidates if column in df.columns), None)
    text_column = next((column for column in text_candidates if column in df.columns), None)
    label_column = next((column for column in label_candidates if column in df.columns), None)

    if not id_column or not image_column or not text_column or not label_column:
        raise ValueError(
            f"Could not infer required columns. Available columns: {list(df.columns)}. "
            f"Required: id, image, text, label"
        )

    return id_column, image_column, text_column, label_column


def resolve_image_path(image_value: Any) -> Path | None:
    if pd.isna(image_value):
        return None

    image_str = str(image_value).strip()
    if not image_str:
        return None

    candidate = Path(image_str)
    if candidate.is_absolute():
        return candidate

    return IMAGE_ROOT / candidate


def load_input_rows() -> tuple[pd.DataFrame, str, str, str, str]:
    if not INPUT_CSV.exists():
        raise FileNotFoundError(f"Input CSV not found: {INPUT_CSV}")

    df = pd.read_csv(INPUT_CSV)
    id_column, image_column, text_column, label_column = infer_columns(df)

    log(f"Loaded CSV: {INPUT_CSV}")
    log(f"Inferred id column: {id_column}")
    log(f"Inferred image column: {image_column}")
    log(f"Inferred text column: {text_column}")
    log(f"Inferred label column: {label_column}")
    log(f"Image root: {IMAGE_ROOT}")

    candidate_rows: list[int] = []
    for index, row in df.iterrows():
        row_id = row.get(id_column)
        image_path = resolve_image_path(row.get(image_column))
        text_value = row.get(text_column)
        label_value = row.get(label_column)

        if pd.isna(row_id) or image_path is None or pd.isna(text_value) or pd.isna(label_value) or not str(text_value).strip():
            continue

        candidate_rows.append(index)
        if len(candidate_rows) >= MAX_ROWS:
            break

    candidate_df = df.loc[candidate_rows].copy()
    log(f"Selected {len(candidate_df)} candidate rows for the pilot run")
    return candidate_df, id_column, image_column, text_column, label_column


def load_models() -> tuple[Any, CLIPModel, CLIPProcessor]:
    log(f"Using device: {DEVICE}")
    log(f"Model cache directory: {CACHE_DIR}")
    log("Loading Faster R-CNN model")
    os.environ["TORCH_HOME"] = str(TORCH_CACHE_DIR)
    try:
        detector = fasterrcnn_resnet50_fpn(
            weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT
        ).to(DEVICE)
    except Exception as error:
        raise RuntimeError(
            "Failed to load Faster R-CNN weights. "
            "If this is the first run, make sure the machine can download "
            "torchvision weights or pre-populate the local cache directory: "
            f"{TORCH_CACHE_DIR}"
        ) from error
    detector.eval()

    log(f"Loading CLIP model: {CLIP_MODEL_NAME}")
    try:
        clip_model = CLIPModel.from_pretrained(
            CLIP_MODEL_NAME,
            cache_dir=str(HF_CACHE_DIR),
            local_files_only=True,
        ).to(DEVICE)
        clip_processor = CLIPProcessor.from_pretrained(
            CLIP_MODEL_NAME,
            cache_dir=str(HF_CACHE_DIR),
            local_files_only=True,
        )
    except Exception as error:
        log(f"Local CLIP load failed: {error}")
        raise RuntimeError(
            "Failed to load CLIP model from local cache. "
            "Populate the cache directory first: "
            f"{HF_CACHE_DIR}"
        ) from error

    clip_model.eval()
    return detector, clip_model, clip_processor


def normalize_embedding(embedding) -> list[float]:
    if hasattr(embedding, 'pooler_output'):
        embedding = embedding.pooler_output
    elif hasattr(embedding, 'last_hidden_state'):
        embedding = embedding.last_hidden_state.mean(dim=1)  # For sequence outputs, mean pool
    normalized = embedding / embedding.norm(p=2, dim=-1, keepdim=True).clamp(min=1e-12)
    return normalized.squeeze(0).detach().cpu().to(torch.float32).tolist()


def extract_clip_embedding(
    clip_model: CLIPModel,
    clip_processor: CLIPProcessor,
    image: Image.Image,
) -> list[float]:
    clip_inputs = clip_processor(images=image, return_tensors="pt")
    clip_inputs = {key: value.to(DEVICE) for key, value in clip_inputs.items()}

    with torch.no_grad():
        image_features = clip_model.get_image_features(**clip_inputs)

    return normalize_embedding(image_features)


def detect_objects(
    detector: Any,
    image: Image.Image,
) -> tuple[list[list[float]], list[float], list[int]]:
    image_tensor = TO_TENSOR(image).to(DEVICE)

    with torch.no_grad():
        outputs = detector([image_tensor])[0]

    boxes = outputs["boxes"].detach().cpu()
    scores = outputs["scores"].detach().cpu()
    labels = outputs["labels"].detach().cpu()

    keep_mask = scores >= SCORE_THRESHOLD
    boxes = boxes[keep_mask][:TOPK_OBJECTS]
    scores = scores[keep_mask][:TOPK_OBJECTS]
    labels = labels[keep_mask][:TOPK_OBJECTS]

    return (
        boxes.to(torch.float32).tolist(),
        scores.to(torch.float32).tolist(),
        labels.to(torch.int64).tolist(),
    )


def crop_regions(image: Image.Image, boxes: list[list[float]]) -> list[Image.Image]:
    width, height = image.size
    crops: list[Image.Image] = []

    for box in boxes:
        x1, y1, x2, y2 = box
        left = max(0, min(int(x1), width - 1))
        top = max(0, min(int(y1), height - 1))
        right = max(left + 1, min(int(x2), width))
        bottom = max(top + 1, min(int(y2), height))
        crops.append(image.crop((left, top, right, bottom)))

    return crops


def empty_record(sample_id: str, image_path: str, text: str, label: str, status: str) -> dict[str, Any]:
    return {
        "sample_id": sample_id,
        "image_path": image_path,
        "text": text,
        "label": label,
        "boxes": [],
        "scores": [],
        "labels": [],
        "full_image_embedding": [],
        "object_embeddings": [],
        "num_objects": 0,
        "status": status,
    }


def process_row(
    row: pd.Series,
    id_column: str,
    image_column: str,
    text_column: str,
    label_column: str,
    detector: Any,
    clip_model: CLIPModel,
    clip_processor: CLIPProcessor,
) -> dict[str, Any]:
    sample_id = str(row[id_column])
    text = str(row[text_column]).strip()
    label = str(row[label_column]).strip()
    
    # Assert input has label
    assert label, f"Label is empty for sample_id {sample_id}"
    
    image_path = resolve_image_path(row[image_column])
    image_path_str = "" if image_path is None else str(image_path)

    if image_path is None:
        log(f"[{sample_id}] missing image path in CSV")
        return empty_record(sample_id, image_path_str, text, label, "missing_image_path")

    if not image_path.exists():
        log(f"[{sample_id}] missing image file: {image_path}")
        return empty_record(sample_id, str(image_path), text, label, "missing_image")

    try:
        with Image.open(image_path) as pil_image:
            image_rgb = pil_image.convert("RGB")
    except Exception as error:
        log(f"[{sample_id}] unreadable image: {image_path} | {error}")
        return empty_record(sample_id, str(image_path), text, label, "unreadable_image")

    try:
        full_image_embedding = extract_clip_embedding(clip_model, clip_processor, image_rgb)
        boxes, scores, labels = detect_objects(detector, image_rgb)
        object_embeddings = [
            extract_clip_embedding(clip_model, clip_processor, crop)
            for crop in crop_regions(image_rgb, boxes)
        ]
    except Exception as error:
        log(f"[{sample_id}] inference error: {image_path} | {error}")
        record = empty_record(sample_id, str(image_path), text, label, "inference_error")
        record["full_image_embedding"] = full_image_embedding if "full_image_embedding" in locals() else []
        return record

    return {
        "sample_id": sample_id,
        "image_path": str(image_path),
        "text": text,
        "label": label,
        "boxes": boxes,
        "scores": scores,
        "labels": labels,
        "full_image_embedding": full_image_embedding,
        "object_embeddings": object_embeddings,
        "num_objects": len(object_embeddings),
        "status": "ok",
    }


def save_outputs(records: list[dict[str, Any]]) -> None:
    with OUTPUT_JSONL.open("w", encoding="utf-8") as jsonl_file:
        for record in records:
            jsonl_file.write(json.dumps(record, ensure_ascii=False) + "\n")

    torch.save(records, OUTPUT_PT)

    log(f"Saved JSONL output: {OUTPUT_JSONL}")
    log(f"Saved PT output: {OUTPUT_PT}")
    log(f"Saved log file: {OUTPUT_LOG}")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    TORCH_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    HF_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    log("Starting object detection + CLIP extraction pipeline")
    candidate_df, id_column, image_column, text_column, label_column = load_input_rows()
    detector, clip_model, clip_processor = load_models()

    records: list[dict[str, Any]] = []
    total_rows = len(candidate_df)

    for idx, (_, row) in enumerate(candidate_df.iterrows(), start=1):
        if idx == 1 or idx % PROGRESS_EVERY == 0 or idx == total_rows:
            log(f"Processing row {idx}/{total_rows}")

        record = process_row(
            row=row,
            id_column=id_column,
            image_column=image_column,
            text_column=text_column,
            label_column=label_column,
            detector=detector,
            clip_model=clip_model,
            clip_processor=clip_processor,
        )
        records.append(record)

    save_outputs(records)

    ok_count = sum(record["status"] == "ok" for record in records)
    zero_object_count = sum(record["status"] == "ok" and record["num_objects"] == 0 for record in records)

    log(f"Completed {len(records)} records")
    log(f"Successful records: {ok_count}")
    log(f"Successful records with zero detected objects: {zero_object_count}")

    # Inspect first record
    if records:
        log(f"First record sample: {records[0]}")
        assert "label" in records[0], "Label missing in output record"
        log(f"Total records: {len(records)}")
        log(f"Records with label: {sum('label' in r for r in records)}")


if __name__ == "__main__":
    main()
