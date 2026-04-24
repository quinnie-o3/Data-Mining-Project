from __future__ import annotations

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

from config import (
    HF_CACHE_DIR,
    IMAGE_DIR,
    OBJECTS_JSONL,
    OBJECTS_LOG,
    TORCH_CACHE_DIR,
    downloaded_csv_input_path,
    ensure_data_dirs,
)

ImageFile.LOAD_TRUNCATED_IMAGES = True

MAX_ROWS = None
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

    file_handler = logging.FileHandler(OBJECTS_LOG, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    return logger


ensure_data_dirs()
LOGGER = setup_logger()


def log(message: str) -> None:
    LOGGER.info(message)


def infer_columns(df: pd.DataFrame) -> tuple[str, str, str, str]:
    id_candidates = ["id"]
    image_candidates = ["local_image_path"]
    text_candidates = ["tweetText_en"]
    label_candidates = ["label"]

    id_column = next((column for column in id_candidates if column in df.columns), None)
    image_column = next((column for column in image_candidates if column in df.columns), None)
    text_column = next((column for column in text_candidates if column in df.columns), None)
    label_column = next((column for column in label_candidates if column in df.columns), None)

    if not id_column or not image_column or not text_column or not label_column:
        raise ValueError("Could not infer required columns.")
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
    return IMAGE_DIR / candidate


def get_processed_ids() -> set[str]:
    processed_ids = set()
    if OBJECTS_JSONL.exists():
        log(f"Found existing JSONL at {OBJECTS_JSONL}. Reading processed sample IDs...")
        with open(OBJECTS_JSONL, "r", encoding="utf-8") as handle:
            for line in handle:
                try:
                    record = json.loads(line)
                    if "sample_id" in record:
                        processed_ids.add(str(record["sample_id"]))
                except Exception:
                    pass
    return processed_ids


def load_input_rows(processed_ids: set[str]) -> tuple[pd.DataFrame, str, str, str, str]:
    input_csv = downloaded_csv_input_path()
    if not input_csv.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_csv}")

    log(f"Reading downloaded-image manifest from: {input_csv}")
    df = pd.read_csv(input_csv)
    id_column, image_column, text_column, label_column = infer_columns(df)

    candidate_rows: list[int] = []
    for index, row in df.iterrows():
        row_id = str(row.get(id_column))
        if row_id in processed_ids:
            continue

        image_path = resolve_image_path(row.get(image_column))
        text_value = row.get(text_column)
        label_value = row.get(label_column)

        if pd.isna(row_id) or image_path is None or pd.isna(text_value) or pd.isna(label_value) or not str(text_value).strip():
            continue

        candidate_rows.append(index)
        if MAX_ROWS is not None and len(candidate_rows) >= MAX_ROWS:
            break

    candidate_df = df.loc[candidate_rows].copy()
    log(f"Remaining rows to process: {len(candidate_df)}. Skipped already-processed rows: {len(processed_ids)}")
    return candidate_df, id_column, image_column, text_column, label_column


def load_models() -> tuple[Any, CLIPModel, CLIPProcessor]:
    log("Loading Faster R-CNN model")
    os.environ["TORCH_HOME"] = str(TORCH_CACHE_DIR)
    detector = fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT).to(DEVICE)
    detector.eval()

    log(f"Loading CLIP model: {CLIP_MODEL_NAME}")
    try:
        clip_model = CLIPModel.from_pretrained(CLIP_MODEL_NAME, cache_dir=str(HF_CACHE_DIR)).to(DEVICE)
        clip_processor = CLIPProcessor.from_pretrained(CLIP_MODEL_NAME, cache_dir=str(HF_CACHE_DIR))
    except Exception:
        clip_model = CLIPModel.from_pretrained(CLIP_MODEL_NAME, local_files_only=True).to(DEVICE)
        clip_processor = CLIPProcessor.from_pretrained(CLIP_MODEL_NAME, local_files_only=True)
    clip_model.eval()
    return detector, clip_model, clip_processor


def normalize_embedding(embedding) -> list[float]:
    if hasattr(embedding, "pooler_output"):
        embedding = embedding.pooler_output
    elif hasattr(embedding, "last_hidden_state"):
        embedding = embedding.last_hidden_state.mean(dim=1)
    normalized = embedding / embedding.norm(p=2, dim=-1, keepdim=True).clamp(min=1e-12)
    return normalized.squeeze(0).detach().cpu().to(torch.float32).tolist()


def extract_clip_embedding(clip_model, clip_processor, image: Image.Image) -> list[float]:
    clip_inputs = clip_processor(images=image, return_tensors="pt")
    clip_inputs = {key: value.to(DEVICE) for key, value in clip_inputs.items()}
    with torch.no_grad():
        image_features = clip_model.get_image_features(**clip_inputs)
    return normalize_embedding(image_features)


def detect_objects(detector, image: Image.Image):
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

    return boxes.to(torch.float32).tolist(), scores.to(torch.float32).tolist(), labels.to(torch.int64).tolist()


def crop_regions(image: Image.Image, boxes: list[list[float]]) -> list[Image.Image]:
    width, height = image.size
    crops = []
    for box in boxes:
        x1, y1, x2, y2 = box
        left = max(0, min(int(x1), width - 1))
        top = max(0, min(int(y1), height - 1))
        right = max(left + 1, min(int(x2), width))
        bottom = max(top + 1, min(int(y2), height))
        crops.append(image.crop((left, top, right, bottom)))
    return crops


def empty_record(sample_id: str, image_path: str, text: str, label: int, status: str) -> dict[str, Any]:
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


def process_row(row, id_column, image_column, text_column, label_column, detector, clip_model, clip_processor):
    sample_id = str(row[id_column])
    text = str(row[text_column]).strip()

    raw_label = str(row[label_column]).strip().lower()
    numeric_label = 1 if raw_label == "real" else 0

    image_path = resolve_image_path(row[image_column])
    image_path_str = "" if image_path is None else str(image_path)

    if image_path is None or not image_path.exists():
        return empty_record(sample_id, image_path_str, text, numeric_label, "missing_image")

    try:
        with Image.open(image_path) as pil_image:
            image_rgb = pil_image.convert("RGB")
    except Exception:
        return empty_record(sample_id, str(image_path), text, numeric_label, "unreadable_image")

    try:
        full_image_embedding = extract_clip_embedding(clip_model, clip_processor, image_rgb)
        boxes, scores, labels = detect_objects(detector, image_rgb)
        object_embeddings = [extract_clip_embedding(clip_model, clip_processor, crop) for crop in crop_regions(image_rgb, boxes)]
    except Exception:
        record = empty_record(sample_id, str(image_path), text, numeric_label, "inference_error")
        record["full_image_embedding"] = full_image_embedding if "full_image_embedding" in locals() else []
        return record

    return {
        "sample_id": sample_id,
        "image_path": str(image_path),
        "text": text,
        "label": numeric_label,
        "boxes": boxes,
        "scores": scores,
        "labels": labels,
        "full_image_embedding": full_image_embedding,
        "object_embeddings": object_embeddings,
        "num_objects": len(object_embeddings),
        "status": "ok",
    }


def append_output(record: dict[str, Any]) -> None:
    with OBJECTS_JSONL.open("a", encoding="utf-8") as jsonl_file:
        jsonl_file.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> None:
    ensure_data_dirs()
    log("Starting object-detection + CLIP extraction pipeline")
    log(f"Objects JSONL output: {OBJECTS_JSONL}")
    log(f"Objects log output: {OBJECTS_LOG}")

    processed_ids = get_processed_ids()
    candidate_df, id_column, image_column, text_column, label_column = load_input_rows(processed_ids)

    if candidate_df.empty:
        log("All rows are already processed. Stopping pipeline.")
        return

    detector, clip_model, clip_processor = load_models()
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
        append_output(record)

    log(f"Finished processing {total_rows} rows.")
    log(f"All results saved at: {OBJECTS_JSONL}")


if __name__ == "__main__":
    main()
