from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import mimetypes
import sys
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

try:
    import numpy as np
except ImportError as exc:
    raise SystemExit("Missing required package 'numpy'. Install it with: pip install numpy") from exc

try:
    import pandas as pd
except ImportError as exc:
    raise SystemExit(
        "Missing required package 'pandas'. Install it with: pip install pandas openpyxl"
    ) from exc

try:
    import openpyxl  # noqa: F401
except ImportError as exc:
    raise SystemExit("Missing required package 'openpyxl'. Install it with: pip install openpyxl") from exc

try:
    import torch
except ImportError as exc:
    raise SystemExit("Missing required package 'torch'. Install it with: pip install torch") from exc

try:
    from PIL import Image, UnidentifiedImageError
except ImportError as exc:
    raise SystemExit("Missing required package 'Pillow'. Install it with: pip install pillow") from exc

try:
    from tqdm import tqdm
except ImportError as exc:
    raise SystemExit("Missing required package 'tqdm'. Install it with: pip install tqdm") from exc


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

TEXT_COLUMN_CANDIDATES = [
    "text",
    "tweet_text",
    "tweettext",
    "content",
    "caption",
    "full_text",
    "fulltext",
    "post_text",
    "posttext",
    "description",
    "tweet",
    "tweetText_en",
    "tweetText",
    "text_en",
    "translated_text",
    "translatedtext",
]

IMAGE_COLUMN_CANDIDATES = [
    "image_path",
    "img_path",
    "image",
    "filename",
    "filepath",
    "file_path",
    "image_url",
    "img_url",
    "media_path",
    "media_url",
    "photo",
    "picture",
]

LABEL_COLUMN_CANDIDATES = [
    "label",
    "class",
    "target",
    "category",
    "ground_truth",
    "groundtruth",
    "stance",
]

ID_COLUMN_CANDIDATES = [
    "id",
    "tweet_id",
    "tweetid",
    "post_id",
    "postid",
    "status_id",
    "statusid",
]


def configure_stdout() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")


def setup_logger(log_path: Path) -> logging.Logger:
    logger = logging.getLogger("twitter_clip_500")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    file_handler = logging.FileHandler(log_path, encoding="utf-8", mode="w")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    logger.propagate = False
    return logger


def normalize_column_name(name: str) -> str:
    return "".join(character.lower() for character in str(name) if character.isalnum())


def unique_preserve_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            ordered.append(value)
    return ordered


def detect_candidate_columns(columns: list[str], candidates: list[str]) -> list[str]:
    normalized_to_original = {normalize_column_name(column): column for column in columns}
    detected: list[str] = []
    for candidate in candidates:
        normalized = normalize_column_name(candidate)
        if normalized in normalized_to_original:
            detected.append(normalized_to_original[normalized])
    return unique_preserve_order(detected)


def detect_single_column(columns: list[str], candidates: list[str]) -> str | None:
    detected = detect_candidate_columns(columns, candidates)
    return detected[0] if detected else None


def is_missing_value(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    text = str(value).strip()
    return text == "" or text.lower() in {"nan", "none", "null"}


def first_non_empty_value(row: pd.Series, candidate_columns: list[str]) -> tuple[str | None, str | None]:
    for column in candidate_columns:
        if column not in row:
            continue
        value = row[column]
        if not is_missing_value(value):
            return str(value).strip(), column
    return None, None


def build_record_id(row: pd.Series, id_column: str | None) -> str:
    if id_column and id_column in row and not is_missing_value(row[id_column]):
        return str(row[id_column]).strip()
    return f"row_{int(row['source_row_number'])}"


def auto_discover_input_xlsx(project_root: Path) -> Path:
    candidates = sorted(path for path in project_root.glob("*.xlsx") if path.is_file())
    if not candidates:
        raise FileNotFoundError(
            f"No .xlsx file was found in {project_root}. "
            "Place the source dataset in the project root or pass --input-xlsx."
        )
    if len(candidates) > 1:
        formatted = "\n".join(f"  - {path.name}" for path in candidates)
        raise FileNotFoundError(
            "Multiple .xlsx files were found in the project root. "
            "Pass --input-xlsx explicitly.\n"
            f"{formatted}"
        )
    return candidates[0]


def resolve_input_xlsx(input_xlsx: str | None, project_root: Path) -> Path:
    if not input_xlsx:
        return auto_discover_input_xlsx(project_root)

    raw_path = Path(input_xlsx)
    candidate_paths = [raw_path]
    if not raw_path.is_absolute():
        candidate_paths.extend(
            [
                project_root / raw_path,
                SCRIPT_DIR / raw_path,
                Path.cwd() / raw_path,
            ]
        )

    for candidate in candidate_paths:
        if candidate.exists() and candidate.is_file():
            return candidate.resolve()

    raise FileNotFoundError(f"Input dataset not found: {input_xlsx}")


def choose_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def safe_text(value: Any) -> str:
    return str(value).replace("\r", " ").replace("\n", " ").strip()


def create_sample_dataframe(
    df: pd.DataFrame,
    sample_size: int,
    sample_mode: str,
    id_column: str | None,
    label_column: str,
    text_columns: list[str],
    image_columns: list[str],
    logger: logging.Logger,
) -> pd.DataFrame:
    working_df = df.copy()
    working_df.insert(0, "source_row_number", np.arange(2, len(working_df) + 2))

    if label_column != "label":
        working_df["label"] = working_df[label_column]

    if sample_mode == "valid":
        valid_mask = working_df.apply(
            lambda row: (
                first_non_empty_value(row, text_columns)[0] is not None
                or first_non_empty_value(row, image_columns)[0] is not None
            ),
            axis=1,
        )
        valid_rows = int(valid_mask.sum())
        logger.info(
            "Sample mode: valid. Rows with at least text or image: %s/%s",
            valid_rows,
            len(working_df),
        )
        sample_df = working_df.loc[valid_mask].head(sample_size).copy()
    else:
        logger.info("Sample mode: first. Using the first %s rows from the dataset.", sample_size)
        sample_df = working_df.head(sample_size).copy()

    if sample_df.empty:
        raise ValueError("No rows were selected for the sample. Check the dataset content and sample mode.")

    sample_df["record_id"] = sample_df.apply(lambda row: build_record_id(row, id_column), axis=1)
    logger.info("Selected %s rows for the sample file.", len(sample_df))
    return sample_df


def ensure_label_column(sample_df: pd.DataFrame, logger: logging.Logger) -> None:
    if "label" not in sample_df.columns:
        raise ValueError("The sample dataframe does not contain a 'label' column after preparation.")
    missing_labels = int(sample_df["label"].isna().sum())
    logger.info("Label column check passed. Missing label values in sample: %s", missing_labels)


def resolve_local_image_path(
    raw_value: str,
    dataset_dir: Path,
    project_root: Path,
    file_search_cache: dict[str, Path | None],
) -> Path:
    cleaned = raw_value.strip().strip('"').strip("'")
    raw_path = Path(cleaned)

    candidates: list[Path] = []
    if raw_path.is_absolute():
        candidates.append(raw_path)
    else:
        candidates.extend(
            [
                dataset_dir / raw_path,
                project_root / raw_path,
                SCRIPT_DIR / raw_path,
                Path.cwd() / raw_path,
            ]
        )

    for candidate in candidates:
        if candidate.exists() and candidate.is_file():
            return candidate.resolve()

    basename = raw_path.name
    if basename:
        if basename not in file_search_cache:
            matches = list(project_root.rglob(basename))
            file_search_cache[basename] = matches[0].resolve() if matches else None
        cached_match = file_search_cache[basename]
        if cached_match and cached_match.exists():
            return cached_match

    raise FileNotFoundError(f"Image file not found for value: {raw_value}")


def download_image(url: str, image_cache_dir: Path) -> tuple[Path, str]:
    parsed = urlparse(url)
    suffix = Path(parsed.path).suffix.lower()
    if not suffix:
        guessed = mimetypes.guess_extension(mimetypes.guess_type(url)[0] or "")
        suffix = guessed or ".jpg"

    filename = f"{hashlib.md5(url.encode('utf-8')).hexdigest()}{suffix}"
    target_path = image_cache_dir / filename
    if target_path.exists():
        return target_path.resolve(), "cached_url"

    request = Request(url, headers={"User-Agent": "Mozilla/5.0"})
    temporary_path = target_path.with_suffix(target_path.suffix + ".part")
    with urlopen(request, timeout=30) as response:
        content = response.read()

    temporary_path.write_bytes(content)
    temporary_path.replace(target_path)
    return target_path.resolve(), "downloaded_url"


def resolve_image_to_local_path(
    image_value: str,
    dataset_dir: Path,
    project_root: Path,
    image_cache_dir: Path,
    file_search_cache: dict[str, Path | None],
) -> tuple[Path, str]:
    if image_value.lower().startswith(("http://", "https://")):
        return download_image(image_value, image_cache_dir)
    return resolve_local_image_path(image_value, dataset_dir, project_root, file_search_cache), "local_path"


def load_clip_components(model_name: str, pretrained: str, device: str):
    try:
        import open_clip
    except ImportError as exc:
        raise ImportError(
            "open_clip_torch is not installed. Install it with:\n"
            "pip install open_clip_torch"
        ) from exc

    model, _, preprocess = open_clip.create_model_and_transforms(
        model_name=model_name,
        pretrained=pretrained,
        device=device,
    )
    tokenizer = open_clip.get_tokenizer(model_name)
    model.eval()
    return model, preprocess, tokenizer


def get_embedding_dim(model: Any, tokenizer: Any, device: str) -> int:
    with torch.inference_mode():
        try:
            tokenized = tokenizer(["test"], context_length=getattr(model, "context_length", 77))
        except TypeError:
            tokenized = tokenizer(["test"])
        tokenized = tokenized.to(device)
        features = model.encode_text(tokenized)
    return int(features.shape[-1])


def tokenize_text(text: str, tokenizer: Any, context_length: int) -> Any:
    cleaned = safe_text(text)
    try:
        return tokenizer([cleaned], context_length=context_length)
    except TypeError:
        return tokenizer([cleaned])
    except RuntimeError:
        shortened = cleaned[:1000]
        try:
            return tokenizer([shortened], context_length=context_length)
        except TypeError:
            return tokenizer([shortened])


def normalize_feature(feature_tensor: torch.Tensor) -> torch.Tensor:
    return feature_tensor / feature_tensor.norm(dim=-1, keepdim=True).clamp(min=1e-12)


def encode_text_feature(text: str, model: Any, tokenizer: Any, device: str) -> np.ndarray:
    tokenized = tokenize_text(text, tokenizer, getattr(model, "context_length", 77)).to(device)
    with torch.inference_mode():
        feature_tensor = normalize_feature(model.encode_text(tokenized))
    return feature_tensor[0].detach().cpu().numpy().astype(np.float32)


def encode_image_feature(image_path: Path, model: Any, preprocess: Any, device: str) -> np.ndarray:
    with Image.open(image_path) as image:
        rgb_image = image.convert("RGB")
        image_tensor = preprocess(rgb_image).unsqueeze(0).to(device)
    with torch.inference_mode():
        feature_tensor = normalize_feature(model.encode_image(image_tensor))
    return feature_tensor[0].detach().cpu().numpy().astype(np.float32)


def load_object_detector(enable_object_detection: bool, object_model_name: str, logger: logging.Logger) -> tuple[Any | None, str]:
    if not enable_object_detection:
        return None, "not_requested"

    try:
        from ultralytics import YOLO
    except ImportError as exc:
        logger.warning("Object detection requested, but ultralytics is not installed: %s", exc)
        return None, "unavailable"

    try:
        detector = YOLO(object_model_name)
    except Exception as exc:
        logger.warning("Failed to load object detector model '%s': %s", object_model_name, exc)
        return None, "load_failed"

    logger.info("Object detector enabled with model: %s", object_model_name)
    return detector, "ready"


def run_object_detection(detector: Any, image_path: Path) -> list[str]:
    results = detector.predict(source=str(image_path), verbose=False)
    if not results:
        return []

    names = results[0].names
    class_ids = results[0].boxes.cls.tolist() if results[0].boxes is not None else []
    labels = [str(names[int(class_id)]) for class_id in class_ids]
    return unique_preserve_order(labels)


def build_failure_reasons(
    text_status: str,
    image_status: str,
    detector_status: str,
    text_columns_exist: bool,
    image_columns_exist: bool,
) -> list[str]:
    reasons: list[str] = []
    if text_columns_exist and text_status != "success":
        reasons.append(f"text_{text_status}")
    if image_columns_exist and image_status != "success":
        reasons.append(f"image_{image_status}")
    if detector_status not in {"not_requested", "ready", "success", "no_objects"}:
        reasons.append(f"object_detection_{detector_status}")
    return reasons


def process_sample(
    sample_df: pd.DataFrame,
    dataset_path: Path,
    output_dir: Path,
    id_column: str | None,
    text_columns: list[str],
    image_columns: list[str],
    label_column: str,
    model: Any,
    preprocess: Any,
    tokenizer: Any,
    device: str,
    enable_object_detection: bool,
    object_model_name: str,
    logger: logging.Logger,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    dataset_dir = dataset_path.parent
    image_cache_dir = output_dir / "image_cache"
    image_cache_dir.mkdir(parents=True, exist_ok=True)

    detector, detector_state = load_object_detector(enable_object_detection, object_model_name, logger)
    embedding_dim = get_embedding_dim(model, tokenizer, device)
    logger.info("Embedding dimension: %s", embedding_dim)

    text_embeddings = np.full((len(sample_df), embedding_dim), np.nan, dtype=np.float32)
    image_embeddings = np.full((len(sample_df), embedding_dim), np.nan, dtype=np.float32)
    file_search_cache: dict[str, Path | None] = {}
    result_rows: list[dict[str, Any]] = []

    for row_index, (_, row) in enumerate(tqdm(sample_df.iterrows(), total=len(sample_df), desc="Processing sample")):
        row_dict = row.to_dict()
        text_value, text_column_used = first_non_empty_value(row, text_columns)
        image_value, image_column_used = first_non_empty_value(row, image_columns)

        record: dict[str, Any] = {
            **row_dict,
            "record_id": build_record_id(row, id_column),
            "label": row[label_column] if label_column in row else row_dict.get("label"),
            "label_source_column": label_column,
            "text_column_used_for_embedding": text_column_used or "",
            "text_for_embedding": text_value or "",
            "image_column_used": image_column_used or "",
            "image_source_value": image_value or "",
            "resolved_image_path": "",
            "resolved_image_kind": "",
            "text_embedding_status": "missing",
            "image_embedding_status": "missing",
            "object_detection_status": detector_state,
            "object_labels": "",
            "error_message": "",
            "failure_reasons": "",
            "text_embedding_index": row_index,
            "image_embedding_index": row_index,
        }

        errors: list[str] = []

        if text_value is not None:
            try:
                text_embeddings[row_index] = encode_text_feature(text_value, model, tokenizer, device)
                record["text_embedding_status"] = "success"
            except Exception as exc:
                record["text_embedding_status"] = "failed"
                errors.append(f"text_error={exc}")

        if image_value is not None:
            try:
                image_path, image_kind = resolve_image_to_local_path(
                    image_value=image_value,
                    dataset_dir=dataset_dir,
                    project_root=PROJECT_ROOT,
                    image_cache_dir=image_cache_dir,
                    file_search_cache=file_search_cache,
                )
                record["resolved_image_path"] = str(image_path)
                record["resolved_image_kind"] = image_kind

                if detector_state == "ready" and detector is not None:
                    try:
                        detected_objects = run_object_detection(detector, image_path)
                        if detected_objects:
                            record["object_detection_status"] = "success"
                            record["object_labels"] = ",".join(detected_objects)
                        else:
                            record["object_detection_status"] = "no_objects"
                    except Exception as exc:
                        record["object_detection_status"] = "failed"
                        errors.append(f"object_detection_error={exc}")

                image_embeddings[row_index] = encode_image_feature(image_path, model, preprocess, device)
                record["image_embedding_status"] = "success"
            except (FileNotFoundError, HTTPError, URLError, UnidentifiedImageError, OSError, ValueError) as exc:
                record["image_embedding_status"] = "failed"
                errors.append(f"image_error={exc}")
            except Exception as exc:
                record["image_embedding_status"] = "failed"
                errors.append(f"image_error={exc}")

        text_success = record["text_embedding_status"] == "success"
        image_success = record["image_embedding_status"] == "success"
        if text_success and image_success:
            record["row_feature_status"] = "success_both"
        elif text_success:
            record["row_feature_status"] = "success_text_only"
        elif image_success:
            record["row_feature_status"] = "success_image_only"
        else:
            record["row_feature_status"] = "no_features"

        reasons = build_failure_reasons(
            text_status=record["text_embedding_status"],
            image_status=record["image_embedding_status"],
            detector_status=record["object_detection_status"],
            text_columns_exist=bool(text_columns),
            image_columns_exist=bool(image_columns),
        )
        record["failure_reasons"] = "; ".join(reasons)
        record["error_message"] = " | ".join(errors)
        result_rows.append(record)

    result_df = pd.DataFrame(result_rows)
    ensure_label_column(result_df, logger)
    return result_df, text_embeddings, image_embeddings


def reorder_output_columns(df: pd.DataFrame) -> pd.DataFrame:
    preferred_front = [
        "record_id",
        "source_row_number",
        "label",
        "label_source_column",
        "text_for_embedding",
        "text_column_used_for_embedding",
        "image_source_value",
        "image_column_used",
        "resolved_image_path",
        "resolved_image_kind",
        "text_embedding_status",
        "image_embedding_status",
        "object_detection_status",
        "object_labels",
        "row_feature_status",
        "failure_reasons",
        "error_message",
        "text_embedding_index",
        "image_embedding_index",
    ]
    remaining_columns = [column for column in df.columns if column not in preferred_front]
    ordered_columns = [column for column in preferred_front if column in df.columns] + remaining_columns
    return df.loc[:, ordered_columns]


def build_summary(result_df: pd.DataFrame) -> dict[str, Any]:
    return {
        "rows_total": int(len(result_df)),
        "rows_success_both": int((result_df["row_feature_status"] == "success_both").sum()),
        "rows_success_text_only": int((result_df["row_feature_status"] == "success_text_only").sum()),
        "rows_success_image_only": int((result_df["row_feature_status"] == "success_image_only").sum()),
        "rows_no_features": int((result_df["row_feature_status"] == "no_features").sum()),
        "text_success": int((result_df["text_embedding_status"] == "success").sum()),
        "text_missing": int((result_df["text_embedding_status"] == "missing").sum()),
        "text_failed": int((result_df["text_embedding_status"] == "failed").sum()),
        "image_success": int((result_df["image_embedding_status"] == "success").sum()),
        "image_missing": int((result_df["image_embedding_status"] == "missing").sum()),
        "image_failed": int((result_df["image_embedding_status"] == "failed").sum()),
        "failed_rows_file_count": int((result_df["failure_reasons"].fillna("") != "").sum()),
    }


def save_outputs(
    sample_size_requested: int,
    sample_df: pd.DataFrame,
    result_df: pd.DataFrame,
    text_embeddings: np.ndarray,
    image_embeddings: np.ndarray,
    output_dir: Path,
    metadata: dict[str, Any],
    logger: logging.Logger,
) -> dict[str, Path]:
    sample_output_path = output_dir / f"twitter_sample_{sample_size_requested}.xlsx"
    features_csv_path = output_dir / f"clip_features_{sample_size_requested}.csv"
    features_xlsx_path = output_dir / f"clip_features_{sample_size_requested}.xlsx"
    failed_rows_path = output_dir / "failed_rows.csv"
    text_embeddings_path = output_dir / f"text_embeddings_{sample_size_requested}.npy"
    image_embeddings_path = output_dir / f"image_embeddings_{sample_size_requested}.npy"
    metadata_path = output_dir / "run_metadata.json"

    sample_df.to_excel(sample_output_path, index=False)
    logger.info("Saved sample file: %s", sample_output_path)

    ordered_result_df = reorder_output_columns(result_df)
    ordered_result_df.to_csv(features_csv_path, index=False, encoding="utf-8-sig")
    ordered_result_df.to_excel(features_xlsx_path, index=False)
    logger.info("Saved feature tables: %s and %s", features_csv_path, features_xlsx_path)

    failed_rows_df = ordered_result_df.loc[ordered_result_df["failure_reasons"].fillna("") != ""].copy()
    failed_rows_df.to_csv(failed_rows_path, index=False, encoding="utf-8-sig")
    logger.info("Saved failed rows file: %s", failed_rows_path)

    np.save(text_embeddings_path, text_embeddings)
    np.save(image_embeddings_path, image_embeddings)
    logger.info("Saved embeddings: %s and %s", text_embeddings_path, image_embeddings_path)

    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Saved metadata file: %s", metadata_path)

    return {
        "sample_output_path": sample_output_path,
        "features_csv_path": features_csv_path,
        "features_xlsx_path": features_xlsx_path,
        "failed_rows_path": failed_rows_path,
        "text_embeddings_path": text_embeddings_path,
        "image_embeddings_path": image_embeddings_path,
        "metadata_path": metadata_path,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a safe CLIP pipeline on a 500-row Twitter sample.")
    parser.add_argument(
        "--input-xlsx",
        default=None,
        help="Path to the source .xlsx dataset. If omitted, the script auto-detects one .xlsx file in the project root.",
    )
    parser.add_argument(
        "--sheet-name",
        default=0,
        help="Excel sheet name or sheet index to read. Default: first sheet.",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=500,
        help="Number of rows for the test sample. Default: 500.",
    )
    parser.add_argument(
        "--sample-mode",
        choices=["first", "valid"],
        default="valid",
        help="Use the first rows or the first valid rows (valid means at least text or image exists). Default: valid.",
    )
    parser.add_argument(
        "--model",
        default="ViT-B-32",
        help="open_clip model name. Default: ViT-B-32.",
    )
    parser.add_argument(
        "--pretrained",
        default="openai",
        help="Pretrained checkpoint name for open_clip. Default: openai.",
    )
    parser.add_argument(
        "--sample-only",
        action="store_true",
        help="Only create the sample file and stop before loading CLIP.",
    )
    parser.add_argument(
        "--enable-object-detection",
        action="store_true",
        help="Optional: also run object detection if ultralytics is installed.",
    )
    parser.add_argument(
        "--object-model",
        default="yolov8n.pt",
        help="Object detector checkpoint name when --enable-object-detection is used.",
    )
    return parser.parse_args()


def main() -> int:
    configure_stdout()
    args = parse_args()

    output_dir = SCRIPT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logger(output_dir / "run_log.txt")

    try:
        dataset_path = resolve_input_xlsx(args.input_xlsx, PROJECT_ROOT)
        if not dataset_path.exists():
            raise FileNotFoundError(f"Dataset file does not exist: {dataset_path}")

        logger.info("Project root: %s", PROJECT_ROOT)
        logger.info("Output directory: %s", output_dir)
        logger.info("Input dataset: %s", dataset_path)

        df = pd.read_excel(dataset_path, sheet_name=args.sheet_name)
        logger.info("Loaded dataset with shape: %s", df.shape)
        logger.info("Dataset columns: %s", list(df.columns))
        print("\nDataset columns:")
        for column in df.columns:
            print(f"  - {column}")

        columns = [str(column) for column in df.columns]
        label_column = detect_single_column(columns, LABEL_COLUMN_CANDIDATES)
        id_column = detect_single_column(columns, ID_COLUMN_CANDIDATES)
        text_columns = detect_candidate_columns(columns, TEXT_COLUMN_CANDIDATES)
        image_columns = detect_candidate_columns(columns, IMAGE_COLUMN_CANDIDATES)

        if not label_column:
            raise ValueError(
                "Could not find a label column. Checked candidates: "
                f"{LABEL_COLUMN_CANDIDATES}"
            )
        if not text_columns and not image_columns:
            raise ValueError(
                "Could not find any text or image columns. "
                f"Text candidates: {TEXT_COLUMN_CANDIDATES}. "
                f"Image candidates: {IMAGE_COLUMN_CANDIDATES}."
            )

        logger.info("Detected id column: %s", id_column or "<none>")
        logger.info("Detected label column: %s", label_column)
        logger.info("Detected text columns: %s", text_columns or "<none>")
        logger.info("Detected image columns: %s", image_columns or "<none>")

        sample_df = create_sample_dataframe(
            df=df,
            sample_size=args.sample_size,
            sample_mode=args.sample_mode,
            id_column=id_column,
            label_column=label_column,
            text_columns=text_columns,
            image_columns=image_columns,
            logger=logger,
        )
        ensure_label_column(sample_df, logger)

        sample_output_path = output_dir / f"twitter_sample_{args.sample_size}.xlsx"
        sample_df.to_excel(sample_output_path, index=False)
        logger.info("Sample file created before CLIP: %s", sample_output_path)

        if args.sample_only:
            logger.info("Sample-only mode enabled. Stopping before CLIP inference.")
            return 0

        device = choose_device()
        logger.info("Device selected: %s", device)
        if device == "cuda":
            logger.info("CUDA device name: %s", torch.cuda.get_device_name(0))
        else:
            logger.info("CUDA unavailable. Falling back to CPU.")

        model, preprocess, tokenizer = load_clip_components(
            model_name=args.model,
            pretrained=args.pretrained,
            device=device,
        )
        logger.info("Loaded CLIP model: model=%s, pretrained=%s", args.model, args.pretrained)

        result_df, text_embeddings, image_embeddings = process_sample(
            sample_df=sample_df,
            dataset_path=dataset_path,
            output_dir=output_dir,
            id_column=id_column,
            text_columns=text_columns,
            image_columns=image_columns,
            label_column=label_column,
            model=model,
            preprocess=preprocess,
            tokenizer=tokenizer,
            device=device,
            enable_object_detection=args.enable_object_detection,
            object_model_name=args.object_model,
            logger=logger,
        )

        summary = build_summary(result_df)
        metadata = {
            "input_dataset": str(dataset_path),
            "output_directory": str(output_dir),
            "sample_mode": args.sample_mode,
            "sample_size_requested": args.sample_size,
            "sample_size_actual": int(len(sample_df)),
            "device": device,
            "clip_model": args.model,
            "clip_pretrained": args.pretrained,
            "detected_columns": {
                "id_column": id_column,
                "label_column": label_column,
                "text_columns": text_columns,
                "image_columns": image_columns,
            },
            "summary": summary,
        }

        output_paths = save_outputs(
            sample_size_requested=args.sample_size,
            sample_df=sample_df,
            result_df=result_df,
            text_embeddings=text_embeddings,
            image_embeddings=image_embeddings,
            output_dir=output_dir,
            metadata=metadata,
            logger=logger,
        )

        logger.info("Run completed successfully.")
        for key, value in output_paths.items():
            logger.info("%s: %s", key, value)
        logger.info("Summary: %s", summary)
        return 0
    except Exception as exc:
        logger.exception("Pipeline failed: %s", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
