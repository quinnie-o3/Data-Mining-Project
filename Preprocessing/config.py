from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PREPROCESSING_DIR = PROJECT_ROOT / "Preprocessing"
DATA_DIR = PREPROCESSING_DIR / "Data"

RAW_DIR = DATA_DIR / "RAW"
INTERIM_DIR = DATA_DIR / "INTERIM"
FINAL_DIR = DATA_DIR / "FINAL"
LOG_DIR = DATA_DIR / "LOGS"
CACHE_DIR = DATA_DIR / "CACHE"

IMAGE_DIR = PROJECT_ROOT / "image"
REFERENCES_DIR = PROJECT_ROOT / "references"

RAW_EXCEL = RAW_DIR / "[TWITTER] danh_sach_final_binary.xlsx"
DOWNLOADED_CSV = INTERIM_DIR / "[TWITTER] danh_sach_downloaded.csv"

OBJECTS_JSONL = INTERIM_DIR / "clip_twitter_objects_final.jsonl"
OBJECTS_PT = INTERIM_DIR / "clip_twitter_objects_final.pt"
OBJECTS_LOG = LOG_DIR / "clip_twitter_objects_final.log"

TEXT_FEATURES_PT = INTERIM_DIR / "clip_twitter_text_features.pt"
MULTIMODAL_FEATURES_PT = FINAL_DIR / "twitter_multimodal_features.pt"
MULTIMODAL_FEATURES_JSONL = FINAL_DIR / "twitter_multimodal_features.jsonl"

TORCH_CACHE_DIR = CACHE_DIR / "torch"
HF_CACHE_DIR = CACHE_DIR / "huggingface"

LEGACY_RAW_EXCEL = PROJECT_ROOT / "[TWITTER] danh_sach_final_binary.xlsx"
LEGACY_DOWNLOADED_CSV = PROJECT_ROOT / "[TWITTER] danh_sach_downloaded.csv"
LEGACY_DOWNLOADED_CSV_RAW_COPY = RAW_DIR / "[TWITTER] danh_sach_downloaded - Copy.csv"
LEGACY_DOWNLOADED_CSV_RAW_COPY_2 = RAW_DIR / "[TWITTER] danh_sach_downloaded - Copy - Copy.csv"
LEGACY_OBJECTS_JSONL_ROOT = PROJECT_ROOT / "clip_twitter_objects_final.jsonl"
LEGACY_OBJECTS_JSONL_ARCHIVE = DATA_DIR / "File quá trình xử lý CLIP" / "clip_twitter_objects_final.jsonl"
LEGACY_TEXT_FEATURES_PT_ROOT = PROJECT_ROOT / "clip_twitter_text_features.pt"
LEGACY_TEXT_FEATURES_PT_ARCHIVE = DATA_DIR / "File quá trình xử lý CLIP" / "clip_twitter_text_features.pt"
LEGACY_MULTIMODAL_PT_ROOT = PROJECT_ROOT / "twitter_multimodal_features.pt"
LEGACY_MULTIMODAL_JSONL_ROOT = PROJECT_ROOT / "twitter_multimodal_features.jsonl"


def ensure_data_dirs() -> None:
    for directory in (
        RAW_DIR,
        INTERIM_DIR,
        FINAL_DIR,
        LOG_DIR,
        CACHE_DIR,
        TORCH_CACHE_DIR,
        HF_CACHE_DIR,
        IMAGE_DIR,
        REFERENCES_DIR,
    ):
        directory.mkdir(parents=True, exist_ok=True)


def first_existing_path(preferred: Path, *fallbacks: Path) -> Path:
    for candidate in (preferred, *fallbacks):
        if candidate.exists():
            return candidate
    return preferred


def input_excel_path() -> Path:
    return first_existing_path(RAW_EXCEL, LEGACY_RAW_EXCEL)


def downloaded_csv_input_path() -> Path:
    return first_existing_path(
        DOWNLOADED_CSV,
        LEGACY_DOWNLOADED_CSV,
        LEGACY_DOWNLOADED_CSV_RAW_COPY,
        LEGACY_DOWNLOADED_CSV_RAW_COPY_2,
    )


def objects_jsonl_input_path() -> Path:
    return first_existing_path(
        OBJECTS_JSONL,
        LEGACY_OBJECTS_JSONL_ROOT,
        LEGACY_OBJECTS_JSONL_ARCHIVE,
    )


def text_features_input_path() -> Path:
    return first_existing_path(
        TEXT_FEATURES_PT,
        LEGACY_TEXT_FEATURES_PT_ROOT,
        LEGACY_TEXT_FEATURES_PT_ARCHIVE,
    )


def multimodal_pt_input_path() -> Path:
    return first_existing_path(MULTIMODAL_FEATURES_PT, LEGACY_MULTIMODAL_PT_ROOT)


def multimodal_jsonl_input_path() -> Path:
    return first_existing_path(MULTIMODAL_FEATURES_JSONL, LEGACY_MULTIMODAL_JSONL_ROOT)
