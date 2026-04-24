from __future__ import annotations

import shutil
from pathlib import Path

from config import (
    DOWNLOADED_CSV,
    LEGACY_DOWNLOADED_CSV,
    LEGACY_DOWNLOADED_CSV_RAW_COPY,
    LEGACY_DOWNLOADED_CSV_RAW_COPY_2,
    LEGACY_MULTIMODAL_JSONL_ROOT,
    LEGACY_MULTIMODAL_PT_ROOT,
    LEGACY_OBJECTS_JSONL_ARCHIVE,
    LEGACY_OBJECTS_JSONL_ROOT,
    LEGACY_RAW_EXCEL,
    LEGACY_TEXT_FEATURES_PT_ARCHIVE,
    LEGACY_TEXT_FEATURES_PT_ROOT,
    MULTIMODAL_FEATURES_JSONL,
    MULTIMODAL_FEATURES_PT,
    OBJECTS_JSONL,
    RAW_EXCEL,
    TEXT_FEATURES_PT,
    ensure_data_dirs,
)


MIGRATION_PAIRS: list[tuple[Path, Path]] = [
    (LEGACY_RAW_EXCEL, RAW_EXCEL),
    (LEGACY_DOWNLOADED_CSV, DOWNLOADED_CSV),
    (LEGACY_DOWNLOADED_CSV_RAW_COPY, DOWNLOADED_CSV),
    (LEGACY_DOWNLOADED_CSV_RAW_COPY_2, DOWNLOADED_CSV),
    (LEGACY_OBJECTS_JSONL_ROOT, OBJECTS_JSONL),
    (LEGACY_OBJECTS_JSONL_ARCHIVE, OBJECTS_JSONL),
    (LEGACY_TEXT_FEATURES_PT_ROOT, TEXT_FEATURES_PT),
    (LEGACY_TEXT_FEATURES_PT_ARCHIVE, TEXT_FEATURES_PT),
    (LEGACY_MULTIMODAL_PT_ROOT, MULTIMODAL_FEATURES_PT),
    (LEGACY_MULTIMODAL_JSONL_ROOT, MULTIMODAL_FEATURES_JSONL),
]


def copy_if_missing(source: Path, destination: Path) -> None:
    if destination.exists() or not source.exists():
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    print(f"Copied {source} -> {destination}")


def main() -> None:
    ensure_data_dirs()
    for source, destination in MIGRATION_PAIRS:
        copy_if_missing(source, destination)
    print("Legacy artifact migration finished.")


if __name__ == "__main__":
    main()
