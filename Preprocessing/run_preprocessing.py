from __future__ import annotations

import argparse
import runpy
from pathlib import Path

from config import (
    DOWNLOADED_CSV,
    MULTIMODAL_FEATURES_PT,
    OBJECTS_JSONL,
    RAW_EXCEL,
    TEXT_FEATURES_PT,
    ensure_data_dirs,
)


SCRIPT_SEQUENCE = [
    ("download_images", "1_download_images.py", DOWNLOADED_CSV),
    ("extract_image_objects", "2_extract_image_objects.py", OBJECTS_JSONL),
    ("extract_text", "3_extract_text.py", TEXT_FEATURES_PT),
    ("merge_multimodal", "4_merge_multimodal.py", MULTIMODAL_FEATURES_PT),
]


def run_script(script_path: Path) -> None:
    runpy.run_path(str(script_path), run_name="__main__")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Twitter multimodal preprocessing pipeline.")
    parser.add_argument(
        "--start-from",
        choices=[name for name, _, _ in SCRIPT_SEQUENCE],
        default=SCRIPT_SEQUENCE[0][0],
        help="Resume the pipeline from a specific step.",
    )
    parser.add_argument(
        "--stop-after",
        choices=[name for name, _, _ in SCRIPT_SEQUENCE],
        default=SCRIPT_SEQUENCE[-1][0],
        help="Stop the pipeline after a specific step.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip a step when its canonical output already exists.",
    )
    args = parser.parse_args()

    ensure_data_dirs()

    start_index = next(i for i, item in enumerate(SCRIPT_SEQUENCE) if item[0] == args.start_from)
    stop_index = next(i for i, item in enumerate(SCRIPT_SEQUENCE) if item[0] == args.stop_after)
    if stop_index < start_index:
        raise ValueError("--stop-after must be the same step as --start-from or come later.")

    print(f"Canonical raw input location: {RAW_EXCEL}")
    for _, script_name, output_path in SCRIPT_SEQUENCE[start_index:stop_index + 1]:
        if args.skip_existing and output_path.exists():
            print(f"Skipping {script_name} because output already exists at {output_path}")
            continue

        print(f"Running {script_name}")
        run_script(Path(__file__).resolve().parent / script_name)
        print(f"Finished {script_name}")

    print("Preprocessing pipeline completed.")


if __name__ == "__main__":
    main()
