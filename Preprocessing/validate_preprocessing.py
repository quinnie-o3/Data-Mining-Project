from __future__ import annotations

from collections import Counter

import torch

from config import (
    DOWNLOADED_CSV,
    MULTIMODAL_FEATURES_PT,
    OBJECTS_JSONL,
    TEXT_FEATURES_PT,
    downloaded_csv_input_path,
    multimodal_pt_input_path,
    objects_jsonl_input_path,
    text_features_input_path,
)


def count_jsonl_rows(path) -> int:
    with open(path, "r", encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def count_csv_rows(path) -> int:
    with open(path, "r", encoding="utf-8-sig") as handle:
        return max(sum(1 for _ in handle) - 1, 0)


def main() -> None:
    downloaded_csv = downloaded_csv_input_path()
    objects_jsonl = objects_jsonl_input_path()
    text_pt = text_features_input_path()
    multimodal_pt = multimodal_pt_input_path()

    print("Canonical targets:")
    print(f"- downloaded_csv: {DOWNLOADED_CSV}")
    print(f"- objects_jsonl: {OBJECTS_JSONL}")
    print(f"- text_features_pt: {TEXT_FEATURES_PT}")
    print(f"- multimodal_features_pt: {MULTIMODAL_FEATURES_PT}")

    print("Resolved inputs:")
    print(f"- downloaded_csv: {downloaded_csv}")
    print(f"- objects_jsonl: {objects_jsonl}")
    print(f"- text_features_pt: {text_pt}")
    print(f"- multimodal_features_pt: {multimodal_pt}")

    if downloaded_csv.exists():
        print(f"Downloaded manifest rows: {count_csv_rows(downloaded_csv)}")
    if objects_jsonl.exists():
        print(f"Object JSONL rows: {count_jsonl_rows(objects_jsonl)}")
    if text_pt.exists():
        text_data = torch.load(text_pt, map_location="cpu", weights_only=False)
        print(f"Text PT rows: {len(text_data)}")
    if multimodal_pt.exists():
        data = torch.load(multimodal_pt, map_location="cpu", weights_only=False)
        labels = Counter(record["label"] for record in data)
        object_counts = [record.get("num_objects", 0) for record in data]
        print(f"Multimodal PT rows: {len(data)}")
        print(f"Label distribution: {dict(labels)}")
        print(f"Object count avg: {sum(object_counts) / len(object_counts):.4f}")
        print(f"Object count zero rows: {sum(1 for value in object_counts if value == 0)}")
        if data:
            print(f"Schema keys: {list(data[0].keys())}")


if __name__ == "__main__":
    main()
