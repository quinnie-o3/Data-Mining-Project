from __future__ import annotations

import json

import torch

from config import objects_jsonl_input_path, text_features_input_path


def main() -> None:
    file_image = objects_jsonl_input_path()
    file_text = text_features_input_path()

    text_data = torch.load(file_text, map_location="cpu", weights_only=False)
    valid_text_ids = {item["sample_id"] for item in text_data}

    missing_records = []
    with open(file_image, "r", encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            sample_id = record["sample_id"]
            if sample_id not in valid_text_ids:
                missing_records.append(
                    {
                        "sample_id": sample_id,
                        "status": record.get("status", "unknown"),
                        "text": record.get("text", ""),
                    }
                )

    print(f"Total image records missing from text branch: {len(missing_records)}")
    print("-" * 50)
    for record in missing_records:
        print(
            f"ID: {record['sample_id']} | "
            f"Image status: {record['status']} | "
            f"Text: '{record['text']}'"
        )


if __name__ == "__main__":
    main()
