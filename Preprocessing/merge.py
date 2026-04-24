from __future__ import annotations

import json

import torch

from config import (
    MULTIMODAL_FEATURES_PT,
    ensure_data_dirs,
    objects_jsonl_input_path,
    text_features_input_path,
)


def main() -> None:
    ensure_data_dirs()
    file_image = objects_jsonl_input_path()
    file_text = text_features_input_path()

    print("Loading text features...")
    text_data = torch.load(file_text, map_location="cpu")
    text_lookup = {item["sample_id"]: item for item in text_data}

    print("Loading image features...")
    combined_data = []
    missing_count = 0

    with open(file_image, "r", encoding="utf-8") as handle:
        for line in handle:
            image_record = json.loads(line)
            sample_id = image_record["sample_id"]

            if sample_id in text_lookup:
                text_record = text_lookup[sample_id]
                combined_data.append(
                    {
                        "sample_id": sample_id,
                        "label": image_record["label"],
                        "text": image_record["text"],
                        "image_path": image_record["image_path"],
                        "boxes": image_record["boxes"],
                        "labels_obj": image_record["labels"],
                        "full_image_embedding": image_record["full_image_embedding"],
                        "object_embeddings": image_record["object_embeddings"],
                        "num_objects": image_record["num_objects"],
                        "text_embedding": text_record["text_embedding"],
                        "token_embeddings": text_record["token_embeddings"],
                        "attention_mask": text_record["attention_mask"],
                    }
                )
            else:
                missing_count += 1

    print(f"Merged records: {len(combined_data)}")
    if missing_count > 0:
        print(f"Warning: {missing_count} image records did not match a text record.")

    torch.save(combined_data, MULTIMODAL_FEATURES_PT)
    print(f"Multimodal feature file saved at: {MULTIMODAL_FEATURES_PT}")


if __name__ == "__main__":
    main()
