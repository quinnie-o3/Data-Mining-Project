from pathlib import Path

import torch


SCRIPT_DIR = Path(__file__).resolve().parent
IMAGE_FEATURES_PT = SCRIPT_DIR / "clip_500_image_features.pt"
TEXT_FEATURES_PT = SCRIPT_DIR / "clip_500_text_features.pt"
OUTPUT_PT = SCRIPT_DIR / "clip_500_multimodal_features.pt"


def load_records(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"Missing input file: {path}")
    records = torch.load(path, map_location="cpu", weights_only=False)
    if not records:
        raise RuntimeError(f"No records found in {path}")
    return records


def main() -> None:
    image_records = load_records(IMAGE_FEATURES_PT)
    text_records = load_records(TEXT_FEATURES_PT)

    image_ok_records = [record for record in image_records if record.get("status") == "ok"]
    text_by_id = {str(record["sample_id"]): record for record in text_records}

    merged_records: list[dict] = []

    for image_record in image_ok_records:
        sample_id = str(image_record["sample_id"])
        text_record = text_by_id.get(sample_id)
        if text_record is None:
            raise KeyError(f"Missing text features for sample_id {sample_id}")

        image_text = str(image_record.get("text", "")).strip()
        text_text = str(text_record.get("text", "")).strip()
        if image_text != text_text:
            raise ValueError(f"Text mismatch for sample_id {sample_id}")

        image_label = str(image_record.get("label", "")).strip()
        text_label = str(text_record.get("label", "")).strip()
        if image_label != text_label:
            raise ValueError(f"Label mismatch for sample_id {sample_id}")

        image_embedding = image_record.get("full_image_embedding", [])
        text_embedding = text_record.get("text_embedding", [])
        if not image_embedding or not text_embedding:
            raise ValueError(f"Missing feature vector for sample_id {sample_id}")

        multimodal_embedding = image_embedding + text_embedding

        merged_records.append(
            {
                "sample_id": sample_id,
                "label": image_label,
                "text": image_text,
                "image_path": image_record.get("image_path", ""),
                "image_embedding": image_embedding,
                "text_embedding": text_embedding,
                "multimodal_embedding": multimodal_embedding,
                "object_boxes": image_record.get("boxes", []),
                "object_scores": image_record.get("scores", []),
                "object_labels": image_record.get("labels", []),
                "num_objects": image_record.get("num_objects", 0),
                "object_embeddings": image_record.get("object_embeddings", []),
                "token_ids": text_record.get("token_ids", []),
                "tokens": text_record.get("tokens", []),
                "token_embeddings": text_record.get("token_embeddings", []),
                "attention_mask": text_record.get("attention_mask", []),
            }
        )

    if len(merged_records) != len(text_records):
        raise RuntimeError(
            f"Merged record count mismatch: {len(merged_records)} vs {len(text_records)} text records"
        )

    torch.save(merged_records, OUTPUT_PT)

    print(f"Saved {len(merged_records)} merged records to: {OUTPUT_PT}")
    print(f"Image embedding dim: {len(merged_records[0]['image_embedding'])}")
    print(f"Text embedding dim: {len(merged_records[0]['text_embedding'])}")
    print(f"Multimodal embedding dim: {len(merged_records[0]['multimodal_embedding'])}")


if __name__ == "__main__":
    main()
