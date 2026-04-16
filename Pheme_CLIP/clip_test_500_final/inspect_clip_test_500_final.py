from pathlib import Path

import torch


INPUT_PT = Path(__file__).resolve().parent / "clip_500_image_features.pt"
INDEX = 0


def embedding_dim(value):
    return len(value) if value else 0


def main() -> None:
    if not INPUT_PT.exists():
        raise FileNotFoundError(
            f"Output file not found: {INPUT_PT}. "
            "Run detect_objects_and_extract_clip_500.py successfully first."
        )

    records = torch.load(INPUT_PT, map_location="cpu")
    if not records:
        raise RuntimeError(f"No records found in {INPUT_PT}")

    if INDEX < 0 or INDEX >= len(records):
        raise IndexError(f"INDEX {INDEX} is out of range for {len(records)} records")

    record = records[INDEX]
    print(f"id: {record['sample_id']}")
    print(f"status: {record['status']}")
    print(f"number_of_detected_objects: {record['num_objects']}")
    print(f"full_image_embedding_dim: {embedding_dim(record['full_image_embedding'])}")

    object_embeddings = record.get("object_embeddings", [])
    if object_embeddings:
        print(f"object_embedding_dim: {embedding_dim(object_embeddings[0])}")
    else:
        print("object_embedding_dim: 0")


if __name__ == "__main__":
    main()
