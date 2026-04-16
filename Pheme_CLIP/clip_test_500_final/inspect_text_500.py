from pathlib import Path

import torch


INPUT_PT = Path(__file__).resolve().parent / "text_features_final.pt"
INDEX = 0


def embedding_dim(value):
    return len(value) if value else 0


def main() -> None:
    if not INPUT_PT.exists():
        raise FileNotFoundError(
            f"Output file not found: {INPUT_PT}. "
            "Run extract_text.py successfully first."
        )

    records = torch.load(INPUT_PT, map_location="cpu", weights_only=False)
    if not records:
        raise RuntimeError(f"No records found in {INPUT_PT}")

    if INDEX < 0 or INDEX >= len(records):
        raise IndexError(f"INDEX {INDEX} is out of range for {len(records)} records")

    record = records[INDEX]
    print(f"id: {record['sample_id']}")
    print(f"label: {record['label']}")
    print(f"text_length: {len(record['text'])}")
    print(f"text_embedding_dim: {embedding_dim(record['text_embedding'])}")

    token_embeddings = record.get("token_embeddings", [])
    print(f"token_count: {len(token_embeddings)}")
    if token_embeddings:
        print(f"token_embedding_dim: {embedding_dim(token_embeddings[0])}")
    else:
        print("token_embedding_dim: 0")


if __name__ == "__main__":
    main()
