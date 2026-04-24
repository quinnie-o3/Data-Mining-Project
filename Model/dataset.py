from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset


class MultimodalFeatureDataset(Dataset):
    def __init__(self, data_path: str | Path):
        self.data_path = Path(data_path)
        self.records: list[dict[str, Any]] = torch.load(
            self.data_path,
            map_location="cpu",
            weights_only=False,
        )

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.records[index]


def _to_float_tensor(value: Any) -> torch.Tensor:
    return torch.tensor(value, dtype=torch.float32)


def collate_multimodal_features(records: list[dict[str, Any]]) -> dict[str, Any]:
    tweet_ids = [record.get("tweet_id", record.get("sample_id")) for record in records]
    labels = torch.tensor([record["label"] for record in records], dtype=torch.long)
    text_embeddings = torch.stack([_to_float_tensor(record["text_embedding"]) for record in records])
    image_embeddings = torch.stack([_to_float_tensor(record["full_image_embedding"]) for record in records])
    num_objects = torch.tensor([record.get("num_objects", 0) for record in records], dtype=torch.float32)

    return {
        "tweet_id": tweet_ids,
        "text": [record["text"] for record in records],
        "image_path": [record["image_path"] for record in records],
        "label": labels,
        "text_embedding": text_embeddings,
        "full_image_embedding": image_embeddings,
        "num_objects": num_objects,
        "raw_records": records,
    }
