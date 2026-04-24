from __future__ import annotations

import torch

from config import multimodal_pt_input_path


def check_structure() -> None:
    file_path = multimodal_pt_input_path()
    if not file_path.exists():
        print("Missing multimodal feature file.")
        return

    data = torch.load(file_path, map_location="cpu")

    print(f"--- FILE INFO: {file_path.name} ---")
    print(f"Total records: {len(data)}")

    if data:
        first_record = data[0]
        print("\nKeys and basic shapes:")
        print("-" * 40)
        for key, value in first_record.items():
            data_type = type(value).__name__
            if isinstance(value, (list, torch.Tensor)):
                shape = len(value) if isinstance(value, list) else tuple(value.shape)
                print(f"| {key:<22} | Type: {data_type:<10} | Shape/Len: {shape}")
            else:
                print(f"| {key:<22} | Type: {data_type:<10} | Value: {value}")
        print("-" * 40)


if __name__ == "__main__":
    check_structure()
