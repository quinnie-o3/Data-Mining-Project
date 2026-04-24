from __future__ import annotations

import json

import torch

from config import MULTIMODAL_FEATURES_JSONL, ensure_data_dirs, multimodal_pt_input_path


def main() -> None:
    ensure_data_dirs()
    input_pt = multimodal_pt_input_path()
    if not input_pt.exists():
        print(f"Missing input file: {input_pt}")
        return

    print(f"Loading data from {input_pt.name}...")
    data = torch.load(input_pt, map_location="cpu", weights_only=False)
    print(f"Loaded {len(data)} records. Converting to JSONL...")

    with open(MULTIMODAL_FEATURES_JSONL, "w", encoding="utf-8") as handle:
        for idx, record in enumerate(data, start=1):
            processed_record = {}
            for key, value in record.items():
                if isinstance(value, torch.Tensor):
                    processed_record[key] = value.tolist()
                else:
                    processed_record[key] = value

            handle.write(json.dumps(processed_record, ensure_ascii=False) + "\n")

            if idx % 500 == 0:
                print(f"Converted {idx}/{len(data)} rows...")

    print("-" * 30)
    print(f"Done. JSONL saved at: {MULTIMODAL_FEATURES_JSONL}")


if __name__ == "__main__":
    main()
