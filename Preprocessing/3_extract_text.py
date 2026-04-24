from __future__ import annotations

import json

import torch
from transformers import CLIPModel, CLIPTokenizer

from config import HF_CACHE_DIR, TEXT_FEATURES_PT, ensure_data_dirs, objects_jsonl_input_path


CLIP_MODEL_NAME = "openai/clip-vit-base-patch16"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_clip_text_components() -> tuple[CLIPModel, CLIPTokenizer]:
    try:
        tokenizer = CLIPTokenizer.from_pretrained(CLIP_MODEL_NAME, cache_dir=str(HF_CACHE_DIR))
        clip_model = CLIPModel.from_pretrained(CLIP_MODEL_NAME, cache_dir=str(HF_CACHE_DIR)).to(DEVICE)
    except Exception:
        tokenizer = CLIPTokenizer.from_pretrained(CLIP_MODEL_NAME, local_files_only=True)
        clip_model = CLIPModel.from_pretrained(CLIP_MODEL_NAME, local_files_only=True).to(DEVICE)

    clip_model.eval()
    return clip_model, tokenizer


def main() -> None:
    ensure_data_dirs()
    input_jsonl = objects_jsonl_input_path()

    print(f"Running on device: {DEVICE}")
    if not input_jsonl.exists():
        print(f"Missing input JSONL: {input_jsonl}. Run step 2 first.")
        return

    image_records = []
    print(f"Reading image branch data from {input_jsonl}...")
    with open(input_jsonl, "r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                image_records.append(json.loads(line))

    print(f"Loaded {len(image_records)} records from the image branch.")

    print("Loading CLIP model for projected text features...")
    clip_model, tokenizer = load_clip_text_components()

    text_records = []

    print("Extracting text features...")
    with torch.no_grad():
        for idx, record in enumerate(image_records):
            if record.get("status") != "ok":
                continue

            sample_id = record["sample_id"]
            text = record["text"]
            label = record["label"]

            if not isinstance(text, str) or not text.strip():
                continue

            inputs = tokenizer(
                text,
                padding="max_length",
                max_length=77,
                truncation=True,
                return_tensors="pt",
            ).to(DEVICE)

            text_features = clip_model.get_text_features(**inputs)
            text_embedding = text_features.pooler_output if hasattr(text_features, "pooler_output") else text_features
            text_embedding = text_embedding / text_embedding.norm(p=2, dim=-1, keepdim=True).clamp(min=1e-12)
            text_embedding = text_embedding.squeeze(0).cpu()

            outputs = clip_model.text_model(**inputs)
            last_hidden = outputs.last_hidden_state.squeeze(0)

            attention_mask = inputs["attention_mask"].squeeze(0)
            token_embeddings = last_hidden.cpu()
            real_token_embeddings = token_embeddings[attention_mask.bool().cpu()]

            text_records.append(
                {
                    "sample_id": sample_id,
                    "label": label,
                    "text": text,
                    "text_embedding": text_embedding.tolist(),
                    "token_embeddings": real_token_embeddings.tolist(),
                    "attention_mask": attention_mask.cpu().tolist(),
                }
            )

            if (idx + 1) % 50 == 0:
                print(f"Processed {idx + 1}/{len(image_records)} posts...")

    torch.save(text_records, TEXT_FEATURES_PT)
    print(f"\nDone. Extracted CLIP text features for {len(text_records)} records.")
    print(f"Saved at: {TEXT_FEATURES_PT}")


if __name__ == "__main__":
    main()
