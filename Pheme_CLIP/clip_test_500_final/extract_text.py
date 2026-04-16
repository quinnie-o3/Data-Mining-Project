import json
from pathlib import Path

import torch
from transformers import CLIPModel, CLIPTokenizer


SCRIPT_DIR = Path(__file__).resolve().parent
INPUT_PT = SCRIPT_DIR / "clip_500_image_features.pt"
INPUT_JSONL = SCRIPT_DIR / "clip_500_image_features.jsonl"
OUTPUT_PT = SCRIPT_DIR / "clip_500_text_features.pt"
CACHE_DIR = SCRIPT_DIR / "model_cache" / "huggingface"
CLIP_MODEL_NAME = "openai/clip-vit-base-patch16"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PROGRESS_EVERY = 50


def load_image_records() -> list[dict]:
    if INPUT_PT.exists():
        records = torch.load(INPUT_PT, map_location="cpu", weights_only=False)
    elif INPUT_JSONL.exists():
        with INPUT_JSONL.open("r", encoding="utf-8") as jsonl_file:
            records = [json.loads(line) for line in jsonl_file if line.strip()]
    else:
        raise FileNotFoundError(
            f"Missing image result input. Expected one of: {INPUT_PT} or {INPUT_JSONL}"
        )

    if not records:
        raise RuntimeError("No image records found for the 500-row text extraction run")

    return records


def normalize_embedding(embedding: torch.Tensor) -> torch.Tensor:
    return embedding / embedding.norm(p=2, dim=-1, keepdim=True).clamp(min=1e-12)


def main() -> None:
    print(f"Running on device: {DEVICE}")
    print("Loading 500-row image references...")
    image_records = load_image_records()

    ok_records = [record for record in image_records if record.get("status") == "ok"]
    if not ok_records:
        raise RuntimeError("No successful image records found to align text features with")

    print(f"Loaded {len(ok_records)} successful records from image pipeline")
    print(f"Loading CLIP text model from cache: {CACHE_DIR}")

    tokenizer = CLIPTokenizer.from_pretrained(
        CLIP_MODEL_NAME,
        cache_dir=str(CACHE_DIR),
        local_files_only=True,
    )
    clip_model = CLIPModel.from_pretrained(
        CLIP_MODEL_NAME,
        cache_dir=str(CACHE_DIR),
        local_files_only=True,
    ).to(DEVICE)
    clip_model.eval()

    text_records: list[dict] = []

    with torch.no_grad():
        for index, record in enumerate(ok_records, start=1):
            sample_id = str(record["sample_id"])
            text = str(record["text"]).strip()
            label = str(record["label"]).strip()

            if not text:
                raise ValueError(f"Empty text found in referenced 500-row data for sample_id {sample_id}")

            inputs = tokenizer(
                text,
                padding="max_length",
                max_length=77,
                truncation=True,
                return_tensors="pt",
            )
            inputs = {key: value.to(DEVICE) for key, value in inputs.items()}
            input_ids = inputs["input_ids"].squeeze(0).cpu()

            text_outputs = clip_model.text_model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
            )
            text_features = clip_model.text_projection(text_outputs.pooler_output)
            text_embedding = normalize_embedding(text_features).squeeze(0).cpu()
            last_hidden = text_outputs.last_hidden_state.squeeze(0).cpu()
            attention_mask = inputs["attention_mask"].squeeze(0).cpu()
            keep_mask = attention_mask.bool()
            real_token_embeddings = last_hidden[keep_mask]
            real_token_ids = input_ids[keep_mask]
            real_tokens = tokenizer.convert_ids_to_tokens(real_token_ids.tolist())

            text_records.append(
                {
                    "sample_id": sample_id,
                    "label": label,
                    "text": text,
                    "text_embedding": text_embedding.tolist(),
                    "token_ids": real_token_ids.tolist(),
                    "tokens": real_tokens,
                    "token_embeddings": real_token_embeddings.tolist(),
                    "attention_mask": attention_mask.tolist(),
                }
            )

            if index == 1 or index % PROGRESS_EVERY == 0 or index == len(ok_records):
                print(f"Processed {index}/{len(ok_records)} records")

    image_ids = [str(record["sample_id"]) for record in ok_records]
    text_ids = [record["sample_id"] for record in text_records]
    if image_ids != text_ids:
        raise RuntimeError("Text feature order does not match the 500-row image reference order")

    torch.save(text_records, OUTPUT_PT)
    print(f"Saved {len(text_records)} text records to: {OUTPUT_PT}")


if __name__ == "__main__":
    main()
