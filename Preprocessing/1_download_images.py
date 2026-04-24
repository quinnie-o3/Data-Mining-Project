from __future__ import annotations

from pathlib import Path

import pandas as pd
import requests
from tqdm import tqdm

from config import DOWNLOADED_CSV, IMAGE_DIR, ensure_data_dirs, input_excel_path


def download_image(url: str, save_path: Path) -> bool:
    if save_path.exists():
        return True

    try:
        response = requests.get(url, stream=True, timeout=10)
        if response.status_code == 200:
            with open(save_path, "wb") as handle:
                for chunk in response.iter_content(1024):
                    handle.write(chunk)
            return True
        return False
    except Exception:
        return False


def main() -> None:
    ensure_data_dirs()
    input_excel = input_excel_path()

    print(f"Reading source Excel: {input_excel}")
    try:
        df = pd.read_excel(input_excel)
    except Exception as error:
        print(f"Failed to read Excel: {error}")
        return

    if "image_url" not in df.columns:
        print("Missing required column: 'image_url'")
        return

    df = df.dropna(subset=["image_url"])
    valid_rows = []

    print(f"Writing downloaded-image manifest to: {DOWNLOADED_CSV}")
    print("Checking images and downloading missing files...")
    for index, row in tqdm(df.iterrows(), total=len(df)):
        tweet_id = f"tweet_{index}"
        image_url = str(row["image_url"]).strip()

        image_extension = image_url.split(".")[-1].split("?")[0]
        if len(image_extension) > 4 or not image_extension:
            image_extension = "jpg"

        local_filename = f"{tweet_id}.{image_extension}"
        local_path = IMAGE_DIR / local_filename

        if download_image(image_url, local_path):
            row_data = row.to_dict()
            row_data["id"] = tweet_id
            row_data["local_image_path"] = str(local_path)
            valid_rows.append(row_data)

    final_df = pd.DataFrame(valid_rows)
    final_df.to_csv(DOWNLOADED_CSV, index=False, encoding="utf-8-sig")
    print(f"\nDone. Valid images available: {len(valid_rows)}")
    print(f"Manifest saved at: {DOWNLOADED_CSV}")


if __name__ == "__main__":
    main()
