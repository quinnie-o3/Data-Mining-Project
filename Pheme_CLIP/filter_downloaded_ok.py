import pandas as pd

INPUT_CSV = r"D:\ARGGGG\Semester 6\DATA MINING\Pheme_CLIP\download_images_result.csv"
OUTPUT_CSV = r"D:\ARGGGG\Semester 6\DATA MINING\Pheme_CLIP\filter_downloaded_ok_result.csv"

df = pd.read_csv(INPUT_CSV)

df = df[df["download_ok"] == 1].copy()
df = df[df["local_image_path"].notna()].copy()
df = df[df["local_image_path"].astype(str).str.strip() != ""].copy()

df = df.rename(
    columns={
        "tweet_text": "tweetText",
        "tweet_text_clean": "tweetText_clean",
    }
)

if "link_status" not in df.columns:
    if "download_status" in df.columns:
        df["link_status"] = df["download_status"]
    else:
        df["link_status"] = ""

if "tweetText_clean" not in df.columns:
    df["tweetText_clean"] = ""

for col in ["tweetText", "label", "image_link"]:
    if col not in df.columns:
        df[col] = ""

df = df[
    [
        "tweetText_clean",
        "tweetText",
        "label",
        "image_link",
        "link_status",
    ]
].copy()

df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")

print("Da tao file:", OUTPUT_CSV)
print("So dong giu lai:", len(df))
