import pandas as pd

INPUT_CSV = r"D:\ARGGGG\Semester 6\DATA MINING\Pheme_CLIP\pheme_clip_ready_downloaded.csv"
OUTPUT_CSV = r"D:\ARGGGG\Semester 6\DATA MINING\Pheme_CLIP\pheme_clip_final.csv"

df = pd.read_csv(INPUT_CSV)

df = df[df["download_ok"] == 1].copy()
df = df[df["local_image_path"].notna()].copy()
df = df[df["local_image_path"].astype(str).str.strip() != ""].copy()

df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")

print("Da tao file:", OUTPUT_CSV)
print("So dong giu lai:", len(df))