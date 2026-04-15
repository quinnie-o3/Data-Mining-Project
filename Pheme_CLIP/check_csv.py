import pandas as pd

for fname in ["clip_metadata.csv", "pheme_clip_final.csv"]:
    try:
        print("\n" + "=" * 20, fname, "=" * 20)
        df = pd.read_csv(fname)
        print("Columns:", df.columns.tolist())
        print("Shape:", df.shape)
        print(df.head(2))
        print("Has label:", "label" in df.columns)
    except Exception as e:
        print("\n" + "=" * 20, fname, "=" * 20)
        print("ERROR:", e)