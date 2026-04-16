import pandas as pd

for fname in [
    "extract_clip_features_result_metadata.csv",
    "filter_downloaded_ok_result.csv",
]:
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
