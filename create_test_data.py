import csv
import os
import random

# Create a minimal test with just data we have
csv_path = "twitter_clip_ready.csv"
output_path = "twitter_clip_ready_downloaded_test.csv"

with open(csv_path, "r", encoding="utf-8-sig") as f:
    reader = csv.DictReader(f)
    fieldnames = reader.fieldnames
    rows = list(reader)

# Mark first 50 rows as failed (to test), rest as needs download
test_rows = []
for idx, row in enumerate(rows[:50]):
    row["local_image_path"] = f"twitter_images_test/row_{idx+1}.jpg"
    row["download_ok"] = 0
    row["download_status"] = "test_stub"
    test_rows.append(row)

fieldnames_out = list(fieldnames) if fieldnames else []
if "local_image_path" not in fieldnames_out:
    fieldnames_out.append("local_image_path")
if "download_ok" not in fieldnames_out:
    fieldnames_out.append("download_ok")
if "download_status" not in fieldnames_out:
    fieldnames_out.append("download_status")

with open(output_path, "w", encoding="utf-8-sig", newline="") as f:
    w = csv.DictWriter(f, fieldnames_out)
    w.writeheader()
    w.writerows(test_rows)

print(f"Created {output_path} with {len(test_rows)} test rows")
