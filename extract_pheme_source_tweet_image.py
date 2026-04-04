import os
import json
import csv
import re

ROOT_DIR = r"D:\ARGGGG\phemernrdataset"
OUTPUT_CSV = r"D:\ARGGGG\pheme_source_tweet_image_links.csv"

def clean_text(text):
    if not text:
        return ""
    text = text.replace("\n", " ").replace("\r", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text

def extract_tweet_text(tweet):
    return clean_text(tweet.get("full_text") or tweet.get("text") or "")

def extract_image_links(tweet):
    links = []

    media = tweet.get("extended_entities", {}).get("media", [])
    if not media:
        media = tweet.get("entities", {}).get("media", [])

    for m in media:
        if m.get("type") == "photo":
            url = m.get("media_url_https") or m.get("media_url")
            if url and url not in links:
                links.append(url)

    return links

def get_label_from_path(path):
    path_lower = path.lower().replace("\\", "/")
    if "/non-rumours/" in path_lower or "/non-rumors/" in path_lower:
        return "real"
    elif "/rumours/" in path_lower or "/rumors/" in path_lower:
        return "fake"
    return ""

def get_event_from_path(path):
    parts = path.replace("\\", "/").split("/")
    keywords = {"rumours", "rumors", "non-rumours", "non-rumors"}
    for i, p in enumerate(parts):
        if p.lower() in keywords and i > 0:
            return parts[i - 1]
    return ""

rows = []
total_json = 0
has_image_count = 0
no_image_count = 0
error_count = 0

for root, dirs, files in os.walk(ROOT_DIR):
    root_norm = root.replace("\\", "/").lower()

    # chỉ lấy source-tweet
    if "source-tweet" not in root_norm:
        continue

    for file in files:
        if not file.endswith(".json"):
            continue

        total_json += 1
        file_path = os.path.join(root, file)

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                tweet = json.load(f)

            tweet_id = tweet.get("id_str") or str(tweet.get("id", ""))
            tweet_text = extract_tweet_text(tweet)
            image_links = extract_image_links(tweet)
            label = get_label_from_path(file_path)
            event = get_event_from_path(file_path)

            if image_links:
                has_image_count += 1
                for img in image_links:
                    rows.append({
                        "id": tweet_id,
                        "event": event,
                        "label": label,
                        "tweet_text": tweet_text,
                        "image_link": img,
                        "has_image": 1,
                        "json_path": file_path
                    })
            else:
                no_image_count += 1
                rows.append({
                    "id": tweet_id,
                    "event": event,
                    "label": label,
                    "tweet_text": tweet_text,
                    "image_link": "",
                    "has_image": 0,
                    "json_path": file_path
                })

        except Exception as e:
            error_count += 1
            print(f"Loi doc file: {file_path}")
            print("Chi tiet loi:", e)

with open(OUTPUT_CSV, "w", newline="", encoding="utf-8-sig") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=["id", "event", "label", "tweet_text", "image_link", "has_image", "json_path"]
    )
    writer.writeheader()
    writer.writerows(rows)

print("=" * 50)
print("XUAT FILE THANH CONG")
print("Tong file json da doc:", total_json)
print("So tweet co anh:", has_image_count)
print("So tweet khong co anh:", no_image_count)
print("So file loi:", error_count)
print("File CSV:", OUTPUT_CSV)
print("=" * 50)