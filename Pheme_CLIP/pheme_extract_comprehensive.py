#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script to extract tweet data from PHEME dataset JSON files.
Recursively scans source-tweet folders and extracts tweet info, images, and labels.
"""

import os
import json
import csv
import re
from pathlib import Path
from typing import Optional, List, Dict, Tuple

class PHEMEExtractor:
    def __init__(self, dataset_root: str, output_csv: str):
        self.dataset_root = dataset_root
        self.output_csv = output_csv
        self.total_json_files = 0
        self.tweets_with_images = 0
        self.tweets_without_images = 0
        self.error_files = 0
        self.rows = []

    def clean_tweet_text(self, text: str) -> str:
        """
        Clean tweet text: replace newlines with space, collapse multiple spaces.
        """
        if not text:
            return ""
        # Replace newlines, tabs with spaces
        text = re.sub(r'[\n\r\t]+', ' ', text)
        # Collapse multiple spaces into one
        text = re.sub(r' +', ' ', text)
        # Strip leading/trailing spaces
        return text.strip()

    def extract_images(self, json_data: Dict) -> List[str]:
        """
        Extract image URLs from tweet JSON.
        Priority: extended_entities.media > entities.media
        Only extract photo type media.
        """
        images = []
        
        # Try extended_entities.media first
        if 'extended_entities' in json_data and 'media' in json_data['extended_entities']:
            media_list = json_data['extended_entities']['media']
            for media in media_list:
                if media.get('type') == 'photo':
                    url = media.get('media_url_https') or media.get('media_url')
                    if url:
                        images.append(url)
        
        # If no images from extended_entities, try entities.media
        if not images and 'entities' in json_data and 'media' in json_data['entities']:
            media_list = json_data['entities']['media']
            for media in media_list:
                if media.get('type') == 'photo':
                    url = media.get('media_url_https') or media.get('media_url')
                    if url:
                        images.append(url)
        
        return images

    def determine_label_and_event(self, json_path: str) -> Tuple[Optional[str], Optional[str]]:
        """
        Determine label (real/fake) and event from the path.
        Path structure: .../event_name/rumours or non-rumours/source-tweet/...
        """
        path_parts = json_path.lower().replace('\\', '/').split('/')
        
        label = None
        event = None
        
        # Find rumours/non-rumours and get event name
        for i, part in enumerate(path_parts):
            if part in ['rumours', 'rumors']:
                label = 'fake'
                if i > 0:
                    event = path_parts[i - 1]
                break
            elif part in ['non-rumours', 'non-rumors']:
                label = 'real'
                if i > 0:
                    event = path_parts[i - 1]
                break
        
        return label, event

    def process_json_file(self, json_path: str) -> None:
        """
        Process a single JSON file and extract relevant data.
        """
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                json_data = json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            print(f"ERROR reading {json_path}: {e}")
            self.error_files += 1
            return
        
        self.total_json_files += 1
        
        # Extract basic fields
        tweet_id = json_data.get('id_str') or json_data.get('id')
        tweet_text = json_data.get('full_text') or json_data.get('text', '')
        tweet_text = self.clean_tweet_text(tweet_text)
        
        # Determine label and event
        label, event = self.determine_label_and_event(json_path)
        
        # Extract images
        images = self.extract_images(json_data)
        has_image = 1 if images else 0
        
        # Update statistics
        if has_image:
            self.tweets_with_images += 1
        else:
            self.tweets_without_images += 1
        
        # Create rows (one per image, or one empty if no images)
        if images:
            for image_url in images:
                row = {
                    'id': tweet_id,
                    'tweet_text': tweet_text,
                    'image_link': image_url,
                    'label': label,
                    'event': event,
                    'json_path': json_path,
                    'has_image': has_image
                }
                self.rows.append(row)
        else:
            row = {
                'id': tweet_id,
                'tweet_text': tweet_text,
                'image_link': '',
                'label': label,
                'event': event,
                'json_path': json_path,
                'has_image': has_image
            }
            self.rows.append(row)

    def scan_dataset(self) -> None:
        """
        Recursively scan dataset folder for JSON files in source-tweet directories.
        """
        for root, dirs, files in os.walk(self.dataset_root):
            # Check if current directory is 'source-tweet'
            if os.path.basename(root).lower() == 'source-tweet':
                for file in files:
                    if file.lower().endswith('.json'):
                        json_path = os.path.join(root, file)
                        self.process_json_file(json_path)

    def save_to_csv(self) -> None:
        """
        Save extracted data to CSV file with utf-8-sig encoding.
        """
        if not self.rows:
            print("No data to save.")
            return
        
        fieldnames = ['id', 'tweet_text', 'image_link', 'label', 'event', 'json_path', 'has_image']
        
        try:
            with open(self.output_csv, 'w', newline='', encoding='utf-8-sig') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(self.rows)
            print(f"✓ CSV file saved successfully: {self.output_csv}")
        except IOError as e:
            print(f"ERROR writing CSV: {e}")

    def print_statistics(self) -> None:
        """
        Print extraction statistics.
        """
        print("\n" + "="*50)
        print("EXTRACTION COMPLETE")
        print("="*50)
        print(f"Total JSON files processed: {self.total_json_files}")
        print(f"Tweets with images: {self.tweets_with_images}")
        print(f"Tweets without images: {self.tweets_without_images}")
        print(f"Error files: {self.error_files}")
        print(f"Output CSV: {self.output_csv}")
        print("="*50)

    def run(self) -> None:
        """
        Execute the full extraction pipeline.
        """
        print(f"Starting PHEME dataset extraction from: {self.dataset_root}")
        print(f"Output file: {self.output_csv}\n")
        
        self.scan_dataset()
        self.save_to_csv()
        self.print_statistics()


if __name__ == '__main__':
    # Configuration
    DATASET_ROOT = r'D:\ARGGGG\phemernrdataset'
    OUTPUT_CSV = r'D:\ARGGGG\pheme_source_tweet_image_links.csv'
    
    # Run extraction
    extractor = PHEMEExtractor(DATASET_ROOT, OUTPUT_CSV)
    extractor.run()
