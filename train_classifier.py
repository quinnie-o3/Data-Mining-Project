import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)

# Đường dẫn file
METADATA_PATH = r"D:\ARGGGG\Semester 6\DATA MINING\Pheme_CLIP\clip_metadata.csv"
TEXT_FEATURES_PATH = r"D:\ARGGGG\Semester 6\DATA MINING\Pheme_CLIP\clip_text_features.npy"
IMAGE_FEATURES_PATH = r"D:\ARGGGG\Semester 6\DATA MINING\Pheme_CLIP\clip_image_features.npy"
RESULTS_TXT_PATH = r"D:\ARGGGG\Semester 6\DATA MINING\Pheme_CLIP\classifier_results.txt"
RESULTS_CSV_PATH = r"D:\ARGGGG\Semester 6\DATA MINING\Pheme_CLIP\classifier_results.csv"

def load_data():
    """Load metadata và features"""
    # Load metadata
    metadata = pd.read_csv(METADATA_PATH)
    labels = metadata['label'].map({'fake': 1, 'real': 0}).values

    # Load features
    text_features = np.load(TEXT_FEATURES_PATH)
    image_features = np.load(IMAGE_FEATURES_PATH)

    # Kiểm tra shape
    num_samples = len(metadata)
    if text_features.shape[0] != num_samples or image_features.shape[0] != num_samples:
        raise ValueError(f"Shape không khớp: metadata {num_samples}, text {text_features.shape[0]}, image {image_features.shape[0]}")

    return metadata, labels, text_features, image_features

def create_experiments(text_features, image_features):
    """Tạo 3 thí nghiệm"""
    experiments = {
        'text-only': text_features,
        'image-only': image_features,
        'multimodal': np.concatenate([text_features, image_features], axis=1)
    }
    return experiments

def train_and_evaluate(X, y, experiment_name):
    """Train và đánh giá classifier"""
    # Chia train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Tạo pipeline với scaler và logistic regression
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', LogisticRegression(max_iter=2000, random_state=42))
    ])

    # Train
    pipeline.fit(X_train, y_train)

    # Predict
    y_pred = pipeline.predict(X_test)

    # Tính metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='binary')
    recall = recall_score(y_test, y_pred, average='binary')
    f1 = f1_score(y_test, y_pred, average='binary')
    report = classification_report(y_test, y_pred, target_names=['real', 'fake'])
    conf_matrix = confusion_matrix(y_test, y_pred)

    return {
        'experiment': experiment_name,
        'shape': X.shape,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'classification_report': report,
        'confusion_matrix': conf_matrix
    }

def main():
    # Load data
    metadata, labels, text_features, image_features = load_data()
    print(f"Đã load {len(metadata)} samples")

    # Tạo experiments
    experiments = create_experiments(text_features, image_features)

    # Train và evaluate từng experiment
    results = []
    with open(RESULTS_TXT_PATH, 'w', encoding='utf-8') as f:
        for exp_name, X in experiments.items():
            print(f"\nĐang train {exp_name}...")
            result = train_and_evaluate(X, labels, exp_name)

            # In ra console và file
            print(f"Experiment: {exp_name}")
            print(f"Shape của X: {result['shape']}")
            print(f"Accuracy: {result['accuracy']:.4f}")
            print(f"Precision: {result['precision']:.4f}")
            print(f"Recall: {result['recall']:.4f}")
            print(f"F1-score: {result['f1']:.4f}")
            print("Classification Report:")
            print(result['classification_report'])
            print("Confusion Matrix:")
            print(result['confusion_matrix'])
            print("-" * 50)

            # Ghi vào file
            f.write(f"Experiment: {exp_name}\n")
            f.write(f"Shape của X: {result['shape']}\n")
            f.write(f"Accuracy: {result['accuracy']:.4f}\n")
            f.write(f"Precision: {result['precision']:.4f}\n")
            f.write(f"Recall: {result['recall']:.4f}\n")
            f.write(f"F1-score: {result['f1']:.4f}\n")
            f.write("Classification Report:\n")
            f.write(result['classification_report'] + "\n")
            f.write("Confusion Matrix:\n")
            f.write(str(result['confusion_matrix']) + "\n")
            f.write("-" * 50 + "\n")

            # Lưu cho bảng tổng kết
            results.append({
                'Experiment': exp_name,
                'Accuracy': result['accuracy'],
                'Precision': result['precision'],
                'Recall': result['recall'],
                'F1-score': result['f1']
            })

    # Tạo bảng tổng kết
    results_df = pd.DataFrame(results)
    results_df.to_csv(RESULTS_CSV_PATH, index=False)
    print(f"\nĐã lưu kết quả vào {RESULTS_TXT_PATH} và {RESULTS_CSV_PATH}")

if __name__ == "__main__":
    main()