import os
import cv2
import numpy as np
from tqdm import tqdm
import time
from collections import defaultdict
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical

IMG_SIZE = 64  # Standard input size for all models


def load_dataset(data_dir):
    """
    Loads and resizes images. Tracks per-class count and preprocessing time.

    Returns:
        images: np.array of shape (N, IMG_SIZE, IMG_SIZE, 3)
        labels: np.array of string class labels
        class_names: sorted list of class names
        class_times: dict of time taken per class
        class_counts: dict of sample count per class
    """
    images = []
    labels = []
    class_times = {}
    class_counts = defaultdict(int)
    classes = sorted(os.listdir(data_dir))

    for label in classes:
        class_dir = os.path.join(data_dir, label)
        if not os.path.isdir(class_dir):
            continue

        start = time.time()
        count = 0

        for img_file in tqdm(os.listdir(class_dir), desc=f'Loading {label}'):
            try:
                img_path = os.path.join(class_dir, img_file)
                img = cv2.imread(img_path)
                img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                images.append(img)
                labels.append(label)
                count += 1
            except:
                continue

        class_times[label] = round(time.time() - start, 2)
        class_counts[label] = count

    return np.array(images), np.array(labels), classes, class_times, class_counts


def preprocess_images(X, y, label_encoder=None):
    """
    Normalizes image pixel values and encodes labels to one-hot.

    Args:
        X: np.array of images
        y: list of class labels
        label_encoder: optional LabelEncoder for reuse

    Returns:
        X_norm: normalized images
        y_cat: one-hot encoded labels
        label_encoder: fitted or reused LabelEncoder
    """
    X = X.astype('float32') / 255.0

    if label_encoder is None:
        label_encoder = LabelEncoder()
        y_encoded = label_encoder.fit_transform(y)
    else:
        y_encoded = label_encoder.transform(y)

    y_cat = to_categorical(y_encoded)
    return X, y_cat, label_encoder


def split_data(X, y_cat, test_size=0.3, seed=42):
    """
    Splits dataset into Train, Validation, Test (70/15/15).

    Returns:
        X_train, y_train, X_val, y_val, X_test, y_test
    """
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y_cat, test_size=test_size, stratify=y_cat, random_state=seed
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=seed
    )
    return X_train, y_train, X_val, y_val, X_test, y_test
