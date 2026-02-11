import numpy as np
import cv2
from pathlib import Path
from tqdm import tqdm

MAIZE_CATEGORIES = ["0_NOR", "1_F&S", "2_SD", "3_MY", "4_AP", "5_BN", "6_HD", "7_IM"]
RICE_CATEGORIES = ["0_NOR", "1_F&S", "2_SD", "3_MY", "4_AP", "5_BN", "6_UN", "7_IM"]

CATEGORIES_MAP = {
    "maize": MAIZE_CATEGORIES,
    "rice": RICE_CATEGORIES
}


def collect_embeddings(grain_type, split="train"):
    X = []
    y = []

    categories = CATEGORIES_MAP.get(grain_type, [])

    total_images = sum(
        len(
            list(
                Path(f"./dataset/images/{grain_type}/{split}/{category}").glob("*.png")
            )
        )
        for category in categories
    )

    print(f"Collecting {total_images} {split} embeddings...")

    with tqdm(total=total_images, desc=f"Processing {split} images") as pbar:
        for category in categories:
            data_dir = Path(f"./dataset/images/{grain_type}/{split}/{category}")
            images = list(data_dir.glob("*.png"))

            for img_path in images:
                try:
                    embedding = embed_features(str(img_path))
                    X.append(embedding.flatten())
                    label = f"{grain_type}_{category}"
                    y.append(label)
                except Exception as e:
                    print(f"Error processing {img_path}: {e}")

                pbar.update(1)

    X = np.array(X)
    return X, y


def embed_features(img_path):
    img = cv2.imread(img_path)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    mask = (s > 30) & (v > 30)
    mask = mask.astype("uint8")

    hist = cv2.calcHist([hsv], [0, 1, 2], mask, [8, 8, 8], [0, 180, 0, 256, 0, 256])
    embedding = hist.flatten() / hist.sum()
    return embedding.reshape(1, -1)
