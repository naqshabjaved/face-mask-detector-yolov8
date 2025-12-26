import os
import random
from shutil import copyfile

IMG_DIR = "data/processed/images"
LBL_DIR = "data/processed/labels"
OUT_IMG_DIR = "data/processed/images"
OUT_LBL_DIR = "data/processed/labels"

TRAIN_RATIO = 0.7
VAL_RATIO = 0.15

images = [f for f in os.listdir(IMG_DIR) if f.endswith((".png", ".jpg"))]
random.shuffle(images)

total = len(images)
train_end = int(total * TRAIN_RATIO)
val_end = int(total * (TRAIN_RATIO + VAL_RATIO))

splits = {
    "train": images[:train_end],
    "val": images[train_end:val_end],
    "test": images[val_end:]
}

for split, files in splits.items():
    img_out = os.path.join(OUT_IMG_DIR, split)
    lbl_out = os.path.join(OUT_LBL_DIR, split)

    os.makedirs(img_out, exist_ok=True)
    os.makedirs(lbl_out, exist_ok=True)

    for img in files:
        lbl = os.path.splitext(img)[0] + ".txt"

        copyfile(
            os.path.join(IMG_DIR, img),
            os.path.join(img_out, img)
        )

        copyfile(
            os.path.join(LBL_DIR, lbl),
            os.path.join(lbl_out, lbl)
        )

print("Dataset split completed:")
for k, v in splits.items():
    print(f"{k}: {len(v)} images")
