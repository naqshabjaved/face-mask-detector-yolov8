import os
import xml.etree.ElementTree as ET
from tqdm import tqdm
from shutil import copyfile

CLASS_MAP = {
    "with_mask": 0,
    "without_mask": 1,
    "mask_weared_incorrect": 2
}

RAW_IMG_DIR = "data/raw/kaggle/images"
RAW_ANN_DIR = "data/raw/kaggle/annotations"
OUT_IMG_DIR = "data/processed/images"
OUT_LBL_DIR = "data/processed/labels"

os.makedirs(OUT_IMG_DIR, exist_ok=True)
os.makedirs(OUT_LBL_DIR, exist_ok=True)

def convert_bbox(size, box):
    dw = 1.0 / size[0]
    dh = 1.0 / size[1]
    x = (box[0] + box[1]) / 2.0
    y = (box[2] + box[3]) / 2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    return x * dw, y * dh, w * dw, h * dh

for xml_file in tqdm(os.listdir(RAW_ANN_DIR)):
    if not xml_file.endswith(".xml"):
        continue

    xml_path = os.path.join(RAW_ANN_DIR, xml_file)
    tree = ET.parse(xml_path)
    root = tree.getroot()

    filename = root.find("filename").text
    img_src = os.path.join(RAW_IMG_DIR, filename)
    img_dst = os.path.join(OUT_IMG_DIR, filename)

    size = root.find("size")
    w = int(size.find("width").text)
    h = int(size.find("height").text)

    yolo_lines = []

    for obj in root.findall("object"):
        cls = obj.find("name").text
        if cls not in CLASS_MAP:
            continue

        cls_id = CLASS_MAP[cls]
        bnd = obj.find("bndbox")

        box = (
            float(bnd.find("xmin").text),
            float(bnd.find("xmax").text),
            float(bnd.find("ymin").text),
            float(bnd.find("ymax").text),
        )

        bb = convert_bbox((w, h), box)
        yolo_lines.append(f"{cls_id} {' '.join(map(str, bb))}")

    if not yolo_lines:
        continue

    copyfile(img_src, img_dst)

    label_path = os.path.join(
        OUT_LBL_DIR, os.path.splitext(filename)[0] + ".txt"
    )
    with open(label_path, "w") as f:
        f.write("\n".join(yolo_lines))

print("VOC to YOLO conversion completed.")
