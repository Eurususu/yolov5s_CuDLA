# Generate a COCO-format GT json from YOLO-format txt lists, so that
# `scripts/qat.py ... --save-json` can run pycocotools COCOeval on a custom dataset.
#
# The output must match val.py's prediction encoding:
#   - image_id:    int(stem) if stem.isnumeric() else stem   (val.py save_one_json)
#   - category_id: identity (val.py class_map = list(range(1000)) for non-COCO data)
#   - bbox:        absolute pixel [x_topleft, y_topleft, w, h] in the ORIGINAL image
#
# Usage (from repo root):
#   python scripts/make_coco_json.py --cocodir /root/dataset/3classes \
#       --data data/3classes.yaml \
#       --val-list origin_val.txt ship_add_val.txt ship_blank_add_val.txt

import sys
import os
import json
import argparse
from pathlib import Path

# Add the current directory to PYTHONPATH for yolov5 imports
sys.path.insert(0, os.path.abspath("."))

from PIL import Image

from utils.dataloaders import img2label_paths  # same images/ -> labels/ swap as training/val
from utils.general import check_dataset


def yolo_to_coco(cocodir, lists, names, save):

    images, annotations = [], []
    ann_id = 1  # COCO annotation ids start at 1
    seen = set()

    for name in lists:
        with open(Path(cocodir) / name) as f:
            im_files = [line.strip() for line in f if line.strip()]

        for im_file, lb_file in zip(im_files, img2label_paths(im_files)):
            im_file, lb_file = Path(im_file), Path(lb_file)
            if im_file in seen:  # the same image may appear in several lists
                continue
            seen.add(im_file)

            # image id must match val.py save_one_json for non-COCO data: string stem
            # (COCOeval sorts imgIds, so a mix of int/str ids raises TypeError)
            image_id = im_file.stem

            with Image.open(im_file) as im:
                width, height = im.size
            images.append({"id": image_id, "width": width, "height": height, "file_name": im_file.name})

            if not lb_file.is_file():  # images without labels still need an entry (their preds count as FPs)
                continue
            with open(lb_file) as f:
                for line in f:
                    parts = line.split()
                    if len(parts) < 5:
                        continue
                    cls, cx, cy, w, h = int(parts[0]), *map(float, parts[1:5])
                    x, y = (cx - w / 2) * width, (cy - h / 2) * height  # top-left, absolute pixels
                    bw, bh = w * width, h * height
                    annotations.append({
                        "id": ann_id,
                        "image_id": image_id,
                        "category_id": cls,
                        "bbox": [round(x, 3), round(y, 3), round(bw, 3), round(bh, 3)],
                        "area": round(bw * bh, 3),
                        "iscrowd": 0,
                    })
                    ann_id += 1

    categories = [{"id": i, "name": n} for i, n in enumerate(names)]
    save = Path(save)
    save.parent.mkdir(parents=True, exist_ok=True)
    with open(save, "w") as f:
        json.dump({"images": images, "annotations": annotations, "categories": categories}, f)
    print(f"Saved {save}: {len(images)} images, {len(annotations)} annotations, {len(categories)} categories")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="make_coco_json.py")
    parser.add_argument("--cocodir", type=str, required=True, help="dataset directory containing the list files")
    parser.add_argument("--data", type=str, default="data/3classes.yaml", help="dataset yaml (for nc/names)")
    parser.add_argument("--val-list", type=str, nargs="+", required=True, help="val image list filename(s) inside cocodir")
    parser.add_argument("--save", type=str, default=None, help="output json (default <cocodir>/annotations/instances_val2017.json)")
    args = parser.parse_args()

    data = check_dataset(args.data)
    names = data["names"]
    if isinstance(names, dict):
        names = [names[i] for i in range(data["nc"])]

    yolo_to_coco(args.cocodir, args.val_list, names, args.save or Path(args.cocodir) / "annotations" / "instances_val2017.json")
