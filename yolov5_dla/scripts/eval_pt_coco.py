# Evaluate .pt checkpoints (FP32 best.pt and/or fake-quantized qat.pt) with BOTH
# yolov5-native mAP and pycocotools COCOeval — the apples-to-apples comparison
# against the on-device (Jetson/DLA) validation: same GT json (make_coco_json.py),
# same pycocotools metrics, same conf/iou as the C++ app's validation path.
#
# Run on the GPU SERVER from the yolov5_dla root (needs torch):
#   python scripts/eval_pt_coco.py \
#       --weights runs/train/3classes/weights/best.pt qat.pt \
#       --cocodir /root/dataset/3classes --data data/3classes.yaml --imgsz 672 \
#       --val-list origin_val.txt ship_add_val.txt ship_blank_add_val.txt
#
# Requires the COCO-format GT json at <yaml path>/annotations/instances_val2017.json:
#   python scripts/make_coco_json.py --cocodir <dir> --data <yaml> --val-list <same lists>
#
# Notes:
# - qat.pt is loaded AS PICKLED (Quant* modules intact → fake-quant active). Do NOT
#   evaluate it through load_yolov5s_model: that rebuilds a plain DetectionModel and
#   silently drops the quantizers.
# - Per qat.py's convention the Detect head's (model.model[24]) quantizers are
#   disabled during eval — ONNX export prunes the head's decode part.
# - --conf/--iou default to 0.001/0.65, matching validate_coco.cpp's validation path,
#   so numbers line up with the on-device run.

import sys
import os
import argparse
from contextlib import nullcontext
from pathlib import Path

sys.path.insert(0, os.path.abspath("."))

import torch

import quantization.quantize as quantize
from scripts.qat import create_coco_val_dataloader, evaluate_coco
from utils.general import check_dataset


def load_checkpoint(weight, device):
    model = torch.load(weight, map_location=device, weights_only=False)["model"]
    model = model.float().to(device)
    model.eval()
    if not quantize.have_quantizer(model):  # fuse Conv+BN only on FP32 checkpoints
        with torch.no_grad():
            model.fuse()
    return model


def main():
    parser = argparse.ArgumentParser(prog="eval_pt_coco.py")
    parser.add_argument("--weights", type=str, nargs="+", required=True,
                        help="checkpoint(s): FP32 best.pt and/or fake-quantized qat.pt")
    parser.add_argument("--cocodir", type=str, required=True)
    parser.add_argument("--data", type=str, required=True, help="dataset yaml (nc/names; its path/ needs annotations/instances_val2017.json)")
    parser.add_argument("--val-list", type=str, nargs="+", default=["val2017.txt"])
    parser.add_argument("--imgsz", type=int, default=672)
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--conf", type=float, default=0.001)   # matches the C++ validation
    parser.add_argument("--iou", type=float, default=0.65)     # matches the C++ validation
    args = parser.parse_args()

    data = check_dataset(args.data)
    gt_json = Path(data.get("path", "")) / "annotations" / "instances_val2017.json"
    if not gt_json.is_file():
        sys.exit(f"GT json not found: {gt_json}\nGenerate it first:\n"
                 f"  python scripts/make_coco_json.py --cocodir {args.cocodir} --data {args.data} "
                 f"--val-list {' '.join(args.val_list)}")

    device = torch.device(args.device)
    loader = create_coco_val_dataloader(args.cocodir, batch_size=args.batch_size,
                                        imgsz=args.imgsz, val_list=args.val_list)

    for w in args.weights:
        model = load_checkpoint(w, device)
        quantized = quantize.have_quantizer(model)
        print(f"\n===== {w} ({'fake-quantized' if quantized else 'FP32'}) =====")
        save_dir = f"runs/eval/{Path(w).stem}"
        Path(save_dir).mkdir(parents=True, exist_ok=True)  # val.py writes <save_dir>/_predictions.json
        ctx = quantize.disable_quantization(model.model[24]) if quantized else nullcontext()
        with ctx:
            ap = evaluate_coco(model, loader, True, save_dir=save_dir,
                               conf_thres=args.conf, iou_thres=args.iou, data_yaml=args.data)
        print(f">>> {w}: yolov5-native mAP50-95 = {ap:.5f} (pycocotools table above)")
        del model
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
