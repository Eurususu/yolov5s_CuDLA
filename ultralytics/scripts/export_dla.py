# Export an Ultralytics detection model to ONNX with RAW per-level head
# outputs for DLA deployment (the counterpart of yolov5_dla's `qat.py export
# --noanchor`): outputs s8/s16/s32 = [batch, 4*reg_max + nc, H, W] each, where
# channels are concat(box-distribution [4*reg_max], cls [nc]). The DFL +
# dist2bbox decode happens on-device in C++ (see docs/ultralytics-support
# design doc in the cuDLA-samples repo).
#
# Usage (from the ultralytics repo root, on a GPU machine):
#   python scripts/export_dla.py --weights weights/yolov8s.pt --size 672 \
#       --save ../data/model/yolov8s_raw_672.onnx [--dynamic] [--device cuda:0]
#   --size accepts "672" (square) or "HxW" e.g. "736x1280" (rectangular).
#
# For end2end models (yolo26), the ONE2MANY head (self.cv2/self.cv3) is
# exported — the one2one head is skipped, so on-device results come from the
# NMS-based decode path (slightly below the model's e2e accuracy ceiling).

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.abspath("."))  # run from the ultralytics repo root

import torch

from ultralytics import YOLO
from ultralytics.nn.modules.head import Detect


def raw_forward(self, x):
    """Per-level raw head outputs: concat(box [4*reg_max], cls [nc]) per level."""
    outs = []
    for i in range(self.nl):
        box = self.cv2[i](x[i])   # [b, 4*reg_max, h, w]
        cls = self.cv3[i](x[i])   # [b, nc, h, w]
        outs.append(torch.cat((box, cls), 1))
    return outs


def main():
    parser = argparse.ArgumentParser(prog="export_dla.py")
    parser.add_argument("--weights", type=str, required=True)
    parser.add_argument("--size", type=str, default="672", help="int for square or HxW e.g. 736x1280")
    parser.add_argument("--save", type=str, default=None)
    parser.add_argument("--dynamic", action="store_true", help="dynamic batch axis")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--opset", type=int, default=13)
    args = parser.parse_args()

    if "x" in args.size.lower():
        h, w = (int(v) for v in args.size.lower().split("x"))
    else:
        h = w = int(args.size)
    save = args.save or (Path(args.weights).stem + f"_raw_{h}x{w}.onnx")

    model = YOLO(args.weights)
    net = model.model.to(args.device).float().eval()
    if hasattr(net, "fuse"):
        net.fuse()

    head = net.model[-1]
    assert isinstance(head, Detect), f"unsupported head type: {type(head).__name__}"
    nc, reg_max, nl, strides = head.nc, head.reg_max, head.nl, head.stride.tolist()
    print(f"head: nc={nc} reg_max={reg_max} nl={nl} strides={strides} "
          f"channels/level={4*reg_max + nc}")

    Detect.forward = raw_forward  # emit raw per-level outputs during export

    dummy = torch.zeros(1, 3, h, w, device=args.device)
    with torch.no_grad():
        torch.onnx.export(
            net, dummy, str(save), opset_version=args.opset,
            input_names=["images"], output_names=["s8", "s16", "s32"],
            dynamic_axes={"images": {0: "batch"}, "s8": {0: "batch"},
                          "s16": {0: "batch"}, "s32": {0: "batch"}} if args.dynamic else None,
        )
    print(f"Save onnx to {save}")


if __name__ == "__main__":
    main()
