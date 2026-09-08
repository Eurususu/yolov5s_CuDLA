# QAT toolkit for Ultralytics models on DLA (the counterpart of yolov5_dla's
# scripts/qat.py, adapted to the ultralytics module set — Option#1 quantization:
# Conv inputs/weights get quantizers, residual adds get QuantAdd; concat/mul
# scales are left to qdq_translator's --infer_concat_scales/--infer_mul_scales).
#
# Quantization = pytorch-quantization explicit Q/DQ (same as the yolov5 line):
#   quantize: load weights -> replace modules -> calibrate (histogram, mse amax)
#             -> optional QAT finetune (label-free MSE distillation against a
#             frozen FP copy) -> save ptq.pt / qat.pt (whole model object)
#   Export the QAT checkpoint with export_dla.py --qat (raw head + chunk split
#   + Q/DQ emission), then qdq_translator -> trtexec INT8 on the Jetson.
#
# Calibration/finetune use a label-free image-folder loader (letterbox to the
# network size, /255) — MSE distillation needs no labels, so a local val split
# is enough to drive the flow; run the full mAP-selected finetune on the server
# with train data.
#
# Usage (from the ultralytics repo root):
#   python scripts/qat_dla.py quantize weights/yolov8s.pt \
#       --calib-dir /media/data/jia/coco/images/val2017 --imgsz 672 \
#       --ptq ptq.pt [--qat qat.pt --epochs 1 --iters 20] [--device cuda:0]

import argparse
import os
import sys
from copy import deepcopy
from types import MethodType

sys.path.insert(0, os.path.abspath("."))

import torch
import torch.optim as optim
from tqdm import tqdm

from pytorch_quantization import nn as quant_nn
from pytorch_quantization import calib
from pytorch_quantization.tensor_quant import QuantDescriptor

import cv2
from ultralytics import YOLO


# --- module replacement (Option#1) ---------------------------------------------------

class QuantAdd(torch.nn.Module):
    """Quantized residual add: both inputs share ONE quantizer (INT8 add needs
    aligned scales)."""

    def __init__(self):
        super().__init__()
        self._input_quantizer = quant_nn.TensorQuantizer(QuantDescriptor(num_bits=8, calib_method="histogram"))
        self._input_quantizer._calibrator._torch_hist = True

    def forward(self, x, y):
        return self._input_quantizer(x) + self._input_quantizer(y)


def _transfer(src: torch.nn.Module, dst_cls) -> torch.nn.Module:
    """Swap a module's class keeping its state (weights/BN/hyperparams intact),
    then attach the quantizers. init_quantizer() refuses callers outside the
    module __init__ (frame-name check), so the init runs inside a function
    literally named __init__ — the same trick yolov5_dla uses."""
    from pytorch_quantization.nn.modules import _utils as quant_nn_utils

    dst = dst_cls.__new__(dst_cls)
    for k, v in vars(src).items():
        setattr(dst, k, v)

    def __init__(self):
        quant_desc_input, quant_desc_weight = quant_nn_utils.pop_quant_desc_in_kwargs(dst_cls)
        self.init_quantizer(quant_desc_input, quant_desc_weight)
        for q in (getattr(self, "_input_quantizer", None), getattr(self, "_weight_quantizer", None)):
            if q is not None and isinstance(q._calibrator, calib.HistogramCalibrator):
                q._calibrator._torch_hist = True

    __init__(dst)
    return dst


def replace_to_quantization_module(model) -> int:
    """nn.Conv2d -> QuantConv2d everywhere (the inner conv of ultralytics Conv
    wrappers included — the recursive walk visits children)."""
    count = 0

    def rec(module):
        nonlocal count
        for name in module._modules:
            sub = module._modules[name]
            rec(sub)
            if type(sub) is torch.nn.Conv2d:
                module._modules[name] = _transfer(sub, quant_nn.QuantConv2d)
                count += 1

    rec(model)
    return count


def bottleneck_forward_quant(self, x):
    if hasattr(self, "addop"):
        return self.addop(x, self.cv2(self.cv1(x))) if self.add else self.cv2(self.cv1(x))
    return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


def replace_bottleneck_forward(model) -> int:
    from ultralytics.nn.modules.block import Bottleneck

    n = 0
    for name, m in model.named_modules():
        if isinstance(m, Bottleneck) and m.add and not hasattr(m, "addop"):
            m.addop = QuantAdd()  # instance attr (picklable via __main__ injection on load)
            n += 1
    # CLASS-level forward patch (same as yolov5_dla): a bound instance-level
    # forward would be pickled into the checkpoint and break torch.load elsewhere
    Bottleneck.forward = bottleneck_forward_quant
    return n


def have_quantizer(module) -> bool:
    return any(isinstance(m, quant_nn.TensorQuantizer) for m in module.modules())


# --- label-free calibration / finetune data -------------------------------------------

class ImageFolderBatches:
    """Letterboxed /255 batches from a folder of images (no labels needed —
    calibration statistics and MSE distillation are label-free)."""

    def __init__(self, folder, size, batch_size=10, limit=500, letterbox=True):
        self.files = sorted(
            os.path.join(folder, f) for f in os.listdir(folder) if f.lower().endswith((".jpg", ".jpeg", ".png"))
        )[:limit]
        h, w = (size, size) if isinstance(size, int) else size
        self.h, self.w, self.bs, self.letterbox = h, w, batch_size, letterbox

    def __len__(self):
        return (len(self.files) + self.bs - 1) // self.bs

    def __iter__(self):
        buf = []
        for f in self.files:
            im = cv2.cvtColor(cv2.imread(f), cv2.COLOR_BGR2RGB)
            if self.letterbox:
                s = min(self.w / im.shape[1], self.h / im.shape[0])
                nh, nw = int(im.shape[0] * s + 0.5), int(im.shape[1] * s + 0.5)
                im = cv2.resize(im, (nw, nh))
                canvas = np_full((self.h, self.w, 3), 114, dtype="uint8")
                canvas[(self.h - nh) // 2 : (self.h - nh) // 2 + nh,
                       (self.w - nw) // 2 : (self.w - nw) // 2 + nw] = im
                im = canvas
            else:
                im = cv2.resize(im, (self.w, self.h))
            buf.append(torch.from_numpy(im).permute(2, 0, 1).float() / 255.0)
            if len(buf) == self.bs:
                yield torch.stack(buf)
                buf = []
        if buf:
            yield torch.stack(buf)


def np_full(shape, value, dtype):
    import numpy as np

    return np.full(shape, value, dtype=dtype)


# --- calibrate / finetune (ported from yolov5_dla/quantization/quantize.py) -----------

def calibrate_model(model, loader, device, num_batch=25):
    for _, m in model.named_modules():
        if isinstance(m, quant_nn.TensorQuantizer) and m._calibrator is not None:
            m.disable_quant()
            m.enable_calib()
    with torch.no_grad():
        for i, imgs in tqdm(enumerate(loader), total=num_batch, desc="calibrating"):
            model(imgs.to(device))
            if i >= num_batch:
                break
    for _, m in model.named_modules():
        if isinstance(m, quant_nn.TensorQuantizer) and m._calibrator is not None:
            m.enable_quant()
            m.disable_calib()
    for _, m in model.named_modules():
        if isinstance(m, quant_nn.TensorQuantizer) and m._calibrator is not None:
            if isinstance(m._calibrator, calib.MaxCalibrator):  # weight quantizers keep the max default
                m.load_calib_amax()
            else:
                m.load_calib_amax(method="mse")
    print("calibration done (input: histogram+mse, weight: max)")


def finetune(model, loader, device, nepochs=10, iters_per_epoch=1000, lr=1e-5):
    """Label-free QAT: MSE-distil the quantized model's layer outputs against a
    frozen FP copy (identical objective to the yolov5_dla line)."""
    teacher = deepcopy(model).eval()
    for _, m in teacher.named_modules():
        if isinstance(m, quant_nn.TensorQuantizer):
            m._disabled = True

    model.train()
    model.requires_grad_(True)
    optimizer = optim.Adam(model.parameters(), lr)
    lossfn = torch.nn.MSELoss()

    pairs = [(m, t) for (n, m), (_, t) in zip(model.named_modules(), teacher.named_modules())
             if not isinstance(m, quant_nn.TensorQuantizer) and len(list(m.children())) == 0]

    def hook(buf):
        def fn(module, inp, out):
            buf.append(out)
        return fn

    for iepoch in range(nepochs):
        for imgs in tqdm(loader, total=iters_per_epoch, desc=f"QAT {iepoch + 1}/{nepochs}"):
            outs_q, outs_fp, handles = [], [], []
            for m, t in pairs[::3]:  # supervise every 3rd leaf module (see yolov5_dla notes)
                handles.append(m.register_forward_hook(hook(outs_q)))
                handles.append(t.register_forward_hook(hook(outs_fp)))
            with torch.no_grad():
                teacher(imgs.to(device))
            model(imgs.to(device))
            loss = sum(lossfn(a, b) for a, b in zip(outs_q, outs_fp) if a.shape == b.shape)
            for h in handles:
                h.remove()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            print(f"  loss {loss.item():.5f}")
    print("QAT finetune done")


def main():
    parser = argparse.ArgumentParser(prog="qat_dla.py")
    sub = parser.add_subparsers(dest="cmd", required=True)
    q = sub.add_parser("quantize")
    q.add_argument("weight", type=str)
    q.add_argument("--calib-dir", type=str, required=True, help="folder of images for calibration/finetune")
    q.add_argument("--imgsz", type=str, default="672")
    q.add_argument("--batch-size", type=int, default=10)
    q.add_argument("--limit", type=int, default=500, help="max images used")
    q.add_argument("--calib-batches", type=int, default=25)
    q.add_argument("--ptq", type=str, default=None)
    q.add_argument("--qat", type=str, default=None)
    q.add_argument("--epochs", type=int, default=10)
    q.add_argument("--iters", type=int, default=1000, help="batches per epoch cap (small for dry runs)")
    q.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()

    size = tuple(int(v) for v in args.imgsz.lower().split("x")) if "x" in args.imgsz.lower() else int(args.imgsz)
    device = torch.device(args.device)

    net = YOLO(args.weight).model.to(device).float().eval()
    if hasattr(net, "fuse"):
        net.fuse()

    n_conv = replace_to_quantization_module(net)
    n_add = replace_bottleneck_forward(net)
    print(f"quantized {n_conv} convs, {n_add} residual adds (QuantAdd)")

    loader = ImageFolderBatches(args.calib_dir, size, args.batch_size, args.limit)
    print(f"calibration data: {len(loader.files)} images @ {size}")
    calibrate_model(net, loader, device, num_batch=args.calib_batches)

    if args.ptq:
        torch.save({"model": net}, args.ptq)
        print(f"saved PTQ model to {args.ptq}")

    if args.qat:
        finetune(net, loader, device, nepochs=args.epochs, iters_per_epoch=args.iters)
        torch.save({"model": net}, args.qat)
        print(f"saved QAT model to {args.qat}")


if __name__ == "__main__":
    main()
