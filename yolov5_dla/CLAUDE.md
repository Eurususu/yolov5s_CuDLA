# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

A Chinese version lives at `CLAUDE.zh-CN.md`. **Keep both in sync: whenever this file is updated, update the Chinese version too** (and vice versa).

## What this repo is

Ultralytics YOLOv5 **v7.0** (HEAD is detached at the `v7.0` tag; `master` tracks upstream) with a local, **uncommitted** INT8 quantization layer (PTQ/QAT via NVIDIA `pytorch_quantization`) for TensorRT/DLA deployment — hence `yolov5_dla`. The fork's changes live only in the working tree:

- `quantization/` (new) — quantization toolkit
- `scripts/qat.py` (new) — CLI driver for the PTQ/QAT/export workflow
- `models/common.py` (modified) — `torch.cat` calls in `C3TR`, `C3`, `SPP`, `SPPF`, `Focus`, `GhostConv`, `Classify` replaced with a `Concat(1)` submodule so concat ops become quantizable modules in the exported graph
- `data/coco.yaml` (modified) — `path:` points to a machine-specific absolute path (`/home/jia/dataset/coco`); adjust per machine

Git history does NOT reflect any of the quantization work — do not use it to reason about these files.

`pytorch_quantization` is not installed by default; it comes from NVIDIA's index:
```bash
pip install nvidia-pyindex && pip install pytorch-quantization
```

## Standard YOLOv5 commands

```bash
python train.py --data coco128.yaml --weights yolov5s.pt --epochs 1 --imgsz 640   # train (auto-downloads data/weights)
python val.py --weights yolov5s.pt --data coco.yaml                              # validate
python detect.py --weights yolov5s.pt --source data/images/bus.jpg               # inference
python export.py --weights yolov5s.pt --include onnx                            # export (torchscript/onnx/engine/...)
python benchmarks.py --data coco128.yaml --weights yolov5n.pt --img 320         # CI-style benchmark
python models/yolo.py --cfg yolov5s.yaml                                        # build model from yaml
```
`classify/` and `segment/` have their own `train.py`/`val.py`/`predict.py`. Fast CPU smoke tests (what CI runs): add `--imgsz 64 --batch 32 --device cpu --epochs 1`. There is no pytest suite; CI (.github/workflows/ci-testing.yml) is functional shell tests. Python >=3.7, torch >=1.7 per upstream; this container runs Python 3.11 + torch 2.x + CUDA.

Lint/format: `pre-commit run --all-files` (isort, yapf, flake8; config in `setup.cfg`, max line length 120).

## Quantization workflow (the fork's purpose)

All commands run from the repo root (`scripts/qat.py` inserts `.` into `sys.path` and opens `data/hyps/...` relative to cwd). Requires a dataset at `--cocodir` with image-list txt files and a GPU. `--train-list`/`--val-list` each accept one or more list filenames inside `--cocodir` (default `train2017.txt` / `val2017.txt`). The class count is read from the checkpoint automatically; point `--data` at the matching dataset yaml (default `data/coco.yaml`) and align `--imgsz` with the trained size (default 640). mAP is YOLOv5-native by default; `--save-json` additionally runs pycocotools eval, which needs a COCO-format GT json at `<yaml path>/annotations/instances_val2017.json` — for custom datasets generate it first with `scripts/make_coco_json.py` (encodes image_id as the string filename stem and identity category ids, matching val.py's prediction side — `save_one_json` was made `is_coco`-aware: real COCO keeps int ids + coco91 class map, non-COCO uses string ids so COCOeval's imgIds sort never sees mixed types). End-to-end pipeline on a custom dataset (the 3-class one at `/root/dataset/3classes`, lists `origin_*.txt` / `ship_add_*.txt` / `ship_blank_add_*.txt`, `data/3classes.yaml`):

```bash
# 1. Train — 2-GPU DDP (batch 64 is the TOTAL, auto-split per GPU); single GPU: drop the launcher, add --device 0
python -m torch.distributed.run --nproc_per_node 2 train.py --weights weights/yolov5s.pt \
    --data data/3classes.yaml --imgsz 672 --batch-size 64 --workers 32

# 1b. Resume an interrupted run (all original args are restored from runs/train/exp2/opt.yaml)
python -m torch.distributed.run --nproc_per_node 2 train.py --resume runs/train/exp2/weights/last.pt

# 2. (only needed for --save-json) generate the COCO-format GT json → <yaml path>/annotations/instances_val2017.json
python scripts/make_coco_json.py --cocodir /root/dataset/3classes --data data/3classes.yaml \
    --val-list origin_val.txt ship_add_val.txt ship_blank_add_val.txt

# 3. PTQ + QAT — replace modules, calibrate, eval Origin/PTQ, finetune (best-mAP checkpoint kept in qat.pt)
python scripts/qat.py quantize runs/train/exp2/weights/best.pt --cocodir /root/dataset/3classes \
    --data data/3classes.yaml --imgsz 672 \
    --train-list origin_train.txt ship_add_train.txt ship_blank_add_train.txt \
    --val-list origin_val.txt ship_add_val.txt ship_blank_add_val.txt \
    --ptq ptq.pt --qat qat.pt --eval-origin --eval-ptq
#    add --save-json to also log pycocotools AP/AR (requires step 2)

# 4. Export Q/DQ ONNX for trtexec INT8 — raw s8/s16/s32 head outputs, dynamic batch
# 正方形（向后兼容，纯数字）
python scripts/qat.py export qat_1280.pt --size=672 --save=... --dynamic --noanchor

# 矩形（新增，HxW 格式）
python scripts/qat.py export qat_1280.pt --size=736x1280 --save=yolov5_3classes_qat_720p.onnx --dynamic --noanchor
```

#### The two export flavors (INT8 path vs FP16 path)

`qat.py export` serves two different downstream paths — pick the input checkpoint and flags accordingly:

| | QAT ONNX (INT8 path) | FP32 trimmed ONNX (FP16 path) |
|---|---|---|
| Command | `export qat.pt --size=672 --save=..._qat.onnx --dynamic --noanchor` | `export runs/.../best.pt --size=672 --save=..._fp32_trimmed.onnx --dynamic --noanchor --noqadd` |
| Input checkpoint | `qat.pt` / `ptq.pt` — carries Quant* modules with calibrated scales | `best.pt` — plain FP32 model, no quantizers |
| Graph emitted | **Q/DQ nodes embedded** (explicit quantization, trained scales) | **Clean FP32 graph** — zero Q/DQ nodes |
| Consumer | `qdq_translator.py` → PTQ ONNX + INT8 calib cache → trtexec `--int8 --calib` | trtexec `--fp16` directly (no calibration cache) |

- `--noqadd` matters only for the FP32 flavor: by default `cmd_export` also runs `replace_bottleneck_forward()`, routing Bottleneck residual adds through **QuantAdd**. On a QAT checkpoint those quantizers are calibrated and part of the trained graph (keep them); on a plain `best.pt` they get injected fresh with **uncalibrated scales** → ~14 bogus Q/DQ nodes whose garbage scales corrupt the FP16 build. Always pass `--noqadd` when exporting a non-quantized checkpoint.
- `quantize --imgsz` needs no rectangular variant: the train loader letterboxes to
  square (yolov5 training style) and the val loader is already aspect-preserving
  (`rect=True`, long side = imgsz). Scale statistics are dominated by the
  object-scale distribution, which long-side matching (e.g. `--imgsz 1280` for a
  736x1280 deployment) already captures — padding differences barely move per-tensor
  scales. Only add fixed-HxW letterbox calibration if the deployed quantization
  loss measures anomalously large.

- Shared flags: `--size=672` = deployment resolution; `--dynamic` = dynamic batch axis; `--noanchor` strips the head's anchor decode (the C++ app decodes with its own anchors; outputs raw s8/s16/s32).


```bash
# 1. (optional) find layers that hurt most when quantized
python scripts/qat.py sensitive yolov5s.pt --cocodir datasets/coco

# 2. PTQ: replace modules with quantized counterparts, calibrate (histogram + MSE amax), save
python scripts/qat.py quantize yolov5s.pt --cocodir datasets/coco --ptq ptq.pt --eval-ptq
#    QAT: add --qat qat.pt --iters 200  (finetunes against frozen FP outputs, keeps best-mAP checkpoint)
#    skip quantizing head convs: --ignore-policy "model\.24\.m\.(.*)"
#    Q/DQ on SiLU/Concat/Add too: --all-node-with-qdq (then custom rules are NOT applied)

# 3. Export Q/DQ ONNX for trtexec INT8
python scripts/qat.py export qat.pt --save qat.onnx --dynamic
#    DLA-friendly raw head outputs: --noanchor (outputs s8/s16/s32 instead of decoded "outputs")

# 4. Evaluate a .pt
python scripts/qat.py test yolov5s.pt --cocodir datasets/coco
```

### How the pieces fit

> A deep-dive on the QAT principles and a code walkthrough of all four stages
> (module replacement → calibration → STE-based distillation fine-tune → Q/DQ export)
> lives in [docs/qat-internals.zh-CN.md](docs/qat-internals.zh-CN.md) (Chinese).

- `quantize.replace_to_quantization_module()` swaps `nn.Conv2d`/`MaxPool2d`/`Linear` for `QuantConv2d`/... by walking `_DEFAULT_QUANT_MAP`; `replace_bottleneck_forward()` patches `Bottleneck.forward` to route the residual add through `QuantAdd` (the add op needs explicit input quantizers). Ordering matters: `replace_bottleneck_forward` → `replace_to_quantization_module` → (`apply_custom_rules_to_quantizer` unless `--all-node-with-qdq`) → `calibrate_model`.
- `quantization/rules.py` parses the intermediate ONNX graph to find conv pairs feeding the same Concat/MaxPool and shares one input quantizer between them (TensorRT Q/DQ folding requirement); it also ties `Bottleneck.addop` quantizers to `cv1`'s.
- Calibration feeds `num_batch=25` batches (images `/255`, no augment beyond the loader), then `load_calib_amax(method="mse")`.
- QAT (`quantize.finetune`) supervises layer *outputs* with MSE against a frozen FP copy of the model (`supervision_stride` controls which of `model.model[*]` layers contribute loss) — not the normal detection loss.
- During evaluation the Detect head (`model.model[24]`) quantization is disabled because ONNX export prunes the decode part of the head; PTQ/QAT checkpoints save the whole model via `torch.save({"model": model}, ...)` and are loaded with `weights_only=False`.
- `export_onnx` in `scripts/qat.py` temporarily monkeypatches the Detect head (`_make_grid` to CPU-constant grids; `--noanchor` replaces the forward to emit raw conv outputs) and flips `TensorQuantizer.use_fb_fake_quant` for Q/DQ emission (opset 13).

## Upstream architecture (orientation)

- `models/yolov5*.yaml` are layer lists parsed by `models/yolo.py` (`parse_model`) into a `DetectionModel` — a flat `nn.Sequential`-like `model.model` where e.g. index 24 is the `Detect` head producing strides 8/16/32. Building blocks (`Conv`, `C3`, `SPPF`, `Bottleneck`, ...) are in `models/common.py`; `Detect`/`Segment`/`BaseModel` in `models/yolo.py`.
- Task entry points at repo root: `train.py`, `val.py`, `detect.py`, `export.py`, `hubconf.py` (torch.hub), `benchmarks.py`; `classify/` and `segment/` mirror them per task.
- `utils/` holds the machinery: `dataloaders.py` (create_dataloader + augmentations), `general.py` (NMS, checks, coco eval glue), `loss.py`, `metrics.py`, `augmentations.py`, `loggers/`, `torch_utils.py`.
- `data/` = dataset yamls + `hyps/` hyperparameter presets (`hyp.scratch-{low,med,high}.yaml`); `data/scripts/get_coco*.sh` download datasets/weights.
- `weights/yolov5s.pt` is present locally; other weights auto-download via `attempt_download`.
