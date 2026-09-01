# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

NVIDIA sample deploying a QAT (quantization-aware trained) YOLOv5s on the Orin DLA (Deep Learning Accelerator) via cuDLA, demonstrating both cuDLA **hybrid mode** and cuDLA **standalone mode**. Runs on aarch64 Tegra hardware (Jetson Orin / DriveOS) only — nvcc targets `sm_87` and links `libcudla`, `libnvscibuf`, `libnvscisync`, TensorRT `nvinfer`.

## Environment (this machine)

- **Hardware:** NVIDIA Jetson AGX Orin Developer Kit — 64 GB unified memory, 2 DLA cores (own clock/power domain, physically parallel to the GPU).
- **OS:** Ubuntu 22.04.5 LTS, L4T R36.5.2 = JetPack 6.2.3, kernel `5.15.185-tegra`, aarch64.
- **Toolchain:** CUDA 12.6 (`/usr/local/cuda`, nvcc targets sm_87), gcc 11.4.0, cmake 3.22.1.
- **Libraries:** TensorRT 10.3.0.30 (`libnvinfer-dev`, trtexec at `/usr/src/tensorrt/bin/trtexec`), cuDLA (`/usr/local/cuda/lib64/libcudla.so`), OpenCV 4.8.0 (apt), `nvidia-l4t-nvsci` 36.5.2 — NvSci runtime libs only, no dev headers (hence [compat/nvsci-headers/](compat/nvsci-headers/)).
- **Python:** 3.10.12 with pycocotools installed (mAP evaluation ready).
- **Data:** COCO val2017 at `/media/data/jia/coco` (symlinked from `data/coco`); custom 3-class dataset at `/media/data/jia/3classes`. Both gitignored.
- **Network:** GitHub only reliably reachable via LAN proxy `http://192.168.11.61:7890`.
- **Python (export tooling):** fully installed and verified — see [Pipeline A](#pipeline-a--coco-qat-training--dla-deployment) for install commands and [Troubleshooting Log](#troubleshooting-log) for the traps. `torch` is not installed (only needed for QAT fine-tuning).
- **Heavy artifacts (`*.onnx`, `*.pt`, loadables, datasets) are gitignored** — a fresh clone needs them copied in; they are reproducible via the two pipelines below.

**Environment-reinstall episode (2026-08-28/09-01):** the OS rootfs was re-imaged on the first Orin (home survived), and the second Orin's image lacked the same packages. The complete fresh-machine dependency list and one-shot install command live in the [Troubleshooting Log](#troubleshooting-log) ("Fresh/re-imaged machine" entry): TRT (`libnvinfer10` — TRT 10 renamed it), `nvidia-l4t-dla-compiler` (trtexec DLA builds need `libnvdla_compiler.so`), JetPack OpenCV 4.8 (`libopencv` runtime + `libopencv-dev`), `libjsoncpp-dev`, plus `sudo ldconfig` after install. While at it, the vestigial `NvInfer.h`/`NvInferPlugin.h` includes were removed from [src/yolov5.h](src/yolov5.h) — the app uses no TensorRT symbol, so it no longer needs TRT headers to compile.

## Architecture

Pipeline: CPU (OpenCV decode + letterbox) → GPU (MatX reformat FP32→DLA input format) → **DLA (cuDLA inference)** → GPU (MatX reformat to FP16 planar + decode/NMS in `decode_nms.cu`) → CPU (bbox results). See [src/README.md](src/README.md).

- [src/validate_coco.cpp](src/validate_coco.cpp) — `main()`, CLI parsing. Entry point despite the name; handles single-image and validation flows. Image list via `--list` (default `./data/coco_val_2017_list.txt`); when `NUM_CLASSES != 80` predictions use string filename-stem image ids + identity category ids (matching `make_coco_json.py`'s GT encoding).
- [src/yolov5.cpp](src/yolov5.cpp) / [yolov5.h](src/yolov5.h) — pipeline orchestrator. Allocates CUDA buffers, owns the cuDLA context, drives pre/post-processing. `mInputScale` (used at yolov5.cpp:239 to quantize the FP32 input to INT8) comes from the calibration cache's `images:` entry — update it if a new cache has a different input scale. `mOutputScale1-3` are declared but **never used** (dead code — DLA outputs are FP16 and consumed directly). Network input is fixed at 1x3x672x672. Class count comes from the `YOLO_NUM_CLASSES` build macro (`make NUM_CLASSES=<n>`, default 80).
- **Two mutually exclusive cuDLA contexts** (compile-time selection in `yolov5.cpp`; comparison and selection guidance in [Hybrid vs Standalone Mode Selection](#hybrid-vs-standalone-mode-selection) below):
  - [src/cudla_context_hybrid.cpp](src/cudla_context_hybrid.cpp) — hybrid mode: CUDA-allocated buffers registered to cuDLA via `cudlaMemRegister`; task submitted on a CUDA stream. Simplest integration path.
  - [src/cudla_context_standalone.cpp](src/cudla_context_standalone.cpp) — standalone mode: NvSciBuf/NvSciSync for buffers and fences, imported into CUDA as external memory/semaphores. Avoids CUDA context creation on the DLA path; deterministic semaphore variant (`USE_DETERMINISTIC_SEMAPHORE`) is a workaround for older DriveOS/JetPack NvSciSync behavior.
  - Both context classes are intentionally self-contained (no sample dependencies) so users can copy them into their own projects — keep them that way when editing.
- [src/matx_reformat/](src/matx_reformat/) — separate CMake library wrapping the MatX submodule (pimpl pattern: `ReformatRunner`). Converts between DLA tensor layouts and planar formats: `ReformatImage`/`ReformatImageV2` (input CHW→HWC4/CHW16), `Run`/`Transpose` (output CHW16→planar for the 3 YOLOv5 heads at strides 8/16/32). Channel geometry derives from the same `YOLO_NUM_CLASSES` macro (CHW16/CHW32 pad head channels to multiples of 16/32).
- [data/model/](data/model/) — trtexec scripts that compile ONNX models into DLA loadables. INT8 loadable uses `--inputIOFormats=int8:dla_hwc4 --outputIOFormats=fp16:chw16` with the last head convs forced to FP16 (`--layerPrecisions`). `build_dla_standalone_loadable_v2.sh` falls back more layers to FP16 (higher mAP, slower). Requires trtexec with `--buildDLAStandalone` (patch in `data/trtexec-dla-standalone-trtv8.5.patch` only for TRT 8.5 / pre-JetPack-6.0).
- [export/](export/) — training-side tooling, independent of the C++ app: `yolov5-qat/` is the original overlay for an ultralytics yolov5 v7.0 checkout; `qdq_translator/` converts a QAT ONNX (Q/DQ nodes) into a PTQ ONNX + INT8 calibration cache. The server-side working copy lives at `yolov5_dla/` (v7.0 + the overlay, hardened with a CLI-parameterized `qat.py`, torch-2.x amp fixes and custom-dataset pycocotools support — see its own CLAUDE.md). The overlay is exactly 4 files: 3 new (`quantization/quantize.py`, `quantization/rules.py`, `scripts/qat.py`) + 1 modified (`models/common.py`).

**What the `yolov5-qat` overlay changes in `models/common.py`** (the only modified file, ~35 lines vs upstream v7.0): every functional `torch.cat(..., 1)` in `C3TR`, `C3`, `SPP`, `SPPF`, `Focus`, `GhostConv` and `Classify` is rerouted through a `self.concat = Concat(1)` submodule. Why: `quantization/quantize.py:initialize()` with `--all-node-with-qdq` (Option#2 in [export/README.md](export/README.md)) registers `models.common.Concat → QuantConcat` (and `nn.SiLU → QuantSiLU`) in pytorch-quantization's module-replacement map — a class-based swap that is only possible because Concat is a module; functional `torch.cat` could never be replaced. DLA requires an INT8 scale on every op including Concat (on GPU, TensorRT may let concat run at higher precision; qdq_translator's `--infer_concat_scales` exists for the Option#1 path where concat has no trained scale). The change is purely structural: identical numerics, identical exported ONNX graph.

DLA I/O format constraints (why the MatX reformat steps exist): INT8 input must be `kDLA_LINEAR`/`kDLA_HWC4`/`kCHW32`, FP16 in/out `kCHW16`; the sample uses INT8 HWC4 input + FP16 CHW16 output.

## Hybrid vs Standalone Mode Selection

The mode is a **compile-time** switch — `make clean` is required when switching (the Makefile does not track this dependency):

```bash
make clean && make run                                    # hybrid mode (default)
make clean && make run USE_DLA_STANDALONE_MODE=1          # standalone mode
make clean && make run USE_DLA_STANDALONE_MODE=1 USE_DETERMINISTIC_SEMAPHORE=1   # old DriveOS/JetPack only
```

The mode does not change results, accuracy, or the loadables — only how the DLA task's memory/submission/synchronization path is wired.

Facts that are easy to get wrong:

- **In both modes the model inference runs on DLA hardware** — the GPU never executes the convolutions. "Standalone" means the DLA *submission path* avoids CUDA (NvSciBuf/NvSciSync instead of CUDA memory/stream), not that the app stops using the GPU.
- In this sample the MatX pre/post reformat kernels run on the GPU in **both** modes (~1.6 ms total; see `yolov5::infer()`). Only `cudla_context_*.cpp` differs. A fully GPU-free pipeline would require moving pre/post elsewhere (e.g. feeding FP16 input to skip the INT8 quantize-reformat step).

| | Hybrid | Standalone |
|---|---|---|
| Buffers | `cudaMalloc` + `cudlaMemRegister` | NvSciBuf imported into CUDA as external memory |
| Sync | CUDA stream | NvSciSync fences ↔ CUDA external semaphores |
| CUDA context | required (constructor calls `cudaFree(0)` to create one) | not needed on the DLA path |
| Integration effort | low (~200-line self-contained class) | high (NvSci attr lists, reconcile, import/export) |

Choose **hybrid** for quick integration in CUDA-centric apps: single process, results post-processed on GPU, DLA tasks naturally ordered with other CUDA work on the same stream. Choose **standalone** when the GPU is heavily loaded and DLA must stay decoupled from CUDA stream scheduling, when extra processes must not create CUDA contexts, or when building a zero-copy NvSci pipeline (camera/NvMedia → DLA → other modules).

Resource framing: hybrid costs ≈0 GPU SM time for the inference itself but needs a CUDA context (tens of MB) and couples DLA submission to stream scheduling. AGX Orin has 2 DLA cores in their own clock/power domain, physically parallel to the GPU. Reference perf (README, bs=1): same YOLOv5s INT8 runs 1.82 ms on GPU vs 3.82 ms on DLA — DLA's value is energy efficiency and freeing the GPU for other work, not raw speed.

## Pipeline A — COCO: QAT Training → DLA Deployment

The original sample's line: fine-tune the official yolov5s on COCO with QAT, deploy INT8 on the DLA, validate with COCO val2017.

### A0. One-time build on the Jetson

```bash
# Heavy inputs are gitignored — copy data/model/*.onnx in first (git-LFS in the original NVIDIA repo).
bash data/model/build_dla_standalone_loadable.sh   # trtexec → INT8 + FP16 loadables into data/loadable/
bash src/matx_reformat/build_matx_reformat.sh      # matx lib (offline via CCCL shims; auto-proxy if GitHub is needed)
make                                               # app; NUM_CLASSES defaults to 80 for COCO
```

matx unit test: `./test` from `src/matx_reformat/build/` (with that dir on `LD_LIBRARY_PATH`). No `LD_LIBRARY_PATH` is needed for the app itself — it is linked with `-Wl,-rpath` to `src/matx_reformat/build`.

### A1. Run and validate (single image / COCO val)

```bash
make run                                # single image → detections drawn to result.jpg
make validate_cudla_int8                # COCO val2017 (5000 imgs) + pycocotools mAP, ~30–60 min
make validate_cudla_int8 ENGINE=data/loadable/<other 80-class loadable>.bin

# or directly:
./build/cudla_yolov5_app --engine data/loadable/yolov5.int8.int8hwc4in.fp16chw16out.standalone.bin \
    --image data/images/image.jpg --backend cudla_int8          # single image
./build/cudla_yolov5_app --engine ... --coco_path data/coco/ --backend cudla_int8   # validation (writes predict.json)
python3 test_coco_map.py --predict predict.json --coco data/coco/
```

Reference results: mAP50-95 **37.5** (DLA FP16) / **37.1** (DLA INT8 QAT) @ 1x3x672x672, ~5.5 ms/img INT8.

### A2. Server — QAT fine-tuning on COCO

Copy **only the `yolov5_dla/` directory** to a GPU server (v7.0 + hardened QAT layer; its own CLAUDE.md documents the internals). **Weight warning:** use `yolov5s.pt` (v7.0), not `yolov5su.pt` — the 'u' variant's anchors differ from the v7.0 defaults hardcoded in src/yolov5.cpp, silently corrupting decode.

```bash
cd yolov5_dla
pip install -r requirements.txt
pip install --no-deps --index-url https://pypi.nvidia.com pytorch-quantization && pip install absl-py prettytable
bash data/scripts/get_coco.sh           # YOLO-format COCO → ../datasets/coco (~20GB)

# Option#1 (what the repo's shipped model used); add --all-node-with-qdq for Option#2
python scripts/qat.py quantize yolov5s.pt --ptq=ptq.pt --qat=qat.pt \
    --cocodir=../datasets/coco --eval-origin --eval-ptq
python scripts/qat.py export qat.pt --size=672 --save=yolov5_trimmed_qat.onnx --dynamic --noanchor
```

The script calibrates → prints Origin/PTQ baseline mAP → fine-tunes, saving the best-AP epoch to `qat.pt` (history in `summary.json`); `--iters` caps batches/epoch for dry runs; full COCO epochs take hours per GPU. `--cocodir` (dataloaders) and `data/coco.yaml`'s `path:` (evaluation) must point to the same dataset.

### A3. Jetson — translate → loadable → validate (the retrain cycle)

```bash
# one-time: translator deps (torch-free)
pip3 install -i https://pypi.tuna.tsinghua.edu.cn/simple onnx onnx_graphsurgeon nvidia-pyindex
pip3 install -i https://pypi.tuna.tsinghua.edu.cn/simple onnxoptimizer   # no aarch64 wheel → source build ~15 min

# translate: QAT ONNX (explicit Q/DQ) → PTQ ONNX + INT8 calib cache + precision config
cd export/qdq_translator
python3 qdq_translator.py --input_onnx_models=../../data/model/yolov5_trimmed_qat.onnx \
    --output_dir=../../data/model/ --infer_concat_scales --infer_mul_scales
```

Then: copy [build_dla_standalone_loadable_8.26.sh](data/model/build_dla_standalone_loadable_8.26.sh) as a template, repoint its 3 paths (cache source, `_noqdq.onnx`, output `.bin`) and run it; the 3 head-conv FP16 `--layerPrecisions` stay unchanged. Compare the new cache's `images:` hex entry (big-endian IEEE-754) with `mInputScale` in src/yolov5.cpp — update if different. Finally `make` + `make validate_cudla_int8 ENGINE=...`.

Reference: the completed 8.26 retrain cycle lives in the repo (`yolov5_trimmed_qat_8.26.*` → `yolov5_8.26.int8...bin`): COCO mAP50-95 = **37.1** (equal to the shipped model), identical input scale, empty `layer_arg.txt` (a maximal FP16 suggestion list — not a requirement).

## Pipeline B — Custom Dataset: Train → QAT → DLA Deployment

End-to-end from your own dataset. Worked example in the repo: the 3-class model (`data/model/yolov5_3clases_qat*`, verified 2026-08-28: 8 detections @ 3.74 ms/img).

**①–⑤ Server — data prep, FP32 training, QAT fine-tune, ONNX export.** Run these inside the `yolov5_dla` toolkit; the detailed commands (environment setup, single/multi-GPU training, optional COCO-json GT generation, PTQ+QAT, export) are documented in [yolov5_dla/CLAUDE.md](yolov5_dla/CLAUDE.md) — follow that doc, then bring back **only the exported `.onnx`** to the Jetson's `data/model/`. Two deployment-critical points from the server side: train at `--imgsz 672` (matches the fixed deployment input), and watch the autoanchor log — if training replaces the anchors they must be synced into `anchors[]` in src/yolov5.cpp (step ⑧).

**⑥ Jetson — translate** (graph-level, dataset-agnostic; same deps and flags as A3):

```bash
cd export/qdq_translator
python3 qdq_translator.py --input_onnx_models=../../data/model/mydata_qat.onnx \
    --output_dir=../../data/model/ --infer_concat_scales --infer_mul_scales
```

Then compare the new cache's `images:` hex entry with `mInputScale` in src/yolov5.cpp — update if different. (`layer_arg.txt` empty is fine.)

**⑦ Jetson — build the loadable.** Copy [build_dla_standalone_loadable_3classes.sh](data/model/build_dla_standalone_loadable_3classes.sh), repoint its 3 paths (cache source, `_noqdq.onnx`, output `.bin`), run it. The 3 head-conv FP16 `--layerPrecisions` stay as-is (nc-independent node names).

**⑧ Jetson — build with the right class count** (nc is a build flag — no source edits):

```bash
NUM_CLASSES=<nc> bash src/matx_reformat/build_matx_reformat.sh   # rebuild the matx lib
make clean && make NUM_CLASSES=<nc>
```

`NUM_CLASSES` defaults to 80 (the shipped COCO model). It drives buffer sizes, the decode call sites and the CHW16/CHW32 reformat group dims (`YOLO_NUM_CLASSES` macro in [src/yolov5.cpp](src/yolov5.cpp) and [matx_reformat.cu](src/matx_reformat/matx_reformat.cu); `decode_nms.cu` is parameterized). **The flag must match the loadable passed to `--engine`/`ENGINE=`** — a mismatch silently garbage-results. Also: `anchors[]` in yolov5.cpp — only if training (①–⑤) replaced them; `mInputScale` — only if ⑥ found a different value.

**⑨ Jetson — run and verify:**

```bash
# single image (detections drawn to result.jpg)
./build/cudla_yolov5_app --engine data/loadable/mydata.int8...bin --image your.jpg --backend cudla_int8
# or: make run ENGINE=... IMAGE=...
```

Accuracy verdict — two options: server-side `val.py` mAP, or **on-device COCO-style eval for custom datasets** (verified 2026-08-31, 3-class model: mAP50-95 **0.466** @ 4356 images):

```bash
# GT json from YOLO txt labels (torch-free)
python3 yolov5_dla/scripts/make_coco_json.py --cocodir /path/to/ds --data <yaml> --val-list <list.txt>
# inference over the val list (predictions auto-switch to string ids + identity categories when nc != 80)
./build/cudla_yolov5_app --engine ... --coco_path /path/to/ds --list /path/to/ds/<list.txt> --backend cudla_int8
python3 test_coco_map.py --predict predict.json --coco /path/to/ds
```

If boxes look systematically wrong on in-domain images, compare the checkpoint's `model.model[-1].anchors` against yolov5.cpp's `anchors[]`.

Lazy alternative for experiments only: map custom classes into unused COCO slots (keep the 80-class head) — zero C++ changes, wasted head compute.

## Troubleshooting Log

Problems actually hit on these machines, with symptom → cause → fix.

**Build (Jetson)**

- trtexec INT8 build fails: *"Calibration table does not match calibrator algorithm type"* then *"Tensor `images` is bound to nullptr"* — TRT 10.x rejects the repo's `TRT-8600-EntropyCalibration2` cache header and tries to recalibrate with no data. Fix: the build scripts `sed` the header to the local TRT version into `data/loadable/*.cache` before `--calib`.
- trtexec logs `kPREFER_PRECISION_CONSTRAINTS cannot be set if kOBEY_PRECISION_CONSTRAINTS is set` — harmless TRT 10.3 noise; obey stays in effect, build succeeds.
- MatX configure fails cloning `libcudacxx` from GitHub — direct GitHub access is unreliable; `build_matx_reformat.sh` auto-routes through the LAN proxy when a direct probe fails. With the CCCL shim below the build needs no network at all.
- MatX 0.4.1 vs CUDA 12.6: NVTX v1-vs-v3 `#error` and missing `<__config>` — MatX fetches libcudacxx 2.1.0 which clashes with the toolkit's CCCL. Fix: [src/matx_reformat/compat/](src/matx_reformat/compat/) shims (`libcudacxx-shim/include` symlinks to `/usr/local/cuda/include` via `FETCHCONTENT_SOURCE_DIR_LIBCUDACXX`; `nvtx-shim/nvToolsExt.h` redirects to the nvtx3 drop-in). If a fetch failed before the shims existed, `rm -rf src/matx_reformat/build` first.
- `nvscibuf.h: No such file or directory` — JetPack ships NvSci runtime libs only, no dev headers. Fix: headers restored from the public DRIVE OS 6.0.9 doxygen `_source.html` pages into [compat/nvsci-headers/](compat/nvsci-headers/) (Makefile adds `-I ./compat/nvsci-headers`).
- `ld: cannot find -lnvscibuf` — the libs live in `/usr/lib/aarch64-linux-gnu/nvidia/`; the Makefile adds `-L` (standalone mode only).
- Running the binary directly fails with `libmatx_reformat.so: cannot open shared object file` — the Makefile's `export LD_LIBRARY_PATH` only applies to make's children. Fix: the link now bakes in `-Wl,-rpath` → no env var needed.
- Fresh/re-imaged machine — enable the commented `deb` lines in `/etc/apt/sources.list.d/nvidia-l4t-apt-source.list` first (it then redirects to the `.cn` mirror, no proxy needed), then install the full dependency set at once:

  ```bash
  sudo apt update && sudo apt install -y \
      libnvinfer10 libnvinfer-dev libnvinfer-bin \
      nvidia-l4t-dla-compiler \
      libopencv libopencv-dev \
      libjsoncpp-dev
  sudo ldconfig
  ```

  Why each: `libnvinfer10` (TRT 10 renamed it — `libnvinfer8` does not exist) + `libnvinfer-bin` = trtexec; `nvidia-l4t-dla-compiler` provides `libnvdla_compiler.so` — without it trtexec DLA builds die at startup with *"Unable to open library: libnvinfer_plugin.so.10 due to libnvdla_compiler.so"*; `libopencv` is the JetPack OpenCV 4.8 runtime body (`libopencv-dev` alone leaves dangling `/usr/lib/libopencv_*.so → *.so.408` symlinks → link failure); `libjsoncpp-dev` = `json/json.h`. If a freshly installed lib still reports "cannot open shared object file", run `sudo ldconfig` (the NVIDIA lib dir `/usr/lib/aarch64-linux-gnu/nvidia` is already registered in `nvidia-tegra.conf`; apt does not always refresh the cache). `pycocotools` comes via pip.

**Python environment**

- `pip install pytorch-quantization` → placeholder error / `sphinx-glpi-theme` unresolvable — the README's `pypi.ngc.nvidia.com` index is dead, PyPI hosts a same-name placeholder, and the real NVIDIA wheel declares a docs theme as a runtime dep. Fix: `pip install --no-deps --index-url https://pypi.nvidia.com pytorch-quantization && pip install absl-py prettytable`.
- `onnxoptimizer` has no aarch64 wheel — pip falls back to a silent ~15-min source build (cmake 3.22 suffices); the `==0.3.2` pin in requirements.txt is not load-bearing (only 4 passes are used, all present in 0.3.13).
- Installing onnx raises user-site numpy to 2.2.6 — pycocotools verified still working; if some package later complains about numpy ABI: `pip install "numpy<2"`.

**Training (server)**

- `yolov5su.pt` loads fine but decodes garbage on-device — 'u'-variant anchors differ from the v7.0 defaults hardcoded in src/yolov5.cpp. Always start from `yolov5s.pt`.
- Side-printed pycocotools AP ≈ 0.001 while yolov5's own table shows normal mAP — non-COCO dataset roots make upstream `save_one_json` write contiguous category ids. Fixed in yolov5_dla (`is_coco`-aware ids, `--save-json` opt-in, `make_coco_json.py` for COCO-format GT).
- `--cocodir` and the eval yaml's `path:` must point to the same dataset or val loads no images.

**Accuracy / runtime**

- **mAP ≈ 0.01 and objects missing** (e.g. bus not detected) — the app was built with a `NUM_CLASSES` that doesn't match the loadable (hit when an 80-class loadable ran against a 3-class build). Rebuild both libs with the matching `NUM_CLASSES` (see Pipeline B ⑧).
- `mOutputScale1-3` look like they need updating on retrain — they are dead code; only `mInputScale` (cache `images:` entry) is used.
- The app prepends `--coco_path` to every list entry — keep list entries relative (or absolute, now also supported); pass `--list` for non-COCO lists.
- Custom-dataset eval mismatch: the app's `coco80_to_coco91_class` map only applies when `NUM_CLASSES == 80`; custom models use identity ids + string image ids to match `make_coco_json.py`.
