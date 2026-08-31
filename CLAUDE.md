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
- **Data:** COCO val2017 already present under `data/coco/`.
- **Network:** GitHub only reliably reachable via LAN proxy `http://192.168.11.61:7890`.
- **Python (export tooling):** fully installed and verified — see [Export Toolchain (Python)](#export-toolchain-python) for the package list, install commands and traps. `torch` is not installed (only needed for QAT fine-tuning).

## Build & Run

One-time setup (order matters — the app links against the matx library and expects loadables under `data/loadable/`):

```bash
bash data/model/build_dla_standalone_loadable.sh   # trtexec → DLA loadables (INT8 + FP16) into data/loadable/
bash src/matx_reformat/build_matx_reformat.sh      # CMake build of matx_reformat lib (needs cmake ≥3.18, script auto-downloads it)
```

**Network note:** direct GitHub access from this machine is unreliable. `build_matx_reformat.sh` was patched to detect that and route through the LAN proxy `http://192.168.11.61:7890` (adjust/remove as needed). After the CCCL shim below, the MatX build no longer needs network at all.

**CUDA 12.6 / JetPack 6 compatibility (matx_reformat):** MatX 0.4.1 (pinned submodule) was written for CUDA 11-era CCCL and fetches `libcudacxx` 2.1.0 at configure time, which conflicts with CUDA 12.6's bundled CCCL (NVTX v1-vs-v3 `#error`, missing `<__config>`). Fix (no submodule edits): [src/matx_reformat/compat/](src/matx_reformat/compat/) contains a `libcudacxx-shim/` whose `include` symlinks to `/usr/local/cuda/include` (forces MatX onto the toolkit's own CCCL via `FETCHCONTENT_SOURCE_DIR_LIBCUDACXX`) and an `nvtx-shim/nvToolsExt.h` that redirects the legacy NVTX v1 include to the nvtx3 drop-in. Wired in [src/matx_reformat/CMakeLists.txt](src/matx_reformat/CMakeLists.txt) before `add_subdirectory(MatX)`. If CUDA moves off `/usr/local/cuda`, fix the symlink. If a fetch failed before the shim existed, `rm -rf src/matx_reformat/build` before retrying.

**TensorRT 10.x note (JetPack 6):** the repo's calibration cache `data/model/qat2ptq.cache` carries a `TRT-8600-EntropyCalibration2` header; TRT 10.x rejects it ("Calibration table does not match calibrator algorithm type") and then fails trying to recalibrate with no data (input tensor bound to nullptr). The build scripts were patched to `sed` the header to the locally installed TRT version into `data/loadable/qat2ptq.cache` and pass that to `--calib`. The `kPREFER_PRECISION_CONSTRAINTS cannot be set if kOBEY_PRECISION_CONSTRAINTS is set` error early in the trtexec log is harmless noise on TRT 10.3 (obey stays in effect).

Main app (root Makefile, output `build/cudla_yolov5_app`):

```bash
make run                          # single image, INT8, hybrid mode
make validate_cudla_int8          # COCO val + mAP (also: validate_cudla_fp16)
make run USE_DLA_STANDALONE_MODE=1
make run USE_DLA_STANDALONE_MODE=1 USE_DETERMINISTIC_SEMAPHORE=1   # for older DriveOS/JetPack
```

- **`make clean` is required when switching between hybrid and standalone mode** — the mode is a compile-time `#ifdef USE_DLA_STANDALONE_MODE` in [src/yolov5.cpp](src/yolov5.cpp), not a runtime option. The Makefile does not track this dependency.
- The Makefile was patched to compile only the cuDLA context matching the selected mode and to link `-lnvscibuf -lnvscisync` (plus `-L /usr/lib/aarch64-linux-gnu/nvidia/`) only in standalone mode. JetPack ships the NvSci runtime libs (`nvidia-l4t-nvsci`) but **not** the dev headers — those were reconstructed from the public DRIVE OS 6.0.9 doxygen `_source.html` pages into [compat/nvsci-headers/](compat/nvsci-headers/) (see its README for provenance/license); the Makefile adds `-I ./compat/nvsci-headers`. Both modes verified working on JetPack 6.2 (L4T r36.5.2). `USE_DETERMINISTIC_SEMAPHORE` was not needed on this JetPack.
- `USE_DETERMINISTIC_SEMAPHORE` only applies within standalone mode.
- `DEBUG=1` for -g, default is -O2.
- Clone must be `--recursive` (MatX submodule in `src/matx_reformat/MatX`); ONNX models in `data/model/` are git-LFS.

Running the binary directly:

```bash
./build/cudla_yolov5_app --engine data/loadable/<loadable>.bin --image data/images/image.jpg --backend cudla_int8
# or for COCO validation: --coco_path data/coco/ instead of --image (writes predict.json)
```

No `LD_LIBRARY_PATH` needed: the executable is linked with `-Wl,-rpath` pointing at `src/matx_reformat/build`, so it finds `libmatx_reformat.so` by itself (the Makefile's `export LD_LIBRARY_PATH=...` line only ever applied to make's own sub-processes — and note its `$LD_LIBRARY_PATH` is a Make typo that expands to `D_LIBRARY_PATH`, harmless).

mAP evaluation (needs COCO val2017 in `data/coco/` via `data/download_coco_validation_set.sh` + `pip3 install pycocotools`):

```bash
python3 test_coco_map.py --predict predict.json --coco ./data/coco/
```

matx_reformat unit test (the only test binary): built by `src/matx_reformat/build_matx_reformat.sh`, run `./test` from `src/matx_reformat/build/` with that dir on `LD_LIBRARY_PATH`.

Reference mAP on COCO 2017 val @ 1x3x672x672: 37.5 (DLA FP16), 37.1 (DLA INT8 QAT).

## Export Toolchain (Python)

Packages for the QAT→PTQ conversion path ([export/](export/)) — installed to the user site (`~/.local`), verified end-to-end on this machine (qdq_translator smoke-tested on `data/model/yolov5_trimmed_qat.onnx`):

| Package | Version | How installed |
|---|---|---|
| pytorch-quantization | 2.1.3 | `pip install --no-deps --index-url https://pypi.nvidia.com pytorch-quantization` (traps below) |
| absl-py, prettytable | latest | `pip install absl-py prettytable` — the real runtime deps of pytorch-quantization |
| onnx | 1.22.0 | `pip install -i https://pypi.tuna.tsinghua.edu.cn/simple onnx` |
| onnx_graphsurgeon | 0.6.1 | same mirror |
| onnxoptimizer | 0.3.13 | `pip install --no-cache-dir -i https://pypi.tuna.tsinghua.edu.cn/simple onnxoptimizer` — **no aarch64 wheel**, so pip automatically falls back to a source build (C++ compiled via cmake; the system cmake 3.22 is sufficient). Takes ~15 min on the Orin; pip stays silent for most of it (run it in a terminal you won't close), finishing with `Successfully built onnxoptimizer` |
| nvidia-pyindex | 1.0.9 | same mirror |

Install traps encountered:

- [export/README.md](export/README.md)'s `--extra-index-url https://pypi.ngc.nvidia.com` is dead (DNS fails) — NVIDIA's index moved to `pypi.nvidia.com`.
- PyPI hosts a same-name `pytorch-quantization` **placeholder** (6.8 kB, setup.py always errors); `--extra-index-url` has no priority over PyPI, so pip may pick the placeholder. Use `--index-url` to *replace* the default index entirely.
- The real NVIDIA wheel falsely declares `sphinx-glpi-theme` (a docs theme) as a runtime dependency, which is unresolvable from the NVIDIA index → install with `--no-deps`, then add `absl-py` + `prettytable` manually. `sphinx-glpi-theme` is never actually needed.
- [export/qdq_translator/requirements.txt](export/qdq_translator/requirements.txt) pins `onnxoptimizer==0.3.2`, but the pin is not load-bearing: the script only calls `optimize(model, passes=[...])` with 4 passes (`extract_constant_to_initializer`, `fuse_bn_into_conv`, `fuse_pad_into_conv`, `fuse_pad_into_pool`), all present in 0.3.13.
- Installing onnx raised user-site numpy to 2.2.6 (system 1.21.5 untouched); pycocotools verified still working. If a package later complains about numpy ABI, `pip install "numpy<2"`.

`torch` is deliberately NOT installed — only the QAT fine-tuning step needs it; `qdq_translator` runs torch-free.

Re-verify anytime:

```bash
python3 -c "import onnx, onnx_graphsurgeon, onnxoptimizer, pytorch_quantization; print('ok')"
cd export/qdq_translator && python3 qdq_translator.py \
    --input_onnx_models=../../data/model/yolov5_trimmed_qat.onnx \
    --output_dir=/tmp/qdq_out --infer_concat_scales --infer_mul_scales   # → PTQ ONNX + calib cache + precision config
```

## QAT Training (off-device, GPU server)

QAT fine-tuning runs on a separate GPU server (the Orin has no torch and training is impractical on-device). Copy **only the `yolov5_dla/` directory** — ultralytics yolov5 v7.0 plus a hardened, CLI-parameterized QAT layer (`quantization/`, `scripts/qat.py`, patched `models/common.py`). Its own CLAUDE.md documents the internals (module-replacement order, `rules.py` quantizer sharing, MSE-supervised fine-tuning, ONNX export tricks).

**Weight warning:** use `yolov5s.pt` (v7.0), not `yolov5su.pt`. The 'u' variant carries different anchors, while [src/yolov5.cpp](src/yolov5.cpp) hardcodes v7.0 default anchors to decode the `--noanchor`-exported model — a mismatch silently corrupts detections. `attempt_download()` auto-fetches yolov5s.pt if missing.

On the server:

```bash
cd yolov5_dla
pip install -r requirements.txt
pip install --no-deps --index-url https://pypi.nvidia.com pytorch-quantization && pip install absl-py prettytable
bash data/scripts/get_coco.sh        # YOLO-format COCO → ../datasets/coco (~20GB)

# COCO Option#1 (what the repo's shipped model used); add --all-node-with-qdq for Option#2
python scripts/qat.py quantize yolov5s.pt --ptq=ptq.pt --qat=qat.pt \
    --cocodir=../datasets/coco --eval-origin --eval-ptq
python scripts/qat.py export qat.pt --size=672 --save=yolov5_trimmed_qat.onnx --dynamic --noanchor
```

`yolov5_dla`'s `qat.py` is fully parameterized — no code edits for custom datasets: the class count is read from the checkpoint automatically, and `--data <yaml>`, `--imgsz`, `--batch-size`, multiple `--train-list`/`--val-list` files and opt-in `--save-json` (pycocotools eval) are all CLI flags. The script calibrates → prints Origin/PTQ baseline mAP → fine-tunes, saving the best-AP epoch to `qat.pt` (history in `summary.json`); `--iters` caps batches/epoch for quick dry-runs. Full COCO epochs take hours on one GPU.

The old **pycocotools AP ≈ 0.001 red herring** (category-id mismatch when the dataset root isn't named `coco`) is properly fixed here: `--save-json` is opt-in, `val.py`'s `save_one_json` is `is_coco`-aware (string image ids for custom datasets), and `scripts/make_coco_json.py` generates a COCO-format GT json so custom datasets can run COCOeval too.

Back on the Jetson: copy **only** `yolov5_trimmed_qat.onnx` to `data/model/`, run `qdq_translator.py --infer_concat_scales --infer_mul_scales` (see Export Toolchain above), rebuild the loadable, and check whether the new cache's `images:` scale differs from `mInputScale` in [src/yolov5.cpp](src/yolov5.cpp) (`mOutputScale1-3` are dead code, see Architecture) — then rebuild and re-validate mAP.

A completed retrain cycle (2026-08-26) lives in the repo as reference: `data/model/yolov5_trimmed_qat_8.26.onnx` (QAT ONNX from the server) + `yolov5_trimmed_qat_8.26_noqdq.onnx`/`.cache` (translator output) + [build_dla_standalone_loadable_8.26.sh](data/model/build_dla_standalone_loadable_8.26.sh) (INT8-only build, same 3 head-conv FP16 precisions) → `data/loadable/yolov5_8.26.int8...standalone.bin` (verified: COCO val2017 mAP50-95 = **37.1** — equal to the original shipped model — at ~5.5 ms/img; identical input scale so no C++ change was needed). The translator's `layer_arg.txt` being empty for the new model is fine — that file lists a maximal FP16 suggestion set, not a requirement. The Makefile's `run`/`validate_cudla_int8` targets accept `ENGINE=<path>` to point at such alternative loadables.

## Custom-Dataset (non-COCO) Workflow

End-to-end from your own dataset to DLA inference. Steps ①–⑤ run on the GPU server (one-time env setup: see [QAT Training (off-device, GPU server)](#qat-training-off-device-gpu-server)); steps ⑥–⑨ on the Jetson. A worked example lives in the repo — the 3-class model (`data/model/yolov5_3clases_qat*`, verified end-to-end 2026-08-28: 8 detections @ 3.74 ms/img vs 5.57 ms for the 80-class model).

**① Server — data prep.** YOLO format (`images/` + `labels/`) + `data/mydata.yaml` (path/train/val/nc/names). Also create image-list txt files (any names, passed via `--train-list`/`--val-list`; paths must contain `images/` so labels resolve):

```bash
find $PWD/datasets/mydata/images/train -name '*.jpg' > datasets/mydata/train.txt
find $PWD/datasets/mydata/images/val   -name '*.jpg' > datasets/mydata/val.txt
```

**② Server — FP32 training.** `--img 672` matches the fixed deployment input. Watch the autoanchor log: if anchors get replaced, they must be synced into C++ in step ⑧.

```bash
python train.py --img 672 --batch 32 --epochs 100 --data data/mydata.yaml --weights yolov5s.pt
```

**③ Server — nothing to adapt.** `yolov5_dla`'s `qat.py` reads the class count from the checkpoint automatically and takes `--data`/`--imgsz`/`--batch-size`/`--train-list`/`--val-list` as CLI flags. (Only the original `export/yolov5-qat` overlay had the 3 COCO hardcodes that needed code edits.)

**④ Server — QAT fine-tune:**

```bash
python scripts/qat.py quantize runs/train/exp/weights/best.pt \
    --ptq=ptq.pt --qat=qat.pt --cocodir=datasets/mydata \
    --data data/mydata.yaml --imgsz 672 \
    --train-list train.txt --val-list val.txt \
    --eval-origin --eval-ptq
```

**⑤ Server — export ONNX, then copy ONLY the `.onnx` back to the Jetson's `data/model/`:**

```bash
python scripts/qat.py export qat.pt --size=672 --save=mydata_qat.onnx --dynamic --noanchor
```

**⑥ Jetson — translate** (graph-level, dataset-agnostic; outputs land next to the ONNX):

```bash
cd export/qdq_translator
python3 qdq_translator.py --input_onnx_models=../../data/model/mydata_qat.onnx \
    --output_dir=../../data/model/ --infer_concat_scales --infer_mul_scales
```

Then compare the new cache's `images:` hex entry (big-endian IEEE-754 → float) with `mInputScale` in src/yolov5.cpp — update if different. An empty `layer_arg.txt` is fine (it is a maximal FP16 suggestion list, not a requirement).

**⑦ Jetson — build the loadable.** Copy [build_dla_standalone_loadable_3classes.sh](data/model/build_dla_standalone_loadable_3classes.sh), repoint its 3 paths (cache source, `_noqdq.onnx`, output `.bin`), run it. The 3 head-conv FP16 `--layerPrecisions` stay unchanged — those node names are nc-independent.

**⑧ Jetson — C++ adaptation** (nc is now a named constant, one line per file):

- `kNumClasses` in [src/yolov5.cpp:31](src/yolov5.cpp) — drives buffer sizes + the two decode call sites
- `kNumClasses` in [matx_reformat.cu](src/matx_reformat/matx_reformat.cu) — `kChPerAnchor`/`kChw16Groups`/`kChw32Groups` derive from it (CHW16/CHW32 pad head channels to multiples of 16/32, so the 5D reformat views' group dim changes with nc); `decode_nms.cu` is parameterized — no change
- `anchors[]` in yolov5.cpp — only if ② replaced them; `mInputScale` — only if ⑥ found a different value

```bash
cd src/matx_reformat/build && make -j4 && cd ../../.. && make
```

**⑨ Jetson — run and verify** (detections drawn to `result.jpg`):

```bash
./build/cudla_yolov5_app --engine data/loadable/mydata.int8...bin --image your.jpg --backend cudla_int8
# or: make run ENGINE=... IMAGE=...
```

Accuracy verdict: server-side `val.py` mAP (`test_coco_map.py` is COCO-only). If boxes look systematically wrong on in-domain images, compare the checkpoint's `model.model[-1].anchors` against yolov5.cpp's `anchors[]`.

Lazy alternative for experiments only: map custom classes into unused COCO slots (keep the 80-class head) — zero C++ changes, wasted head compute.

**Environment-reinstall episode (2026-08-28):** the OS rootfs was re-imaged (home survived). Restored via apt from the NVIDIA Jetson repo (`/etc/apt/sources.list.d/nvidia-l4t-apt-source.list` — ships with all `deb` lines commented out, must be enabled first; it then redirects to the `.cn` mirror, no proxy needed): `libnvinfer10` (**not** `libnvinfer8` — TRT 10 renamed the package) + `libnvinfer-dev` + `libnvinfer-bin` for trtexec, and `libopencv` (the JetPack OpenCV 4.8 runtime lives in a package with that exact name; `libopencv-dev` alone only installs dangling `/usr/lib/libopencv_*.so → *.so.408` symlinks). `pycocotools` reinstalled via pip. While at it, the vestigial `NvInfer.h`/`NvInferPlugin.h` includes were removed from [src/yolov5.h](src/yolov5.h) — the app uses no TensorRT symbol, so it no longer needs TRT headers to compile.

## Architecture

Pipeline: CPU (OpenCV decode + letterbox) → GPU (MatX reformat FP32→DLA input format) → **DLA (cuDLA inference)** → GPU (MatX reformat to FP16 planar + decode/NMS in `decode_nms.cu`) → CPU (bbox results). See [src/README.md](src/README.md).

- [src/validate_coco.cpp](src/validate_coco.cpp) — `main()`, CLI parsing. Entry point despite the name; handles both single-image and COCO validation flows.
- [src/yolov5.cpp](src/yolov5.cpp) / [yolov5.h](src/yolov5.h) — pipeline orchestrator. Allocates CUDA buffers, owns the cuDLA context, drives pre/post-processing. `mInputScale` (used at yolov5.cpp:239 to quantize the FP32 input to INT8) comes from the `images:` entry of `data/model/qat2ptq.cache` — update it if a new calibration cache has a different input scale. Note `mOutputScale1-3` are declared but **never used** (dead code — DLA outputs are FP16 and consumed directly). Network input is fixed at 1x3x672x672.
- **Two mutually exclusive cuDLA contexts** (compile-time selection in `yolov5.cpp`; comparison and selection guidance in [Hybrid vs Standalone Mode Selection](#hybrid-vs-standalone-mode-selection) below):
  - [src/cudla_context_hybrid.cpp](src/cudla_context_hybrid.cpp) — hybrid mode: CUDA-allocated buffers registered to cuDLA via `cudlaMemRegister`; task submitted on a CUDA stream. Simplest integration path.
  - [src/cudla_context_standalone.cpp](src/cudla_context_standalone.cpp) — standalone mode: NvSciBuf/NvSciSync for buffers and fences, imported into CUDA as external memory/semaphores. Avoids CUDA context creation on the DLA path; deterministic semaphore variant (`USE_DETERMINISTIC_SEMAPHORE`) is a workaround for older DriveOS/JetPack NvSciSync behavior.
  - Both context classes are intentionally self-contained (no sample dependencies) so users can copy them into their own projects — keep them that way when editing.
- [src/matx_reformat/](src/matx_reformat/) — separate CMake library wrapping the MatX submodule (pimpl pattern: `ReformatRunner`). Converts between DLA tensor layouts and planar formats: `ReformatImage`/`ReformatImageV2` (input CHW→HWC4/CHW16), `Run`/`Transpose` (output CHW16→planar for the 3 YOLOv5 heads at strides 8/16/32).
- [data/model/](data/model/) — trtexec scripts that compile the two ONNX models into DLA loadables. INT8 loadable uses `--inputIOFormats=int8:dla_hwc4 --outputIOFormats=fp16:chw16 --calib=qat2ptq.cache` with the last head convs forced to FP16 (`--layerPrecisions`). `build_dla_standalone_loadable_v2.sh` falls back more layers to FP16 (higher mAP, slower). Requires a trtexec with `--buildDLAStandalone` support (patch in `data/trtexec-dla-standalone-trtv8.5.patch` for TRT 8.5 / pre-JetPack-6.0).
- [export/](export/) — training-side tooling, independent of the C++ app: `yolov5-qat/` is copied into an ultralytics yolov5 v7.0 checkout for QAT fine-tuning; `qdq_translator/` converts a QAT ONNX (Q/DQ nodes) into a PTQ ONNX + INT8 calibration cache. The server-side working copy lives at `yolov5_dla/` (v7.0 + the overlay, hardened with a CLI-parameterized `qat.py`, torch-2.x amp fixes and custom-dataset pycocotools support — see its own CLAUDE.md). The overlay is exactly 4 files: 3 new (`quantization/quantize.py`, `quantization/rules.py`, `scripts/qat.py`) + 1 modified (`models/common.py`).

**What the `yolov5-qat` overlay changes in `models/common.py`** (the only modified file, ~35 lines vs upstream v7.0): every functional `torch.cat(..., 1)` in `C3TR`, `C3`, `SPP`, `SPPF`, `Focus`, `GhostConv` and `Classify` is rerouted through a `self.concat = Concat(1)` submodule. Why: `quantization/quantize.py:initialize()` with `--all-node-with-qdq` (Option#2 in [export/README.md](export/README.md)) registers `models.common.Concat → QuantConcat` (and `nn.SiLU → QuantSiLU`) in pytorch-quantization's module-replacement map — a class-based swap that is only possible because Concat is a module; functional `torch.cat` could never be replaced. DLA requires an INT8 scale on every op including Concat (on GPU, TensorRT may let concat run at higher precision; qdq_translator's `--infer_concat_scales` exists for the Option#1 path where concat has no trained scale). The change is purely structural: identical numerics, identical exported ONNX graph.

DLA I/O format constraints (why the MatX reformat steps exist): INT8 input must be `kDLA_LINEAR`/`kDLA_HWC4`/`kCHW32`, FP16 in/out `kCHW16`; the sample uses INT8 HWC4 input + FP16 CHW16 output.

## Hybrid vs Standalone Mode Selection

Switching mechanics are in Build & Run (compile-time flag + `make clean`). The mode does not change results, accuracy, or the loadables — only how the DLA task's memory/submission/synchronization path is wired.

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
