# CLAUDE.md（中文版）

本文件为 Claude Code (claude.ai/code) 在此仓库中工作时提供指导。英文原版见 [CLAUDE.md](CLAUDE.md)。

## 项目概述

NVIDIA 官方示例：将 QAT（量化感知训练）后的 YOLOv5s 部署到 Orin DLA（深度学习加速器）上，通过 cuDLA 同时演示 cuDLA **混合模式（hybrid mode）** 与 cuDLA **独立模式（standalone mode）** 两种用法。只能在 aarch64 Tegra 硬件（Jetson Orin / DriveOS）上运行 —— nvcc 编译目标为 `sm_87`，并链接 `libcudla`、`libnvscibuf`、`libnvscisync` 以及 TensorRT 的 `nvinfer`。

## 运行环境（本机）

- **硬件：** NVIDIA Jetson AGX Orin 开发套件 —— 64GB 统一内存，2 个 DLA 核心（独立时钟/供电域，与 GPU 物理并行）。
- **系统：** Ubuntu 22.04.5 LTS，L4T R36.5.2 = JetPack 6.2.3，内核 `5.15.185-tegra`，aarch64。
- **工具链：** CUDA 12.6（`/usr/local/cuda`，nvcc 目标 sm_87）、gcc 11.4.0、cmake 3.22.1。
- **库：** TensorRT 10.3.0.30（`libnvinfer-dev`，trtexec 位于 `/usr/src/tensorrt/bin/trtexec`）、cuDLA（`/usr/local/cuda/lib64/libcudla.so`）、OpenCV 4.8.0（apt 版）、`nvidia-l4t-nvsci` 36.5.2 —— 仅运行库、无开发头文件（因此才有 [compat/nvsci-headers/](compat/nvsci-headers/)）。
- **Python：** 3.10.12，已装 pycocotools（mAP 评估就绪）。
- **数据：** COCO val2017 已下载到 `data/coco/`。
- **网络：** GitHub 仅能通过局域网代理 `http://192.168.11.61:7890` 稳定访问。
- **Python（export 工具链）：** 已完整安装并验证 —— 包清单、安装命令与踩坑说明见下文「Export 工具链（Python）」一节。`torch` 未安装（仅 QAT 微调需要）。

## 构建与运行

一次性准备工作（顺序重要 —— 主程序会链接 matx 库，并期望 loadable 位于 `data/loadable/` 下）：

```bash
bash data/model/build_dla_standalone_loadable.sh   # 用 trtexec 生成 DLA loadable（INT8 + FP16）到 data/loadable/
bash src/matx_reformat/build_matx_reformat.sh      # CMake 构建 matx_reformat 库（需 cmake ≥3.18，脚本会自动下载）
```

**网络注意事项：** 本机直连 GitHub 不稳定。`build_matx_reformat.sh` 已修改为：直连失败时自动走局域网代理 `http://192.168.11.61:7890`（按需调整或删除）。有了下面的 CCCL shim 后，MatX 构建已完全不需要联网。

**CUDA 12.6 / JetPack 6 兼容性（matx_reformat）：** MatX 0.4.1（锁定的子模块版本）是 CUDA 11 时代的代码，配置时会拉取 `libcudacxx` 2.1.0，与 CUDA 12.6 自带的新版 CCCL 冲突（NVTX v1/v3 `#error`、找不到 `<__config>`）。修复方式（不改动子模块）：[src/matx_reformat/compat/](src/matx_reformat/compat/) 下的 `libcudacxx-shim/include` 软链到 `/usr/local/cuda/include`（通过 `FETCHCONTENT_SOURCE_DIR_LIBCUDACXX` 强制 MatX 使用工具链自带的 CCCL），`nvtx-shim/nvToolsExt.h` 把旧版 NVTX v1 引入重定向到 nvtx3 兼容层。在 [src/matx_reformat/CMakeLists.txt](src/matx_reformat/CMakeLists.txt) 的 `add_subdirectory(MatX)` 之前接入。如果 CUDA 不在 `/usr/local/cuda`，需修改软链。若 shim 之前拉取失败过，重试前先 `rm -rf src/matx_reformat/build` 清掉残留。

**TensorRT 10.x 注意事项（JetPack 6）：** 仓库自带的校准缓存 `data/model/qat2ptq.cache` 头部是 `TRT-8600-EntropyCalibration2`；TRT 10.x 会拒绝它（报 "Calibration table does not match calibrator algorithm type"），随后在没有校准数据的情况下尝试重新校准而失败（输入张量绑定到 nullptr）。构建脚本已修复：用 `sed` 把头部版本号改写为本机 TRT 版本，生成到 `data/loadable/qat2ptq.cache` 再传给 `--calib`。trtexec 日志开头出现的 `kPREFER_PRECISION_CONSTRAINTS cannot be set if kOBEY_PRECISION_CONSTRAINTS is set` 错误在 TRT 10.3 上是无害噪音（obey 约束仍然生效）。

主程序（根目录 Makefile，产物为 `build/cudla_yolov5_app`）：

```bash
make run                          # 单张图片，INT8，混合模式
make validate_cudla_int8          # COCO 验证集 + mAP 评估（另有：validate_cudla_fp16）
make run USE_DLA_STANDALONE_MODE=1
make run USE_DLA_STANDALONE_MODE=1 USE_DETERMINISTIC_SEMAPHORE=1   # 用于较老的 DriveOS/JetPack
```

- **在混合模式与独立模式之间切换时必须先 `make clean`** —— 模式切换是 [src/yolov5.cpp](src/yolov5.cpp) 中的编译期 `#ifdef USE_DLA_STANDALONE_MODE`，而非运行时选项。Makefile 不会跟踪这个依赖。
- Makefile 已修改为：只编译与当前模式对应的 cuDLA 上下文源文件，且仅在独立模式链接 `-lnvscibuf -lnvscisync`（并加 `-L /usr/lib/aarch64-linux-gnu/nvidia/`）。JetPack 只带 NvSci 运行库（`nvidia-l4t-nvsci` 包），**不带**开发头文件 —— 头文件已从 DRIVE OS 6.0.9 公开文档的 doxygen `_source.html` 页面还原到 [compat/nvsci-headers/](compat/nvsci-headers/)（来源与许可说明见其 README），Makefile 加了 `-I ./compat/nvsci-headers`。两种模式均已在 JetPack 6.2（L4T r36.5.2）上验证可用，本机的 JetPack 不需要 `USE_DETERMINISTIC_SEMAPHORE`。
- `USE_DETERMINISTIC_SEMAPHORE` 仅在独立模式下生效。
- `DEBUG=1` 使用 -g 编译，默认为 -O2。
- 克隆仓库时必须加 `--recursive`（`src/matx_reformat/MatX` 为子模块）；`data/model/` 下的 ONNX 模型使用 git-LFS。

直接运行可执行文件：

```bash
./build/cudla_yolov5_app --engine data/loadable/<loadable>.bin --image data/images/image.jpg --backend cudla_int8
# COCO 验证时：用 --coco_path data/coco/ 替代 --image（会生成 predict.json）
```

不需要设置 `LD_LIBRARY_PATH`：可执行文件链接时加了 `-Wl,-rpath` 指向 `src/matx_reformat/build`，能自己找到 `libmatx_reformat.so`（Makefile 里那行 `export LD_LIBRARY_PATH=...` 只对 make 自身的子进程生效 —— 另外其中的 `$LD_LIBRARY_PATH` 是 Make 语法笔误，会展开成 `D_LIBRARY_PATH`，无害）。

mAP 评估（需先用 `data/download_coco_validation_set.sh` 将 COCO val2017 下载到 `data/coco/`，并 `pip3 install pycocotools`）：

```bash
python3 test_coco_map.py --predict predict.json --coco ./data/coco/
```

matx_reformat 单元测试（仓库中唯一的测试程序）：由 `src/matx_reformat/build_matx_reformat.sh` 构建，在 `src/matx_reformat/build/` 目录下运行 `./test`，并需将该目录加入 `LD_LIBRARY_PATH`。

COCO 2017 val 参考精度（输入 1x3x672x672）：37.5（DLA FP16）、37.1（DLA INT8 QAT）。

## Export 工具链（Python）

QAT→PTQ 转化路径（[export/](export/)）所需的包 —— 已装入用户目录（`~/.local`），并在本机端到端验证过（用 `data/model/yolov5_trimmed_qat.onnx` 对 qdq_translator 做过冒烟测试）：

| 包 | 版本 | 安装方式 |
|---|---|---|
| pytorch-quantization | 2.1.3 | `pip install --no-deps --index-url https://pypi.nvidia.com pytorch-quantization`（坑见下） |
| absl-py、prettytable | 最新 | `pip install absl-py prettytable` —— pytorch-quantization 真正的运行时依赖 |
| onnx | 1.22.0 | `pip install -i https://pypi.tuna.tsinghua.edu.cn/simple onnx` |
| onnx_graphsurgeon | 0.6.1 | 同上镜像 |
| onnxoptimizer | 0.3.13 | `pip install --no-cache-dir -i https://pypi.tuna.tsinghua.edu.cn/simple onnxoptimizer` —— **无 aarch64 wheel**，pip 自动转为源码编译（C++，经 cmake 构建，系统 cmake 3.22 即可）。Orin 上约 15 分钟；编译期间长时间无输出属正常（建议放后台或别关终端），结束时打印 `Successfully built onnxoptimizer` |
| nvidia-pyindex | 1.0.9 | 同上镜像 |

安装时踩过的坑：

- [export/README.md](export/README.md) 里的 `--extra-index-url https://pypi.ngc.nvidia.com` 已废弃（DNS 解析失败）—— NVIDIA 的 pip 源迁到了 `pypi.nvidia.com`。
- PyPI 上有一个**同名占位包** `pytorch-quantization`（6.8KB，setup.py 故意报错）；`--extra-index-url` 相对 PyPI 没有优先级，pip 可能拿到占位包。必须用 `--index-url` **替换**默认源。
- NVIDIA 源里的真包错把 `sphinx-glpi-theme`（文档主题）声明成运行时依赖，而它不在 NVIDIA 源里 → 用 `--no-deps` 安装，再手动补 `absl-py` + `prettytable`。`sphinx-glpi-theme` 实际永远不会用到。
- [export/qdq_translator/requirements.txt](export/qdq_translator/requirements.txt) 锁了 `onnxoptimizer==0.3.2`，但这个锁并不关键：脚本只调用 `optimize(model, passes=[...])`，用到 4 个 pass（`extract_constant_to_initializer`、`fuse_bn_into_conv`、`fuse_pad_into_conv`、`fuse_pad_into_pool`），在 0.3.13 里全部存在。
- 装 onnx 时把用户级 numpy 升到了 2.2.6（系统的 1.21.5 不受影响）；已验证 pycocotools 仍正常。若日后有包报 numpy ABI 错误，`pip install "numpy<2"` 回落。

`torch` 刻意未安装 —— 只有 QAT 微调那一步需要；`qdq_translator` 无需 torch 即可运行。

随时复验：

```bash
python3 -c "import onnx, onnx_graphsurgeon, onnxoptimizer, pytorch_quantization; print('ok')"
cd export/qdq_translator && python3 qdq_translator.py \
    --input_onnx_models=../../data/model/yolov5_trimmed_qat.onnx \
    --output_dir=/tmp/qdq_out --infer_concat_scales --infer_mul_scales   # → PTQ ONNX + 校准缓存 + 精度配置
```

## QAT 训练（在 GPU 服务器上进行）

QAT 微调在单独的 GPU 服务器上做（本机 Orin 没装 torch，在设备上训练也不现实）。**只需拷贝 `yolov5_dla/` 目录** —— 它是 ultralytics yolov5 v7.0 加上工程化的 QAT 层（`quantization/`、`scripts/qat.py`、改过的 `models/common.py`），全部通过 CLI 参数驱动。它自带的 CLAUDE.md 记录了内部机制（模块替换顺序、`rules.py` 量化器共享、MSE 监督微调、ONNX 导出技巧）。

**权重警告：** 用 `yolov5s.pt`（v7.0），不要用 `yolov5su.pt`。'u' 版本的锚点不同，而 [src/yolov5.cpp](src/yolov5.cpp) 对 `--noanchor` 导出的模型是**硬编码** v7.0 默认锚点解码的 —— 不匹配会导致检测结果悄悄错乱。缺失时 `attempt_download()` 会自动下载 yolov5s.pt。

服务器上：

```bash
cd yolov5_dla
pip install -r requirements.txt
pip install --no-deps --index-url https://pypi.nvidia.com pytorch-quantization && pip install absl-py prettytable
bash data/scripts/get_coco.sh        # 生成 YOLO 格式 COCO → ../datasets/coco（约 20GB）

# Option#1（仓库自带模型的路线）；Option#2 加 --all-node-with-qdq
python scripts/qat.py quantize yolov5s.pt --ptq=ptq.pt --qat=qat.pt \
    --cocodir=../datasets/coco --eval-origin --eval-ptq
python scripts/qat.py export qat.pt --size=672 --save=yolov5_trimmed_qat.onnx --dynamic --noanchor
```

`yolov5_dla` 的 `qat.py` 已完全参数化 —— 自定义数据集**无需改任何代码**：类别数从 checkpoint 自动读取，`--data <yaml>`、`--imgsz`、`--batch-size`、多份 `--train-list`/`--val-list`、可选的 `--save-json`（pycocotools 评测）都是命令行参数。脚本流程：校准 → 打印 Origin/PTQ 基线 mAP → 微调，每个 epoch 评测并保存 mAP 最优的 `qat.pt`（历史在 `summary.json`）；`--iters` 可限制每 epoch 的 batch 数用于快速试跑。完整 COCO 训练单卡需数小时。

旧的 **pycocotools AP ≈ 0.001 干扰项**（数据集根目录不叫 `coco` 时的类别 ID 错配）在此版本已根治：`--save-json` 默认关闭、`val.py` 的 `save_one_json` 支持 `is_coco` 区分（自定义数据集用字符串 image id）、新增 `scripts/make_coco_json.py` 可为自定义数据集生成 COCO 格式 GT json 以跑 COCOeval。

回传 Jetson 后：**只拷** `yolov5_trimmed_qat.onnx` 到 `data/model/`，跑 `qdq_translator.py --infer_concat_scales --infer_mul_scales`（见上文 Export 工具链），重建 loadable，并检查新缓存的 `images:` scale 是否与 [src/yolov5.cpp](src/yolov5.cpp) 的 `mInputScale` 不同（`mOutputScale1-3` 是死代码，见架构一节）—— 然后重新编译、复验 mAP。

仓库里保留了一轮完整的重训练成果（2026-08-26）作参考：`data/model/yolov5_trimmed_qat_8.26.onnx`（服务器来的 QAT ONNX）+ `yolov5_trimmed_qat_8.26_noqdq.onnx`/`.cache`（translator 产物）+ [build_dla_standalone_loadable_8.26.sh](data/model/build_dla_standalone_loadable_8.26.sh)（仅 INT8 构建，沿用 3 个检测头 conv 的 FP16 精度约束）→ `data/loadable/yolov5_8.26.int8...standalone.bin`（已验证：COCO val2017 mAP50-95 = **37.1**，与原版模型持平，~5.5ms/图；输入 scale 与旧模型相同故 C++ 无需改动）。新模型 translator 生成的 `layer_arg.txt` 为空是正常的 —— 该文件只是"最大 FP16 建议清单"，不是必需清单。Makefile 的 `run`/`validate_cudla_int8` 目标支持 `ENGINE=<路径>` 指定其他 loadable。

## 自定义数据集（非 COCO）工作流

从自有数据集到 DLA 推理的端到端流程。第 ①–⑤ 步在 GPU 服务器上（一次性环境搭建见上文「QAT 训练」一节）；第 ⑥–⑨ 步在 Jetson 上。仓库内有一个完整实例 —— 三分类模型（`data/model/yolov5_3clases_qat*`，2026-08-28 端到端验证：8 个检测 @ 3.74ms/图，对比 80 类模型的 5.57ms）。

**① 服务器 —— 数据准备。** YOLO 格式（`images/` + `labels/`）+ `data/mydata.yaml`（path/train/val/nc/names）。另外生成图片列表 txt 文件（名字任意，经 `--train-list`/`--val-list` 传入；路径须含 `images/` 以便推导标签路径）：

```bash
find $PWD/datasets/mydata/images/train -name '*.jpg' > datasets/mydata/train.txt
find $PWD/datasets/mydata/images/val   -name '*.jpg' > datasets/mydata/val.txt
```

**② 服务器 —— FP32 训练。** `--img 672` 对齐部署端固定输入。盯 autoanchor 日志：若锚点被替换，需在第 ⑧ 步同步进 C++。

```bash
python train.py --img 672 --batch 32 --epochs 100 --data data/mydata.yaml --weights yolov5s.pt
```

**③ 服务器 —— 无需适配。** `yolov5_dla` 的 `qat.py` 自动从 checkpoint 读取类别数，`--data`/`--imgsz`/`--batch-size`/`--train-list`/`--val-list` 都是命令行参数。（只有原始 `export/yolov5-qat` 覆盖层存在三处 COCO 硬编码、需要改代码。）

**④ 服务器 —— QAT 微调：**

```bash
python scripts/qat.py quantize runs/train/exp/weights/best.pt \
    --ptq=ptq.pt --qat=qat.pt --cocodir=datasets/mydata \
    --data data/mydata.yaml --imgsz 672 \
    --train-list train.txt --val-list val.txt \
    --eval-origin --eval-ptq
```

**⑤ 服务器 —— 导出 ONNX，然后只把 `.onnx` 拷回 Jetson 的 `data/model/`：**

```bash
python scripts/qat.py export qat.pt --size=672 --save=mydata_qat.onnx --dynamic --noanchor
```

**⑥ Jetson —— 转化**（图级操作，与数据集无关；产物生成在 ONNX 旁边）：

```bash
cd export/qdq_translator
python3 qdq_translator.py --input_onnx_models=../../data/model/mydata_qat.onnx \
    --output_dir=../../data/model/ --infer_concat_scales --infer_mul_scales
```

然后用新缓存的 `images:` 十六进制条目（大端 IEEE-754 → float）对比 src/yolov5.cpp 的 `mInputScale`，不同则更新。`layer_arg.txt` 为空是正常的（那是"最大 FP16 建议清单"，不是必需清单）。

**⑦ Jetson —— 构建 loadable。** 复制 [build_dla_standalone_loadable_3classes.sh](data/model/build_dla_standalone_loadable_3classes.sh)，改 3 处路径（缓存源、`_noqdq.onnx`、输出 `.bin`）后运行。3 个检测头 conv 的 FP16 `--layerPrecisions` 不用动 —— 那些节点名与类别数无关。

**⑧ Jetson —— 用正确的类别数构建**（nc 是构建参数，无需改源码）：

```bash
NUM_CLASSES=<nc> bash src/matx_reformat/build_matx_reformat.sh   # 重编 matx 库
make clean && make NUM_CLASSES=<nc>
```

`NUM_CLASSES` 默认 80（仓库自带的 COCO 模型）。它驱动缓冲区尺寸、decode 调用点和 CHW16/CHW32 重排的分组维度（`YOLO_NUM_CLASSES` 宏，见 [src/yolov5.cpp](src/yolov5.cpp) 与 [matx_reformat.cu](src/matx_reformat/matx_reformat.cu)；`decode_nms.cu` 是参数化的）。**该参数必须与 `--engine`/`ENGINE=` 传入的 loadable 匹配** —— 不匹配会静默产生乱码级结果（症状：mAP ≈ 0.01、目标检不出）。另外：仅当 ② 替换过锚点才需改 yolov5.cpp 的 `anchors[]`；仅当 ⑥ 发现值不同才需改 `mInputScale`。

**⑨ Jetson —— 运行验证**（检测结果画框写入 `result.jpg`）：

```bash
./build/cudla_yolov5_app --engine data/loadable/mydata.int8...bin --image your.jpg --backend cudla_int8
# 或：make run ENGINE=... IMAGE=...
```

精度验收：以服务器端 `val.py` 的 mAP 为准（test_coco_map.py 仅适用于 COCO）。若在自有数据图像上检测框系统性错位，对比 checkpoint 的 `model.model[-1].anchors` 与 yolov5.cpp 的 `anchors[]`。

仅供快速实验的偷懒方案：把自定义类映射到 COCO 未用的槽位（保持 80 类头）—— C++ 零改动，但头层浪费算力。

**系统重装事件（2026-08-28）：** 根文件系统被重刷（/home 保留）。通过 NVIDIA Jetson 源用 apt 恢复（`/etc/apt/sources.list.d/nvidia-l4t-apt-source.list` 默认所有 `deb` 行都被注释，需先启用；之后会自动跳 `.cn` 镜像，无需代理）：`libnvinfer10`（**不是** `libnvinfer8` —— TRT 10 改了包名）+ `libnvinfer-dev` + `libnvinfer-bin` 恢复 trtexec；`libopencv`（JetPack 版 OpenCV 4.8 运行库就叫这个名字；只装 `libopencv-dev` 会得到一堆指向 `*.so.408` 的悬空软链）。pycocotools 用 pip 重装。顺手把 [src/yolov5.h](src/yolov5.h) 里遗留的 `NvInfer.h`/`NvInferPlugin.h` include 删了 —— 应用未使用任何 TensorRT 符号，编译从此不依赖 TRT 头文件。

## 架构

流水线：CPU（OpenCV 解码 + letterbox）→ GPU（MatX 将 FP32 重排为 DLA 输入格式）→ **DLA（cuDLA 推理）** → GPU（MatX 重排为 FP16 平面格式 + `decode_nms.cu` 中的解码/NMS）→ CPU（bbox 结果）。详见 [src/README.md](src/README.md)。

- [src/validate_coco.cpp](src/validate_coco.cpp) —— `main()` 与命令行解析。尽管名字叫 validate_coco，它才是程序入口，同时处理单图推理和 COCO 验证两种流程。
- [src/yolov5.cpp](src/yolov5.cpp) / [yolov5.h](src/yolov5.h) —— 流水线调度核心。分配 CUDA 缓冲区、持有 cuDLA 上下文、驱动前后处理。`mInputScale`（在 yolov5.cpp:239 用于把 FP32 输入量化为 INT8）来自 `data/model/qat2ptq.cache` 的 `images:` 条目 —— 换新校准缓存时若输入 scale 变化需同步更新。注意 `mOutputScale1-3` 声明了但**从未使用**（死代码 —— DLA 输出是 FP16，直接使用）。网络输入固定为 1x3x672x672。
- **两个互斥的 cuDLA 上下文实现**（在 `yolov5.cpp` 中编译期选择；对比与选型指南见下文「混合模式与独立模式的选择」一节）：
  - [src/cudla_context_hybrid.cpp](src/cudla_context_hybrid.cpp) —— 混合模式：CUDA 分配的缓冲区通过 `cudlaMemRegister` 注册到 cuDLA；任务在 CUDA stream 上提交。是集成最简单的路径。
  - [src/cudla_context_standalone.cpp](src/cudla_context_standalone.cpp) —— 独立模式：使用 NvSciBuf/NvSciSync 管理缓冲区和 fence，并作为外部内存/信号量导入 CUDA。使 DLA 路径不依赖 CUDA context 的创建；确定性信号量变体（`USE_DETERMINISTIC_SEMAPHORE`）是对较老 DriveOS/JetPack 上 NvSciSync 行为的变通方案。
  - 两个上下文类都刻意保持自包含（不依赖示例中的其他代码），方便用户直接拷贝到自己项目中使用 —— 修改时请保持这一特性。
- [src/matx_reformat/](src/matx_reformat/) —— 封装 MatX 子模块的独立 CMake 库（pimpl 模式：`ReformatRunner`）。负责 DLA 张量布局与平面格式之间的转换：`ReformatImage`/`ReformatImageV2`（输入 CHW→HWC4/CHW16）、`Run`/`Transpose`（输出 CHW16→平面格式，对应 stride 8/16/32 的三个 YOLOv5 检测头）。
- [data/model/](data/model/) —— 用 trtexec 将两个 ONNX 模型编译为 DLA loadable 的脚本。INT8 loadable 使用 `--inputIOFormats=int8:dla_hwc4 --outputIOFormats=fp16:chw16 --calib=qat2ptq.cache`，并将最后的检测头卷积强制为 FP16（`--layerPrecisions`）。`build_dla_standalone_loadable_v2.sh` 将更多层回退到 FP16（精度更高、速度更慢）。需要 trtexec 支持 `--buildDLAStandalone` 参数（TRT 8.5 / JetPack 6.0 之前版本需打 `data/trtexec-dla-standalone-trtv8.5.patch` 补丁）。
- [export/](export/) —— 训练侧工具链，与 C++ 程序相互独立：`yolov5-qat/` 需拷贝到 ultralytics yolov5 v7.0 的 checkout 中进行 QAT 微调；`qdq_translator/` 将 QAT ONNX（含 Q/DQ 节点）转换为 PTQ ONNX + INT8 校准缓存。服务器端工作副本在 `yolov5_dla/`（v7.0 + 覆盖层，并工程化强化：CLI 参数化的 `qat.py`、torch 2.x amp 兼容、自定义数据集的 pycocotools 支持 —— 详见其自带 CLAUDE.md）。覆盖层恰好 4 个文件：3 个新增（`quantization/quantize.py`、`quantization/rules.py`、`scripts/qat.py`）+ 1 个修改（`models/common.py`）。

**`yolov5-qat` 覆盖层对 `models/common.py` 的改动**（唯一被修改的文件，与上游 v7.0 差异 ~35 行）：`C3TR`、`C3`、`SPP`、`SPPF`、`Focus`、`GhostConv`、`Classify` 中所有函数式 `torch.cat(..., 1)` 都改为经由 `self.concat = Concat(1)` 子模块调用。原因：`quantization/quantize.py` 的 `initialize()` 在 `--all-node-with-qdq`（[export/README.md](export/README.md) 的 Option#2）时会把 `models.common.Concat → QuantConcat`（以及 `nn.SiLU → QuantSiLU`）注册进 pytorch-quantization 的模块替换表 —— 这是按"模块类"做的替换，只有 Concat 是模块才能被换掉，函数式 `torch.cat` 永远无法替换。而 DLA 要求图中每个算子（包括 Concat）都有 INT8 scale（GPU 上 TensorRT 允许 concat 以更高精度运行；Option#1 路径下 concat 没有训练出的 scale，qdq_translator 的 `--infer_concat_scales` 就是为它服务的）。该改动纯属结构性：数值结果与导出的 ONNX 图完全不变。

DLA I/O 格式限制（这正是存在 MatX 重排步骤的原因）：INT8 输入必须为 `kDLA_LINEAR`/`kDLA_HWC4`/`kCHW32`，FP16 输入输出为 `kCHW16`；本示例采用 INT8 HWC4 输入 + FP16 CHW16 输出。

## 混合模式与独立模式的选择

切换方法见「构建与运行」（编译期开关 + `make clean`）。模式不影响检测结果、精度和 loadable —— 改变的只是 DLA 任务的内存/提交/同步路径。

容易误解的关键事实：

- **两种模式下模型推理都运行在 DLA 硬件上** —— GPU 从不执行卷积。"独立模式"指的是 DLA 的*提交路径*绕开 CUDA（用 NvSciBuf/NvSciSync 取代 CUDA 内存/stream），不是应用不再用 GPU。
- 本示例的 MatX 前后处理重排 kernel 在**两种模式**下都跑在 GPU 上（合计 ~1.6ms，见 `yolov5::infer()`），差异只在 `cudla_context_*.cpp`。想要彻底不占 GPU 的流水线，需把前后处理挪到别处（例如直接喂 FP16 输入、省掉 INT8 量化重排那一步）。

| | 混合模式 | 独立模式 |
|---|---|---|
| 缓冲区 | `cudaMalloc` + `cudlaMemRegister` | NvSciBuf 作为外部内存导入 CUDA |
| 同步 | CUDA stream | NvSciSync fence ↔ CUDA 外部信号量 |
| CUDA context | 必须有（构造函数先 `cudaFree(0)` 创建） | DLA 路径不需要 |
| 集成复杂度 | 低（~200 行自包含类） | 高（NvSci 属性列表、reconcile、导入导出） |

**选混合模式**：CUDA 中心的应用快速集成 —— 单进程、结果在 GPU 上后处理、DLA 任务与其他 CUDA 工作在同一 stream 上自然有序。**选独立模式**：GPU 负载重、需要 DLA 与 CUDA stream 调度彻底解耦；多进程且不想额外创建 CUDA context；或要搭零拷贝 NvSci 流水线（摄像头/NvMedia → DLA → 其他模块）。

资源占用角度：混合模式下推理本体几乎不占 GPU SM 算力，但需要一份 CUDA context（几十 MB）且 DLA 提交与 stream 调度耦合。AGX Orin 有 2 个 DLA 核心，独立时钟/供电域，与 GPU 物理并行。参考性能（README，bs=1）：同一 YOLOv5s INT8 在 GPU 上 1.82ms、DLA 上 3.82ms —— DLA 的价值在于能效和把 GPU 腾给其他任务，而不是绝对速度。
