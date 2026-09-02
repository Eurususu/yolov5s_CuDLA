# CLAUDE.md（中文版）

本文件为 Claude Code (claude.ai/code) 在此仓库中工作时提供指导。英文原版见 [CLAUDE.md](CLAUDE.md)。

## 项目概述

NVIDIA 官方示例：将 QAT（量化感知训练）后的 YOLOv5s 部署到 Orin DLA（深度学习加速器）上，通过 cuDLA 同时演示 cuDLA **混合模式（hybrid mode）** 与 cuDLA **独立模式（standalone mode）** 两种用法。只能在 aarch64 Tegra 硬件（Jetson Orin / DriveOS）上运行 —— nvcc 编译目标为 `sm_87`，并链接 `libcudla`、`libnvscibuf`、`libnvscisync` 以及 TensorRT 的 `nvinfer`。

## 运行环境（本机）

- **硬件：** NVIDIA Jetson AGX Orin 开发套件 —— 64GB 统一内存，2 个 DLA 核心（独立时钟/供电域，与 GPU 物理并行）。
- **系统：** Ubuntu 22.04.5 LTS，L4T R36.5.2 = JetPack 6.2.3，内核 `5.15.185-tegra`，aarch64。
- **工具链：** CUDA 12.6（`/usr/local/cuda`，nvcc 目标 sm_87）、gcc 11.4.0、cmake 3.22.1。
- **库：** TensorRT 10.3.0.30（`libnvinfer-dev`，trtexec 位于 `/usr/src/tensorrt/bin/trtexec`）、cuDLA（`/usr/local/cuda/lib64/libcudla.so`）、OpenCV 4.8.0（apt 版）、`nvidia-l4t-nvsci` 36.5.2 —— 仅运行库、无开发头文件（因此才有 [compat/nvsci-headers/](compat/nvsci-headers/)）。
- **Python：** 3.10.12，已装 pycocotools（mAP 评估就绪）。**PyTorch 2.5.0a0+nv24.08（JetPack 6.1 构建）已安装** —— 位于 `/media/data/jia/pylib`（经 `~/.local/lib/python3.10/site-packages/torch_nvme.pth` 注册，任何 `python3` 直接 `import torch`），配源码编译的 torchvision 0.20.0（含 CUDA 算子）。可在本机跑 `yolov5_dla` 的评估（val / eval_pt_coco.py）。安装过程见「问题记录」的"Jetson 上的 PyTorch"条目。
- **数据：** COCO val2017 在 `/media/data/jia/coco`（`data/coco` 软链指向它）；自定义三分类数据集在 `/media/data/jia/3classes`。均被 gitignore。
- **网络：** GitHub 仅能通过局域网代理 `http://192.168.11.61:7890` 稳定访问。
- **Python（export 工具链）：** 已完整安装并验证 —— 安装命令见「线路 A」，踩坑记录见「问题记录」。`torch` 未安装（仅 QAT 微调需要）。
- **重型产物（`*.onnx`、`*.pt`、loadable、数据集）均被 gitignore** —— 新克隆的仓库需先拷入，或按下面两条流水线重新生成。

**系统重装事件（2026-08-28 / 09-01）：** 第一台 Orin 重刷过根文件系统（/home 保留），第二台 Orin 的镜像缺的是同一批包。完整的新机器依赖清单与一键安装命令见「问题记录」的"全新/重刷机器"条目：TRT（`libnvinfer10`，TRT 10 改了包名）、`nvidia-l4t-dla-compiler`（trtexec 的 DLA 构建需要 `libnvdla_compiler.so`）、JetPack OpenCV 4.8（`libopencv` 运行库 + `libopencv-dev`）、`libjsoncpp-dev`，装完跑一次 `sudo ldconfig`。顺手把 [src/yolov5.h](src/yolov5.h) 里遗留的 `NvInfer.h`/`NvInferPlugin.h` include 删了 —— 应用未使用任何 TensorRT 符号，编译从此不依赖 TRT 头文件。

## 架构

流水线：CPU（OpenCV 解码 + letterbox）→ GPU（MatX 将 FP32 重排为 DLA 输入格式）→ **DLA（cuDLA 推理）** → GPU（MatX 重排为 FP16 平面格式 + `decode_nms.cu` 中的解码/NMS）→ CPU（bbox 结果）。详见 [src/README.md](src/README.md)。

- [src/validate_coco.cpp](src/validate_coco.cpp) —— `main()` 与命令行解析。尽管名字叫 validate_coco，它才是程序入口，同时处理单图推理和验证两种流程。图片列表经 `--list` 传入（默认 `./data/coco_val_2017_list.txt`）；`NUM_CLASSES != 80` 时预测自动采用字符串文件名 image id + 恒等类别编码（与 `make_coco_json.py` 的 GT 编码对齐）。
- [src/yolov5.cpp](src/yolov5.cpp) / [yolov5.h](src/yolov5.h) —— 流水线调度核心。分配 CUDA 缓冲区、持有 cuDLA 上下文、驱动前后处理。`mInputScale`（yolov5.cpp:239 用于把 FP32 输入量化为 INT8）来自校准缓存的 `images:` 条目 —— 换新缓存时若值变化需同步更新。`mOutputScale1-3` 声明了但**从未使用**（死代码 —— DLA 输出是 FP16，直接消费）。网络输入固定为 1x3x672x672。类别数来自 `YOLO_NUM_CLASSES` 编译宏（`make NUM_CLASSES=<n>`，默认 80）。
- **两个互斥的 cuDLA 上下文实现**（在 `yolov5.cpp` 中编译期选择；对比与选型见下文「混合模式与独立模式的选择」）：
  - [src/cudla_context_hybrid.cpp](src/cudla_context_hybrid.cpp) —— 混合模式：CUDA 分配的缓冲区通过 `cudlaMemRegister` 注册到 cuDLA；任务在 CUDA stream 上提交。是集成最简单的路径。
  - [src/cudla_context_standalone.cpp](src/cudla_context_standalone.cpp) —— 独立模式：使用 NvSciBuf/NvSciSync 管理缓冲区和 fence，并作为外部内存/信号量导入 CUDA。使 DLA 路径不依赖 CUDA context 的创建；确定性信号量变体（`USE_DETERMINISTIC_SEMAPHORE`）是对较老 DriveOS/JetPack 上 NvSciSync 行为的变通方案。
  - 两个上下文类都刻意保持自包含（不依赖示例中的其他代码），方便用户直接拷贝到自己项目中使用 —— 修改时请保持这一特性。
- [src/matx_reformat/](src/matx_reformat/) —— 封装 MatX 子模块的独立 CMake 库（pimpl 模式：`ReformatRunner`）。负责 DLA 张量布局与平面格式之间的转换：`ReformatImage`/`ReformatImageV2`（输入 CHW→HWC4/CHW16）、`Run`/`Transpose`（输出 CHW16→平面格式，对应 stride 8/16/32 的三个 YOLOv5 检测头）。通道几何由同一个 `YOLO_NUM_CLASSES` 宏推导（CHW16/CHW32 会把检测头通道补齐到 16/32 的倍数）。
- [data/model/](data/model/) —— 用 trtexec 将 ONNX 模型编译为 DLA loadable 的脚本。INT8 loadable 使用 `--inputIOFormats=int8:dla_hwc4 --outputIOFormats=fp16:chw16`，并将最后的检测头卷积强制为 FP16（`--layerPrecisions`）。`build_dla_standalone_loadable_v2.sh` 将更多层回退到 FP16（精度更高、速度更慢）。需要 trtexec 支持 `--buildDLAStandalone` 参数（`data/trtexec-dla-standalone-trtv8.5.patch` 补丁仅 TRT 8.5 / JetPack 6.0 之前需要）。
- [export/](export/) —— 训练侧工具链，与 C++ 程序相互独立：`yolov5-qat/` 是面向 ultralytics yolov5 v7.0 checkout 的原始覆盖层；`qdq_translator/` 将 QAT ONNX（含 Q/DQ 节点）转换为 PTQ ONNX + INT8 校准缓存。服务器端工作副本在 `yolov5_dla/`（v7.0 + 覆盖层，并工程化强化：CLI 参数化的 `qat.py`、torch 2.x amp 兼容、自定义数据集 pycocotools 支持 —— 详见其自带 CLAUDE.md）。覆盖层恰好 4 个文件：3 个新增（`quantization/quantize.py`、`quantization/rules.py`、`scripts/qat.py`）+ 1 个修改（`models/common.py`）。

**`yolov5-qat` 覆盖层对 `models/common.py` 的改动**（唯一被修改的文件，与上游 v7.0 差异 ~35 行）：`C3TR`、`C3`、`SPP`、`SPPF`、`Focus`、`GhostConv`、`Classify` 中所有函数式 `torch.cat(..., 1)` 都改为经由 `self.concat = Concat(1)` 子模块调用。原因：`quantization/quantize.py` 的 `initialize()` 在 `--all-node-with-qdq`（[export/README.md](export/README.md) 的 Option#2）时会把 `models.common.Concat → QuantConcat`（以及 `nn.SiLU → QuantSiLU`）注册进 pytorch-quantization 的模块替换表 —— 这是按"模块类"做的替换，只有 Concat 是模块才能被换掉，函数式 `torch.cat` 永远无法替换。而 DLA 要求图中每个算子（包括 Concat）都有 INT8 scale（GPU 上 TensorRT 允许 concat 以更高精度运行；Option#1 路径下 concat 没有训练出的 scale，qdq_translator 的 `--infer_concat_scales` 就是为它服务的）。该改动纯属结构性：数值结果与导出的 ONNX 图完全不变。

DLA I/O 格式限制（这正是存在 MatX 重排步骤的原因）：INT8 输入必须为 `kDLA_LINEAR`/`kDLA_HWC4`/`kCHW32`，FP16 输入输出为 `kCHW16`；本示例采用 INT8 HWC4 输入 + FP16 CHW16 输出。

## 混合模式与独立模式的选择

模式是**编译期**开关 —— 切换必须先 `make clean`（Makefile 不跟踪该依赖）：

```bash
make clean && make run                                    # 混合模式（默认）
make clean && make run USE_DLA_STANDALONE_MODE=1          # 独立模式
make clean && make run USE_DLA_STANDALONE_MODE=1 USE_DETERMINISTIC_SEMAPHORE=1   # 仅老 DriveOS/JetPack
```

模式不影响检测结果、精度和 loadable —— 改变的只是 DLA 任务的内存/提交/同步路径。

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

## 线路 A —— COCO：QAT 训练 → DLA 部署

原始示例的路线：官方 yolov5s 在 COCO 上做 QAT 微调，INT8 部署到 DLA，用 COCO val2017 验证。部署包含单图推理与验证集评测。

### A0. Jetson 端一次性构建

```bash
# 重型输入被 gitignore —— 先把 data/model/*.onnx 拷入（原 NVIDIA 仓库经 git-LFS 发布）
bash data/model/build_dla_standalone_loadable.sh   # trtexec → INT8 + FP16 loadable 到 data/loadable/
bash src/matx_reformat/build_matx_reformat.sh      # matx 库（CCCL shim 加持下离线可编；需要 GitHub 时自动走代理）
make                                               # 主程序；COCO 用默认 NUM_CLASSES=80
```

matx 单元测试：在 `src/matx_reformat/build/` 下运行 `./test`（该目录需在 `LD_LIBRARY_PATH`）。主程序本身不需要任何环境变量 —— 链接时已烧入 `-Wl,-rpath` 指向 `src/matx_reformat/build`。

### A1. 运行与验证（单图 / COCO 验证集）

```bash
make run                                # 单张图 → 检测框画到 result.jpg
make validate_cudla_int8                # COCO val2017（5000 张）+ pycocotools mAP，约 30~60 分钟
make validate_cudla_int8 ENGINE=data/loadable/<其他80类loadable>.bin

# 或直接运行二进制：
./build/cudla_yolov5_app --engine data/loadable/yolov5.int8.int8hwc4in.fp16chw16out.standalone.bin \
    --image data/images/image.jpg --backend cudla_int8          # 单图
./build/cudla_yolov5_app --engine ... --coco_path data/coco/ --backend cudla_int8   # 验证（生成 predict.json）
python3 test_coco_map.py --predict predict.json --coco data/coco/
```

参考结果：mAP50-95 **37.5**（DLA FP16）/ **37.1**（DLA INT8 QAT）@ 1x3x672x672，INT8 约 5.5ms/图。

### A2. 服务器 —— COCO 的 QAT 微调

**只需拷贝 `yolov5_dla/` 目录**到 GPU 服务器（v7.0 + 工程化 QAT 层；其自带 CLAUDE.md 记录内部机制）。**权重警告：** 用 `yolov5s.pt`（v7.0），不要用 `yolov5su.pt` —— 'u' 版锚点与 src/yolov5.cpp 硬编码的 v7.0 默认锚点不同，会静默导致解码错乱。

```bash
cd yolov5_dla
pip install -r requirements.txt
pip install --no-deps --index-url https://pypi.nvidia.com pytorch-quantization && pip install absl-py prettytable
bash data/scripts/get_coco.sh           # 生成 YOLO 格式 COCO → ../datasets/coco（约 20GB）

# Option#1（仓库自带模型的路线）；Option#2 加 --all-node-with-qdq
python scripts/qat.py quantize yolov5s.pt --ptq=ptq.pt --qat=qat.pt \
    --cocodir=../datasets/coco --eval-origin --eval-ptq
python scripts/qat.py export qat.pt --size=672 --save=yolov5_trimmed_qat.onnx --dynamic --noanchor
```

脚本流程：校准 → 打印 Origin/PTQ 基线 mAP → 微调，每个 epoch 评测并保存 mAP 最优的 `qat.pt`（历史在 `summary.json`）；`--iters` 可限制每 epoch 的 batch 数用于快速试跑；完整 COCO 训练单卡需数小时。注意 `--cocodir`（给 dataloader 用）与 `data/coco.yaml` 的 `path:`（给评测用）必须指向同一份数据集。

### A3. Jetson —— 转化 → loadable → 验证（重训练闭环）

```bash
# 一次性安装 translator 依赖（无需 torch）
pip3 install -i https://pypi.tuna.tsinghua.edu.cn/simple onnx onnx_graphsurgeon nvidia-pyindex
pip3 install -i https://pypi.tuna.tsinghua.edu.cn/simple onnxoptimizer   # 无 aarch64 wheel → 源码编译约 15 分钟

# 转化：QAT ONNX（显式 Q/DQ）→ PTQ ONNX + INT8 校准缓存 + 精度配置
cd export/qdq_translator
python3 qdq_translator.py --input_onnx_models=../../data/model/yolov5_trimmed_qat.onnx \
    --output_dir=../../data/model/ --infer_concat_scales --infer_mul_scales
```

之后：复制 [build_dla_standalone_loadable_8.26.sh](data/model/build_dla_standalone_loadable_8.26.sh) 作模板，改 3 处路径（缓存源、`_noqdq.onnx`、输出 `.bin`）后运行；3 个检测头 conv 的 FP16 `--layerPrecisions` 不用动。用新缓存的 `images:` 十六进制条目（大端 IEEE-754）对比 src/yolov5.cpp 的 `mInputScale`，不同则更新。最后 `make` + `make validate_cudla_int8 ENGINE=...`。

参考：仓库保留的 8.26 重训练闭环（`yolov5_trimmed_qat_8.26.*` → `yolov5_8.26.int8...bin`）：COCO mAP50-95 = **37.1**（与原版持平），输入 scale 逐位相同，`layer_arg.txt` 为空（那是"最大 FP16 建议清单"，不是必需清单）。

## 线路 B —— 自定义数据集：训练 → QAT → DLA 部署

从自有数据集到 DLA 推理的端到端流程。仓库内实例：三分类模型（`data/model/yolov5_3clases_qat*`，2026-08-28 验证：8 个检测 @ 3.74ms/图）。

**①–⑤ 服务器 —— 数据准备、FP32 训练、QAT 微调、ONNX 导出。** 这些步骤在 `yolov5_dla` 工具包内完成；详细命令（环境搭建、单卡/多卡训练、可选的 COCO 格式 GT 生成、PTQ+QAT、导出）见 [yolov5_dla/CLAUDE.zh-CN.md](yolov5_dla/CLAUDE.zh-CN.md) —— 按那份文档执行，最后**只把导出的 `.onnx`** 拷回 Jetson 的 `data/model/`。两个部署侧关键点：训练用 `--imgsz 672`（对齐部署端固定输入）；盯 autoanchor 日志 —— 若训练替换了锚点，需同步进 src/yolov5.cpp 的 `anchors[]`（第 ⑧ 步）。

**⑥ Jetson —— 转化**（图级操作，与数据集无关；依赖安装与参数同 A3）：

```bash
cd export/qdq_translator
python3 qdq_translator.py --input_onnx_models=../../data/model/mydata_qat.onnx \
    --output_dir=../../data/model/ --infer_concat_scales --infer_mul_scales
```

然后用新缓存的 `images:` 十六进制条目对比 src/yolov5.cpp 的 `mInputScale`，不同则更新。`layer_arg.txt` 为空是正常的。

**⑦ Jetson —— 构建 loadable。** 复制 [build_dla_standalone_loadable_3classes.sh](data/model/build_dla_standalone_loadable_3classes.sh)，改 3 处路径（缓存源、`_noqdq.onnx`、输出 `.bin`）后运行。3 个检测头 conv 的 FP16 `--layerPrecisions` 不用动（节点名与类别数无关）。

**⑧ Jetson —— 用正确的类别数构建**（nc 是构建参数，无需改源码）：

```bash
NUM_CLASSES=<nc> [INPUT_H=<h> INPUT_W=<w>] bash src/matx_reformat/build_matx_reformat.sh   # 重编 matx 库
make clean && make NUM_CLASSES=<nc> [INPUT_H=<h> INPUT_W=<w>]
```

`NUM_CLASSES` 默认 80（仓库自带的 COCO 模型）。它驱动缓冲区尺寸、decode 调用点和 CHW16/CHW32 重排的分组维度（`YOLO_NUM_CLASSES` 宏，见 [src/yolov5.cpp](src/yolov5.cpp) 与 [matx_reformat.cu](src/matx_reformat/matx_reformat.cu)；`decode_nms.cu` 是参数化的）。`INPUT_H`/`INPUT_W`（均须为 32 的倍数）默认 672×672，经 `YOLO_INPUT_H/W` 宏驱动全部派生几何（检测头网格、锚点总数、letterbox 画布），必须与 loadable 的导出尺寸完全一致 —— 720p：导出用 `--size=736x1280`、构建用 `INPUT_H=736 INPUT_W=1280`，构建脚本为 [build_dla_standalone_loadable_3classes_720p.sh](data/model/build_dla_standalone_loadable_3classes_720p.sh)（已端到端验证：INT8 5.5ms @ 12 目标、FP16 13 目标）。**该参数必须与 `--engine`/`ENGINE=` 传入的 loadable 匹配** —— 不匹配会静默产生乱码级结果。另外：仅当 ② 替换过锚点才需改 yolov5.cpp 的 `anchors[]`；仅当 ⑥ 发现值不同才需改 `mInputScale`。

**⑨ Jetson —— 运行与验证：**

```bash
# 单图（检测框画到 result.jpg）
./build/cudla_yolov5_app --engine data/loadable/mydata.int8...bin --image your.jpg --backend cudla_int8
# 或：make run ENGINE=... IMAGE=...
```

精度验收 —— 三种方式：服务器端 `val.py` 的 mAP、**自定义数据集的设备端 COCO 式评测**（2026-08-31 已验证，三分类模型：mAP50-95 **0.466**，4356 张图）、或用 `yolov5_dla/scripts/eval_pt_coco.py` 做 checkpoint 级评估（本机已装 torch 可直接跑；qat.pt 以伪量化生效方式加载，见脚本头部说明）。**三分类模型的归因基线（同 4356 张、pycocotools）：FP32 0.482 → QAT 0.478 → DLA INT8 0.466** —— 即 QAT 损失 0.4 点、部署损失 1.2 点；yolov5 原生与 pycocotools 之间约 1.8 点的方法学差异（pycocotools 更严格）加上验证集差异，就构成了与训练日志对比时出现的"大差距"。注意：yolov5 的 dataloader 按当前工作目录解析图片列表条目 —— `eval_pt_coco.py` 会自动把相对条目改写为绝对路径（相对 `--cocodir` 解析，生成 `*_abs.txt`），因此同一份相对列表在所有环节通用：

```bash
# 从 YOLO txt 标签生成 GT json（无需 torch）
python3 yolov5_dla/scripts/make_coco_json.py --cocodir /path/to/ds --data <yaml> --val-list <list.txt>
# 在验证列表上推理（nc != 80 时预测自动切换为字符串 id + 恒等类别编码）
./build/cudla_yolov5_app --engine ... --coco_path /path/to/ds --list /path/to/ds/<list.txt> --backend cudla_int8
python3 test_coco_map.py --predict predict.json --coco /path/to/ds
```

若在自有数据图像上检测框系统性错位，对比 checkpoint 的 `model.model[-1].anchors` 与 yolov5.cpp 的 `anchors[]`。

仅供快速实验的偷懒方案：把自定义类映射到 COCO 未用的槽位（保持 80 类头）—— C++ 零改动，但头层浪费算力。

## 问题记录

这些机器上实际踩过的问题，按"症状 → 原因 → 解法"记录。

**构建（Jetson）**

- trtexec INT8 构建失败：*"Calibration table does not match calibrator algorithm type"* 随后 *"Tensor `images` is bound to nullptr"* —— TRT 10.x 拒绝仓库的 `TRT-8600-EntropyCalibration2` 缓存头并在无数据下尝试重新校准。解法：构建脚本先 `sed` 把头部版本改写为本机 TRT 版本再传 `--calib`。
- trtexec 日志出现 `kPREFER_PRECISION_CONSTRAINTS cannot be set if kOBEY_PRECISION_CONSTRAINTS is set` —— TRT 10.3 无害噪音，obey 约束仍生效，构建会成功。
- MatX 配置阶段克隆 `libcudacxx` 失败 —— 直连 GitHub 不稳定；`build_matx_reformat.sh` 会在直连探测失败时自动走局域网代理。有 CCCL shim 后构建完全不需要联网。
- MatX 0.4.1 与 CUDA 12.6 冲突：NVTX v1/v3 `#error`、找不到 `<__config>` —— MatX 拉取的 libcudacxx 2.1.0 与工具链 CCCL 打架。解法：[src/matx_reformat/compat/](src/matx_reformat/compat/) 垫片（`libcudacxx-shim/include` 软链到 `/usr/local/cuda/include`；`nvtx-shim/nvToolsExt.h` 重定向到 nvtx3 兼容层）。若 shim 之前拉取失败过，先 `rm -rf src/matx_reformat/build`。
- `nvscibuf.h: No such file or directory` —— JetPack 只带 NvSci 运行库无开发头文件。解法：头文件已从 DRIVE OS 6.0.9 公开文档的 doxygen `_source.html` 页面还原到 [compat/nvsci-headers/](compat/nvsci-headers/)（Makefile 加了 `-I`）。
- `ld: cannot find -lnvscibuf` —— NvSci 库在 `/usr/lib/aarch64-linux-gnu/nvidia/`；Makefile 已加 `-L`（仅独立模式链接）。
- 直接运行二进制报 `libmatx_reformat.so: cannot open shared object file` —— Makefile 的 `export LD_LIBRARY_PATH` 只对 make 子进程生效。解法：链接已烧入 `-Wl,-rpath`，无需环境变量。
- 全新/重刷机器 —— 先启用 `/etc/apt/sources.list.d/nvidia-l4t-apt-source.list` 里被注释的 `deb` 行（之后自动跳 `.cn` 镜像，无需代理），再一次性装齐全部依赖：

  ```bash
  sudo apt update && sudo apt install -y \
      libnvinfer10 libnvinfer-dev libnvinfer-bin \
      nvidia-l4t-dla-compiler \
      libopencv libopencv-dev \
      libjsoncpp-dev
  sudo ldconfig
  ```

  各包的作用：`libnvinfer10`（TRT 10 改了包名，没有 `libnvinfer8`）+ `libnvinfer-bin` = trtexec；`nvidia-l4t-dla-compiler` 提供 `libnvdla_compiler.so` —— 缺了它 trtexec 的 DLA 构建启动即死（*"Unable to open library: libnvinfer_plugin.so.10 due to libnvdla_compiler.so"*）；`libopencv` 是 JetPack OpenCV 4.8 运行库本体（只装 `libopencv-dev` 会留悬空 `/usr/lib/libopencv_*.so → *.so.408` 软链 → 链接失败）；`libjsoncpp-dev` = `json/json.h`。若装完仍报 "cannot open shared object file"，执行 `sudo ldconfig`（NVIDIA 库目录 `/usr/lib/aarch64-linux-gnu/nvidia` 已登记在 `nvidia-tegra.conf`，但 apt 不总是自动刷新缓存）。pycocotools 用 pip 装。

**Python 环境**

- **Jetson 上的 PyTorch**（2026-09-01 完成，根分区只剩约 7GB → 全部装到 NVMe）：Jetson 版 torch wheel **不在** `pypi.nvidia.com` —— 官方构建在 NVIDIA 的 redist 仓库：
  ```bash
  # 1. torch（wheel 文件名必须保持原样 —— 重命名会破坏 pip 的标签解析）
  curl -L -o torch-2.5.0a0+872d972e41.nv24.08.17622132-cp310-cp310-linux_aarch64.whl \
    "https://developer.download.nvidia.com/compute/redist/jp/v61/pytorch/torch-2.5.0a0+872d972e41.nv24.08.17622132-cp310-cp310-linux_aarch64.whl"   # 约770MB；.cn 镜像限速时走代理
  python3 -m pip install --no-cache-dir --target /media/data/jia/pylib <wheel>   # 依赖走国内镜像
  echo /media/data/jia/pylib > ~/.local/lib/python3.10/site-packages/torch_nvme.pth
  # 2. torchvision —— PyPI 的 wheel 与 nv 版 torch ABI 不兼容（"operator torchvision::nms does not exist"）→ 源码编译：
  curl -L -o vision-0.20.0.tar.gz https://github.com/pytorch/vision/archive/refs/tags/v0.20.0.tar.gz   # 走代理
  tar xf vision-0.20.0.tar.gz && cd vision-0.20.0
  TORCH_CUDA_ARCH_LIST="8.7" FORCE_CUDA=1 MAX_JOBS=4 \
    python3 -m pip install --no-deps --no-build-isolation --target /media/data/jia/pylib .   # 约 10 分钟
  # 3. 缺库：torch 导入报 libcusparseLt.so.0 找不到 →
  python3 -m pip install --no-deps --target /media/data/jia/pylib nvidia-cusparselt-cu12==0.6.3
  ln -s /media/data/jia/pylib/cusparselt/lib/libcusparseLt.so.0 /media/data/jia/pylib/torch/lib/
  # 4. numpy 必须保持 <2（nv 构建按 numpy 1.x 编译；新版 torchvision 的 PyPI 源码包不存在 —— 所以从 GitHub 取）
  python3 -m pip install "numpy==1.26.4" "opencv-python==4.10.0.84"   # opencv 5.x 会强制 numpy>=2 —— 锁版本
  # 5. yolov5 运行时补充依赖：tqdm ipython requests（系统的 pandas/matplotlib 在 numpy 1.26 下可用）
  ```

- `pip install pytorch-quantization` 报占位包错误 / `sphinx-glpi-theme` 无法解析 —— README 的 `pypi.ngc.nvidia.com` 源已废弃、PyPI 有同名占位包、NVIDIA 官方包错标文档主题为运行时依赖。解法：`pip install --no-deps --index-url https://pypi.nvidia.com pytorch-quantization && pip install absl-py prettytable`。
- `onnxoptimizer` 无 aarch64 wheel —— pip 自动转源码编译，约 15 分钟静默无输出（系统 cmake 3.22 够用）；requirements.txt 的 `==0.3.2` 锁不关键（只用到 4 个 pass，0.3.13 全有）。
- 装 onnx 会把用户级 numpy 升到 2.2.6 —— 已验证 pycocotools 不受影响；若日后报 numpy ABI 错误：`pip install "numpy<2"`。

**训练（服务器）**

- `yolov5su.pt` 能加载但设备端解码乱 —— 'u' 版锚点与 src/yolov5.cpp 硬编码的 v7.0 默认锚点不同。始终从 `yolov5s.pt` 开始。
- 附带打印的 pycocotools AP ≈ 0.001 而 yolov5 自身表格正常 —— 非 COCO 目录名使上游 `save_one_json` 写出连续类别号。yolov5_dla 已根治（`is_coco` 感知、`--save-json` 可选、`make_coco_json.py` 生成 COCO 格式 GT）。
- `--cocodir` 与评测 yaml 的 `path:` 必须指向同一数据集，否则 val 加载不到图片。

**精度 / 运行**

- **mAP ≈ 0.01 且目标检不出**（如 bus 检测不到）—— 程序的 `NUM_CLASSES` 构建值与 loadable 不匹配（80 类模型跑在 3 类构建上踩过）。用匹配的 `NUM_CLASSES` 重编两个库（见线路 B ⑧）。
- `mOutputScale1-3` 看起来需要随重训练更新 —— 它们是死代码；只有 `mInputScale`（缓存 `images:` 条目）被使用。
- 分辨率参数化时 letterbox 行为随之改变：原示例把内容映射进 672 画布中央的 640x640 区域（+16 边框，为匹配 640 训练的模型）；现在改为**全画布** letterbox（匹配按部署尺寸训练的模型及 `rect=True` 验证）。同一模型的检测结果数量可能与旧运行略有差异。
- 程序会把 `--coco_path` 拼接到列表每一行前面 —— 列表条目保持相对路径（现在也支持绝对路径）；非 COCO 列表用 `--list` 传入。
- 自定义数据集评测编码：`coco80_to_coco91_class` 映射仅在 `NUM_CLASSES == 80` 时应用；自定义模型用恒等类别 id + 字符串 image id，与 `make_coco_json.py` 对齐。
