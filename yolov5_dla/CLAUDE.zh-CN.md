# CLAUDE.md（中文版）

本文件为 Claude Code (claude.ai/code) 在本仓库中工作时提供指导。与英文版 `CLAUDE.md` 内容保持同步 —— **英文版有更新时，本文件必须同步更新**。

## 本仓库是什么

Ultralytics YOLOv5 **v7.0**（HEAD 分离于 `v7.0` 标签；`master` 跟踪上游）加上一层本地、**未提交**的 INT8 量化方案（基于 NVIDIA `pytorch_quantization` 的 PTQ/QAT），面向 TensorRT/DLA 部署 —— 即仓库名 `yolov5_dla` 的由来。本分支的改动只存在于工作区（working tree）：

- `quantization/`（新增）—— 量化工具库
- `scripts/qat.py`（新增）—— PTQ/QAT/导出工作流的 CLI 驱动
- `models/common.py`（修改）—— `C3TR`、`C3`、`SPP`、`SPPF`、`Focus`、`GhostConv`、`Classify` 中的 `torch.cat` 调用替换为 `Concat(1)` 子模块，使 concat 算子在导出图中成为可量化模块
- `data/coco.yaml`（修改）—— `path:` 指向机器特定的绝对路径（`/home/jia/dataset/coco`）；需按机器调整

Git 历史完全不反映量化相关的工作 —— 不要用 git 历史来推断这些文件。

`pytorch_quantization` 默认未安装，来自 NVIDIA 的索引：
```bash
pip install nvidia-pyindex && pip install pytorch-quantization
```

## YOLOv5 标准命令

```bash
python train.py --data coco128.yaml --weights yolov5s.pt --epochs 1 --imgsz 640   # 训练（自动下载数据/权重）
python val.py --weights yolov5s.pt --data coco.yaml                              # 验证
python detect.py --weights yolov5s.pt --source data/images/bus.jpg               # 推理
python export.py --weights yolov5s.pt --include onnx                            # 导出（torchscript/onnx/engine/...）
python benchmarks.py --data coco128.yaml --weights yolov5n.pt --img 320         # CI 风格基准测试
python models/yolo.py --cfg yolov5s.yaml                                        # 从 yaml 构建模型
```
`classify/` 和 `segment/` 各有自己的 `train.py`/`val.py`/`predict.py`。快速 CPU 冒烟测试（即 CI 的做法）：加上 `--imgsz 64 --batch 32 --device cpu --epochs 1`。没有 pytest 测试套件；CI（.github/workflows/ci-testing.yml）是功能性的 shell 命令。上游要求 Python >=3.7、torch >=1.7；本容器运行 Python 3.11 + torch 2.x + CUDA。

代码检查/格式化：`pre-commit run --all-files`（isort、yapf、flake8；配置在 `setup.cfg`，最大行长 120）。

## 量化工作流（本分支的核心目的）

所有命令都在仓库根目录运行（`scripts/qat.py` 会把 `.` 插入 `sys.path`，并以 cwd 为基准打开 `data/hyps/...`）。需要数据集放在 `--cocodir`（其中含图像清单 txt）以及 GPU。`--train-list`/`--val-list` 各自可接一个或多个清单文件名（默认 `train2017.txt` / `val2017.txt`）。类别数会自动从 checkpoint 读取；用 `--data` 指向对应的数据集 yaml（默认 `data/coco.yaml`），用 `--imgsz` 对齐训练尺寸（默认 640）。默认使用 YOLOv5 原生 mAP；`--save-json` 会额外跑 pycocotools 评测，需要 yaml 的 `path` 下有 COCO 格式 GT json（`annotations/instances_val2017.json`）——自定义数据集先用 `scripts/make_coco_json.py` 生成（image_id 用字符串文件名 stem、类别 id 恒等映射，与 val.py 预测侧一致——`save_one_json` 已改为感知 `is_coco`：真 COCO 保持 int id + coco91 类映射，非 COCO 统一字符串 id，避免 COCOeval 排序 imgIds 时 int/str 混杂报错）。在自定义数据集上从训练到导出的完整流程（以 3 类数据集 `/root/dataset/3classes` 为例，清单为 `origin_*.txt` / `ship_add_*.txt` / `ship_blank_add_*.txt`，配置 `data/3classes.yaml`）：

```bash
# 1. 训练 —— 双卡 DDP（batch 64 为总量，自动均分到各卡）；单卡则去掉启动器、加 --device 0
python -m torch.distributed.run --nproc_per_node 2 train.py --weights weights/yolov5s.pt \
    --data data/3classes.yaml --imgsz 672 --batch-size 64 --workers 32

# 1b. 中断后继续训练（原有全部参数会从 runs/train/exp2/opt.yaml 自动恢复）
python -m torch.distributed.run --nproc_per_node 2 train.py --resume runs/train/exp2/weights/last.pt

# 2. （仅 --save-json 需要）生成 COCO 格式 GT json → <yaml path>/annotations/instances_val2017.json
python scripts/make_coco_json.py --cocodir /root/dataset/3classes --data data/3classes.yaml \
    --val-list origin_val.txt ship_add_val.txt ship_blank_add_val.txt

# 3. PTQ + QAT —— 替换量化模块、校准、评估 Origin/PTQ、微调（mAP 最优的 checkpoint 存入 qat.pt）
python scripts/qat.py quantize runs/train/exp2/weights/best.pt --cocodir /root/dataset/3classes \
    --data data/3classes.yaml --imgsz 672 \
    --train-list origin_train.txt ship_add_train.txt ship_blank_add_train.txt \
    --val-list origin_val.txt ship_add_val.txt ship_blank_add_val.txt \
    --ptq ptq.pt --qat qat.pt --eval-origin --eval-ptq
#    加 --save-json 可附带 pycocotools AP/AR 日志（需先做第 2 步）

# 4. 导出供 trtexec INT8 的 Q/DQ ONNX —— 原始 s8/s16/s32 检测头输出，动态 batch
# 正方形（向后兼容，纯数字）
python scripts/qat.py export qat_1280.pt --size=672 --save=... --dynamic --noanchor

# 矩形（新增，HxW 格式）
python scripts/qat.py export qat_1280.pt --size=736x1280 --save=yolov5_3clases_qat_720p.onnx --dynamic --noanchor
```

#### 两种导出风味（INT8 路径 vs FP16 路径）

`qat.py export` 服务于两条不同的下游路径 —— 按用途选择输入 checkpoint 和参数：

| | QAT ONNX（INT8 路径） | FP32 trimmed ONNX（FP16 路径） |
|---|---|---|
| 命令 | `export qat.pt --size=672 --save=..._qat.onnx --dynamic --noanchor` | `export runs/.../best.pt --size=672 --save=..._fp32_trimmed.onnx --dynamic --noanchor --noqadd` |
| 输入 checkpoint | `qat.pt` / `ptq.pt` —— 带有已校准 scale 的 Quant* 模块 | `best.pt` —— 纯 FP32 模型，无量化器 |
| 导出的图 | **内嵌 Q/DQ 节点**（显式量化，scale 来自训练校准） | **纯净 FP32 图** —— 零 Q/DQ 节点 |
| 下游消费者 | `qdq_translator.py` → PTQ ONNX + INT8 校准缓存 → trtexec `--int8 --calib` | trtexec 直接 `--fp16` 编译（无需校准缓存） |

- `--noqadd` 只对 FP16 风味重要：默认 `cmd_export` 还会执行 `replace_bottleneck_forward()`，把 Bottleneck 残差加法改经 **QuantAdd**。在 QAT checkpoint 上这些量化器是训练图的一部分、scale 已校准（保留）；而在纯 `best.pt` 上它们会被临时注入、**scale 未校准** → 图里多出约 14 个携带垃圾 scale 的伪 Q/DQ 节点，直接毁掉 FP16 构建。导出未量化的 checkpoint 时务必加 `--noqadd`。
- `quantize --imgsz` 不需要矩形版本：train loader 本来就 letterbox 成正方形（yolov5 训练式），
  val loader 已经保纵横比（`rect=True`，长边 = imgsz）。scale 统计主要由目标尺度分布决定，
  长边匹配（如 736x1280 部署用 `--imgsz 1280`）已覆盖 —— padding 差异对逐张量 scale 影响甚微。
  只有当部署后的量化损失实测异常偏大时，才值得加固定 HxW 的 letterbox 校准。

- 共用参数：`--size=672` = 部署分辨率；`--dynamic` = 动态 batch 轴；`--noanchor` 剥掉检测头的锚点解码（C++ 端用自己的锚点解码；输出原始 s8/s16/s32）。


```bash
# 1.（可选）分析哪些层量化后损失最大
python scripts/qat.py sensitive yolov5s.pt --cocodir datasets/coco

# 2. PTQ：把模块替换为量化对应物，校准（histogram + MSE amax），保存
python scripts/qat.py quantize yolov5s.pt --cocodir datasets/coco --ptq ptq.pt --eval-ptq
#    QAT：加 --qat qat.pt --iters 200（针对冻结的 FP 输出做微调，保留 mAP 最优的 checkpoint）
#    跳过检测头卷积的量化：--ignore-policy "model\.24\.m\.(.*)"
#    也给 SiLU/Concat/Add 插 Q/DQ：--all-node-with-qdq（此时不会应用自定义规则）

# 3. 导出供 trtexec INT8 使用的 Q/DQ ONNX
python scripts/qat.py export qat.pt --save qat.onnx --dynamic
#    DLA 友好的原始检测头输出：--noanchor（输出 s8/s16/s32 而非解码后的 "outputs"）

# 4. 评估 .pt
python scripts/qat.py test yolov5s.pt --cocodir datasets/coco
```

### 各部分如何协作

> 关于 QAT 原理与代码的深度剖析（模块替换 → 校准 → 基于 STE 的蒸馏式微调 → Q/DQ 导出
> 四阶段的逐行讲解），见独立文档 [docs/qat-internals.zh-CN.md](docs/qat-internals.zh-CN.md)。

- `quantize.replace_to_quantization_module()` 按 `_DEFAULT_QUANT_MAP` 遍历，把 `nn.Conv2d`/`MaxPool2d`/`Linear` 换成 `QuantConv2d`/...；`replace_bottleneck_forward()` 给 `Bottleneck.forward` 打补丁，让残差加法经过 `QuantAdd`（加法算子需要显式的输入量化器）。顺序很重要：`replace_bottleneck_forward` → `replace_to_quantization_module` →（除非 `--all-node-with-qdq`，否则 `apply_custom_rules_to_quantizer`）→ `calibrate_model`。
- `quantization/rules.py` 解析中间 ONNX 图，找到喂入同一个 Concat/MaxPool 的卷积对，并在它们之间共享同一个输入量化器（TensorRT Q/DQ 折叠的要求）；同时把 `Bottleneck.addop` 的量化器绑定到 `cv1` 的。
- 校准喂入 `num_batch=25` 个批次（图像 `/255`，除 loader 自身外无增强），然后 `load_calib_amax(method="mse")`。
- QAT（`quantize.finetune`）用 MSE 监督各层的**输出**、对齐冻结的 FP 模型副本（`supervision_stride` 控制哪些 `model.model[*]` 层参与损失）—— 而不是常规的检测损失。
- 评估时会禁用 Detect 头（`model.model[24]`）的量化，因为 ONNX 导出会剪掉检测头的解码部分；PTQ/QAT checkpoint 通过 `torch.save({"model": model}, ...)` 保存整个模型，加载时用 `weights_only=False`。
- `scripts/qat.py` 中的 `export_onnx` 会临时对 Detect 头打猴子补丁（`_make_grid` 改为 CPU 常量网格；`--noanchor` 替换 forward 以输出原始卷积结果），并开启 `TensorQuantizer.use_fb_fake_quant` 以生成 Q/DQ（opset 13）。

## 上游架构（导览）

- `models/yolov5*.yaml` 是层列表，由 `models/yolo.py`（`parse_model`）解析成 `DetectionModel` —— 一个类 `nn.Sequential` 的扁平 `model.model`，例如下标 24 是 `Detect` 头，输出 stride 8/16/32。基础构件（`Conv`、`C3`、`SPPF`、`Bottleneck`...）在 `models/common.py`；`Detect`/`Segment`/`BaseModel` 在 `models/yolo.py`。
- 仓库根目录的任务入口：`train.py`、`val.py`、`detect.py`、`export.py`、`hubconf.py`（torch.hub）、`benchmarks.py`；`classify/` 和 `segment/` 按任务镜像了这些入口。
- `utils/` 存放核心机制：`dataloaders.py`（create_dataloader + 增强）、`general.py`（NMS、各种检查、coco 评估胶水）、`loss.py`、`metrics.py`、`augmentations.py`、`loggers/`、`torch_utils.py`。
- `data/` = 数据集 yaml + `hyps/` 超参数预设（`hyp.scratch-{low,med,high}.yaml`）；`data/scripts/get_coco*.sh` 下载数据集/权重。
- `weights/yolov5s.pt` 已在本地；其他权重通过 `attempt_download` 自动下载。
