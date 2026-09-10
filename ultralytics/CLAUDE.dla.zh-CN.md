# CLAUDE.dla（中文版）—— Ultralytics 家族的 DLA 量化训练文档

本文件为 Claude Code (claude.ai/code) 在本仓库工作时提供指导：`scripts/qat_dla.py`（PTQ/QAT 训练）与 `scripts/export_dla.py`（DLA 部署导出）的完整使用说明。上游 `CLAUDE.md` 管的是 ultralytics 本身；**本文件管的是我们加在它上面的 DLA 量化层**。部署命令速查见仓库根 [docs/ultralytics-family-commands.zh-CN.md](../docs/ultralytics-family-commands.zh-CN.md)，原理与设计见 [docs/ultralytics-support.zh-CN.md](../docs/ultralytics-support.zh-CN.md)。

## 本工具层是什么

Ultralytics **8.4.142**（上游 clone，`.git` 已移除并入主仓库）加上两个**新增脚本**（不改动任何上游文件）：

- `scripts/qat_dla.py`（新增）—— PTQ 校准 + QAT 蒸馏微调（yolov5_dla `scripts/qat.py` 的 ultralytics 移植版）
- `scripts/export_dla.py`（新增）—— DLA 部署导出（raw-head + chunk 拆分 + Q/DQ 发射）
- 预训练权重在 `weights/`（yolov8s / yolo11s / yolov5su / yolo26s）

`pytorch_quantization` 来自 NVIDIA 源（注意占位包陷阱，见仓库根 CLAUDE.md 的 Export Toolchain 表）：
```bash
pip install --no-deps --index-url https://p6.nvidia.com pytorch-quantization && pip install absl-py prettytable
```

## Ultralytics 标准命令（本机可用）

```bash
yolo predict model=weights/yolov8s.pt source=../data/images/image.jpg   # 推理
yolo train model=weights/yolov8s.pt data=coco.yaml imgsz=672 epochs=100 # 训练（服务器）
python scripts/qat_dla.py quantize ...                                   # ↓ 见下文
```

## 量化训练全流程（qat_dla.py）

所有命令在 `ultralytics/` 仓库根运行。权重 → 校准/QAT → 导出 → 翻译 → loadable 五步，前两步在这里，后三步见命令手册。

### 数据的两种模式

| 模式 | 触发 | 说明 |
|---|---|---|
| 文件夹（默认） | 只给 `--calib-dir` | 直接扫目录下的图片。**免标签**（校准/MSE 蒸馏都不需要标签） |
| 清单（yolov5_dla 同款） | 加 `--train-list`/`--val-list` | 清单文件位于 `--calib-dir` 内；**条目支持绝对路径（原样）或相对路径（相对 calib-dir 解析）**。读不到的条目跳过并提示 |

### 命令模板（以 3 类数据集 + yolov8s 为例）

```bash
# ── 基础版（只要 PTQ 校准）──
python scripts/qat_dla.py quantize weights/yolov8s.pt \
    --calib-dir /root/dataset/3classes --imgsz 672 \
    --train-list origin_train.txt ship_add_train.txt \
    --ptq ptq_v8.pt

# ── 完整版（Origin/PTQ 基线 + QAT 微调 + best-mAP 保存 + 历史落盘）──
python scripts/qat_dla.py quantize weights/yolov8s.pt \
    --calib-dir /root/dataset/3classes --imgsz 672 \
    --train-list origin_train.txt ship_add_train.txt \
    --val-list origin_val.txt \
    --data data/3classes.yaml \
    --ptq ptq_v8.pt --qat qat_v8.pt \
    --eval-origin --eval-ptq --summary summary_v8.json

# Jetson 上的干跑参数（验证链路用，几分钟出结果）：
#   --limit 500 --calib-batches 25 --epochs 1 --iters 20
# 服务器正式训练：去掉 --limit，--epochs 10（lr 调度 {0:1e-6, 3:1e-5, 8:1e-6}）
```

### 全部参数（与 yolov5_dla qat.py 的对照）

| 参数 | 说明 | yolov5_dla 对应 |
|---|---|---|
| `weight`（位置参数） | .pt 权重，任意 v8 头家族（v5su/v8/v11/v26 通用，模型无关） | 相同 |
| `--calib-dir` | 数据集根目录（文件夹模式直接扫图 / 清单模式作根） | `--cocodir` |
| `--train-list`（可多个） | 校准+微调清单；条目绝对/相对均可 | 相同 |
| `--val-list`（可多个） | 验证清单 → 存 PTQ/QAT 后各报一次**免标签分歧分数**（量化模型 vs 冻结 FP 教师的逐层 MSE，非 mAP） | 相同（那边喂评测） |
| `--imgsz` | 正方形 int 或 `HxW`（如 736x1280），对齐部署分辨率 | 相同 |
| `--limit` / `--calib-batches` | 最多图片数 / 校准 batch 数（默认 500/25） | 用 `--iters` 间接控制 |
| `--ptq` / `--qat` | 输出 checkpoint 路径 | 相同 |
| `--epochs` / `--iters` | 轮数 / 每轮 batch 上限 | 相同 |
| `--data` | **ultralytics 数据集 yaml**（mAP 评测用）。给了它 qat.pt 存 best-mAP 轮，不给存末轮 | 相同（那边默认有） |
| `--eval-origin` / `--eval-ptq` | 量化前 FP / 校准后 PTQ 的 mAP 基线（需 --data） | 相同 |
| `--ignore-policy` | 正则：匹配的 conv 不量化（如 `model\.22\..*` 保住 v8 头） | 相同 |
| `--supervision-stride` | 蒸馏监督层的采样步长（默认 3） | `--supervision-stride` |
| `--summary` | 训练历史 JSON（`[["Origin",...],["PTQ",...],["QAT0",...]]`） | `summary.json`（自动） |
| `--batch-size` / `--device` | 批大小 / 设备 | 相同 |

### 输出示例（实测格式）

```
Origin mAP50-95 = 0.08239          ← FP 基线（--eval-origin）
quantized 64 convs, 6 residual adds (QuantAdd)
calibration done (input: histogram+mse, weight: max)
PTQ mAP50-95 = 0.08345           ← 校准后（--eval-ptq）
QAT 1/2 @lr=1e-06: ...            ← lr 调度生效
  epoch 1: mAP50-95 = 0.08374
  saved best-mAP checkpoint to qat_v8.pt (map=0.08374)
QAT history: [(0, 0.08374), (1, 0.08267)] | best mAP50-95 = 0.08374
summary saved to summary_v8.json
```

三段归因表（Origin → PTQ → QAT）一条命令自动产出。

## 导出（export_dla.py）

两种风味按输入 checkpoint 自动区分（详见命令手册的对照表）：

```bash
# FP16 路径（纯 FP32 权重）—— chunk 拆分自动做，导出前数值等价自检
python scripts/export_dla.py --weights weights/yolov8s.pt --size 672 \
    --save ../data/model/yolov8s_raw.onnx --dynamic

# INT8 路径（QAT/PTQ checkpoint）—— 加 --qat 发射 Q/DQ 节点
python scripts/export_dla.py --weights qat_v8.pt --qat --size 672 --dynamic \
    --save ../data/model/yolov8s_qat.onnx
```

`--size` 支持 `672`（正方形）或 `736x1280`（HxW 矩形）；`--dynamic` **必须加**（trtexec 显式 shapes 需要，静态图会拒绝）。

## 各部分如何协作（机制速览）

> QAT 原理的四阶段逐行剖析（模块替换 → 校准 → STE 蒸馏 → Q/DQ 导出）见 yolov5_dla 侧的
> [docs/qat-internals.zh-CN.md](../yolov5_dla/docs/qat-internals.zh-CN.md) —— 原理完全同源。

- `replace_to_quantization_module()`：递归把 `nn.Conv2d` 换成 `QuantConv2d`（C2f/C3 的 Conv 包装类内部都是标准 Conv2d，**一次实现对全家族生效**）。`--ignore-policy` 在此处按模块路径正则跳过
- `replace_bottleneck_forward()`：**类级**补丁让残差走 `QuantAdd`（单量化器量化两路 —— INT8 加法要求两输入同 scale）。C2f 和 C3 共用同一个 Bottleneck 类
- 校准：直方图收集 + `load_calib_amax(method="mse")`（输入）/ max（权重默认）
- 微调：`finetune()` 冻结 FP 教师副本 + 逐层 MSE 蒸馏 + 预热→训练→退火 lr 调度；`on_epoch_end` 回调驱动 best-mAP 保存
- `evaluate_map()`：借 ultralytics 自带验证器在原地量化模型上评 mAP；**Detect 头量化器评测时临时关闭**（匹配部署 loadable 头 conv 跑 FP16 的现实）；yaml 清单里的相对条目自动绝对化（ultralytics 按 CWD 解析是坑）
- 导出：`raw_forward` 补丁输出原始头 → `make_dla_friendly` 拆 chunk（权重/bias/BN/逐通道 amax 全按半切，相对阈值 1e-2 自检）→ `use_fb_fake_quant` 发射 Q/DQ

### 踩过的坑（都在这两个脚本里修掉，改导出逻辑前先读）

1. **类级 monkeypatch 不进 pickle**：checkpoint 里类级 forward 补丁丢失 → QuantAdd 静默失效 → 瓶颈 conv 意外 FP16（v8s 曾因此慢一倍）。`--qat` 加载后必须重应用 `Bottleneck.forward` 补丁
2. **量化器初始化的调用者检查**：`init_quantizer` 拒绝 `__init__` 之外的调用者（按帧名检查）→ 用"嵌套 `__init__` 函数"技巧绕过
3. **权重量化器逐输出通道**：拆 cv1 时 `_amax`（`[C,1,1,1]`）必须随权重同切
4. **等价性自检必须用相对阈值**：一个 2c conv vs 两个 c conv 的浮点累加顺序不同，绝对阈值会误报
5. **pytorch_quantization 安装**：`--index-url https://pypi.nvidia.com` + `--no-deps`（PyPI 占位包 + 假依赖 `sphinx-glpi-theme`）
6. **numpy 必须 <2**（本机 nv-torch 按 1.x 编译）；opencv 5.x 会强制 numpy≥2，锁 `opencv-python==4.10.0.84`

## 与 yolov5_dla qat.py 的刻意差异

| 未移植 | 原因 |
|---|---|
| Option#2（`--all-node-with-qdq`） | translator 的 `--infer_concat_scales`/`--infer_mul_scales` 已验证够用 |
| rules.py 量化器共享 | 同上；implicit 量化路径不需要 TRT 折叠约束 |
| `sensitive` / `test` 子命令 | **TODO**（sensitive 需 N×mAP 评测，服务器场景按需加） |
| `export` 子命令 | 分离到 export_dla.py（职责分离，且导出要做 chunk 拆分，逻辑不同） |

## 已验证的成绩（供对照）

| 模型 | mAP50-95 | 推理 | 说明 |
|---|---|---|---|
| yolov8s INT8（672） | 44.6 | 3.94ms | 占比修复后 |
| yolov5su INT8（672） | 42.6 | 3.47ms | C3 骨干无拆分开销 |
| yolo11s / yolo26s | — | — | 注意力算子阻断 loadable 构建（非训练问题） |

## 上游架构（导览）

- `ultralytics/` 包即 pip 同名包：`engine/`（trainer/validator/predictor 基类）、`nn/`（`tasks.py` 从 yaml 建模，`modules/` 层动物园，`autobackend.py` 统一推理后端）、`models/`（各家族按任务子类化 engine）、`cfg/`（`default.yaml` 定义全部 train/val/predict/export 参数）、`data/`、`utils/`
- Detect 头（`nn/modules/head.py`）：`no = nc + reg_max*4`；`cv2` 分支出 64ch 框分布 + `cv3` 分支出 nc 类别；DFL + dist2bbox 解码（8.4.142 实测 yolo26 为 reg_max=1，无 DFL）
- 部署端解码（`~/jia/cuDLA-samples/src/decode_dfl.cu`）实现了同样的数学：softmax 积分 + 网格中心 ± 距离 × stride + 纯 sigmoid 类别分
