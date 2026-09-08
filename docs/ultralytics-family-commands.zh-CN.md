# Ultralytics 家族模型部署实操手册（yolov5su / yolov8，从 .pt 到 DLA 检测结果）

本文整理 P0–P2 打通的两条已验证链路的**全部命令**，按执行顺序排列。
原理与设计见 [ultralytics-support.zh-CN.md](ultralytics-support.zh-CN.md)；
v5 v7.0 家族（含自定义数据集）的路线见根目录 CLAUDE.md 的「线路 A / 线路 B」。

> 两个家族的差异只在**检测头**：v5su = v8 头 + C3 骨干（无注意力、无 Slice，最顺）；
> yolov8 = 原生 C2f 骨干（需要 chunk 拆分，P1.5 已自动化）。命令完全同构，仅权重名不同。
> yolo11 / yolo26 因注意力算子暂不支持 standalone DLA，不在本手册范围。

---

## 0. 前置条件

- **Jetson 端**（构建 loadable + C++ 部署 + 演示级校准）：环境见根 CLAUDE.md「运行环境」
  （含 PyTorch，本机可跑 QAT 演示；正式精度验收在 GPU 服务器做）
- **`ultralytics/` 仓库**（8.4.142，含本工具链的 `scripts/export_dla.py`、`scripts/qat_dla.py`）
- **权重**：`ultralytics/weights/yolov5su.pt`、`yolov8s.pt`（COCO 预训练）
- 校准/演示数据：任意图片文件夹（免标签）。示例用 COCO val2017：`/media/data/jia/coco/images/val2017`
- 部署端构建参数三件套必须配套（详见第 4 步）：`HEAD_STYLE=v8`、`INPUT_SCALE=<缓存 images: 值>`、
  分辨率与导出 `--size` 一致

---

## 1. FP16 快速路线（无需量化，最快看到检测结果）

### 1.1 导出 FP32 raw-head ONNX（GPU 机器或 Jetson 均可）

```bash
cd ultralytics

# yolov8s（原生骨干：自动做 chunk->双Conv 拆分，导出前有数值等价自检）
python scripts/export_dla.py --weights weights/yolov8s.pt --size 672 \
    --save ../data/model/yolov8s_raw_672.onnx --dynamic

# yolov5su 同一条命令，换个权重即可
python scripts/export_dla.py --weights weights/yolov5su.pt --size 672 \
    --save ../data/model/yolov5su_raw_672.onnx --dynamic
```

要点：`--size` 支持 `672`（正方形）或 `HxW` 如 `736x1280`（矩形，720p）；
`--dynamic` 必须加（trtexec 显式 shapes 需要）；输出为 s8/s16/s32 三路
`[1, 4*reg_max+nc, H, W]`（v5su/v8 = 144 通道）。

### 1.2 构建 FP16 loadable（Jetson）

```bash
cd ..   # 仓库根
/usr/src/tensorrt/bin/trtexec --onnx=data/model/yolov8s_raw_672.onnx --fp16 \
    --saveEngine=data/loadable/yolov8s.fp16.fp16chw16in.fp16chw16out.standalone.bin \
    --inputIOFormats=fp16:chw16 --outputIOFormats=fp16:chw16 \
    --buildDLAStandalone --useDLACore=0
```

### 1.3 构建匹配的 C++ 并运行

```bash
# matx 库 + 主程序都要 HEAD_STYLE=v8
HEAD_STYLE=v8 bash src/matx_reformat/build_matx_reformat.sh
make clean && make HEAD_STYLE=v8

./build/cudla_yolov5_app \
    --engine data/loadable/yolov8s.fp16.fp16chw16in.fp16chw16out.standalone.bin \
    --image data/images/image.jpg --backend cudla_fp16    # 检测框 → result.jpg
```

参考结果（COCO 街景图，672）：v8s 9 个检测、bus ~0.99、行人/车辆框位经视觉核验全部合理。

---

## 2. INT8 完整路线（QAT 量化 → 翻译 → loadable → 检测）

### 2.1 量化校准 + QAT 微调（GPU 机器或 Jetson）

```bash
cd ultralytics

# PTQ（校准）+ QAT（蒸馏微调）。Jetson 演示参数：500 图、25 batch、20 iter
python scripts/qat_dla.py quantize weights/yolov8s.pt \
    --calib-dir /media/data/jia/coco/images/val2017 --imgsz 672 \
    --limit 500 --calib-batches 25 \
    --ptq ptq_v8.pt --qat qat_v8.pt --epochs 1 --iters 20

# 正式训练（服务器）：--calib-dir 指向训练图文件夹、去掉 --limit/--iters 限制、
# --epochs 10（每轮按 mAP 筛选最优 checkpoint 属后续增强，当前存末轮）
```

产物：`ptq_v8.pt`（校准即存）/ `qat_v8.pt`（微调后）。v5su 同命令换权重。

### 2.2 导出 QAT ONNX（Q/DQ 节点 + chunk 拆分，自动等价自检）

```bash
python scripts/export_dla.py --weights qat_v8.pt --qat --size 672 --dynamic \
    --save ../data/model/yolov8s_qat.onnx
# 期望输出：138 对 Q/DQ、0 个 Slice、worst relative diff = 0.00
```

### 2.3 翻译成 PTQ ONNX + INT8 校准缓存（Jetson，无需 torch）

```bash
cd ../export/qdq_translator
python3 qdq_translator.py --input_onnx_models=../../data/model/yolov8s_qat.onnx \
    --output_dir=../../data/model/ --infer_concat_scales --infer_mul_scales
cd ../..
# 产物：data/model/yolov8s_qat_noqdq.onnx + ..._precision_config_calib.cache（+json/layer_arg）
```

**记录输入 scale**（下一步的 INPUT_SCALE）：

```bash
grep '^images:' data/model/yolov8s_qat_precision_config_calib.cache
# hex → float：python3 -c "import struct;print(struct.unpack('>f',bytes.fromhex('<hex>'))[0])"
# 本链路实测值：0.007874015718698502
```

### 2.4 构建 INT8 loadable（Jetson）

```bash
bash data/model/build_dla_standalone_loadable_v8_int8.sh
# 脚本内含：TRT-10 缓存头改写 + 6 个检测头末层 conv + 3 个头 Concat 的 FP16 约束
# 换分辨率/换输出名时复制该脚本改 SHAPES 与 --saveEngine
```

### 2.5 构建匹配的 C++ 并运行

```bash
HEAD_STYLE=v8 bash src/matx_reformat/build_matx_reformat.sh
make clean && make HEAD_STYLE=v8 INPUT_SCALE=0.007874015718698502f

./build/cudla_yolov5_app \
    --engine data/loadable/yolov8s.int8.int8hwc4in.fp16chw16out.standalone.bin \
    --image data/images/image.jpg --backend cudla_int8
```

参考结果（本机演示级校准）：9 个检测 @ 8.0ms，视觉核验与 FP16 一致、无量化退化。

---

## 3. 验证集精度验收（COCO mAP）

```bash
# 设备端 COCO 式评测（HEAD_STYLE=v8 INPUT_SCALE 同上构建）
./build/cudla_yolov5_app \
    --engine data/loadable/yolov8s.int8.int8hwc4in.fp16chw16out.standalone.bin \
    --coco_path data/coco/ --backend cudla_int8          # 生成 predict.json（~30-60 分钟）
python3 test_coco_map.py --predict predict.json --coco data/coco/
```

对照基线（GPU 服务器或本机，pycocotools 同口径）：

```bash
cd ultralytics && python - << 'EOF'
from ultralytics import YOLO
m = YOLO('weights/yolov8s.pt')
m.val(data='coco.yaml', imgsz=672, split='val')   # 官方 val mAP（FP32 参照）
EOF
```

> 归因方法同三分类模型：FP32 → QAT(伪量化, eval_pt_coco.py 思路) → INT8 设备端，
> 三段对比量化损失。本机演示级校准的数字仅证明链路，正式精度结论以服务器全量
> 训练 + 上述 COCO 评测为准。

---

## 4. 参数配套速查（换模型/分辨率必读）

| 环节 | 参数 | 必须一致的对方 |
|---|---|---|
| 导出 `--size` | `672` / `736x1280` | loadable 的 `SHAPES`、构建的 `INPUT_H/INPUT_W` |
| 导出/量化 `--imgsz` | 同上 | 校准分布与部署分辨率对齐（长边一致即可） |
| loadable 构建 | `--layerPrecisions` 头部清单 | translator 的缺 scale 报告（layer_arg.txt） |
| matx + make | `HEAD_STYLE=v8` | 模型头风格（v5su/v8 都用 v8） |
| make | `INPUT_SCALE=<缓存 images: 值>` | **每个新缓存都要查**，不匹配=结果乱 |
| make | `NUM_CLASSES=80` | 模型类别数 |
| 运行 | `--engine` + `--backend cudla_int8/cudla_fp16` | loadable 类型 |

切参数后一律 `make clean`（+ matx 重编），与 v5 家族的约定相同。
