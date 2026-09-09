# INT8/FP16 层占比分析与提升实录（v5s / v5su / v8s，2026-09-09）

本文完整记录一次"分析 DLA loadable 中 INT8 与 FP16 层的占比 → 发现隐藏 bug → 修复 →
推理速度翻倍"的全过程。所有命令可复现，所有数字为实测。

> 背景：v5s(v7.0) INT8 推理 ~3.9ms，v8s INT8 却要 ~7.9ms，同为 s 级模型差距 2 倍。
> 起点是用户的提问："v5su/v8s 的 INT8 和 FP16 占比如何？能否在不损失精度的前提下
> 提高 INT8 占比？"

---

## 1. 原理：FP16 回退层从哪来

INT8 构建采用**隐式量化**（PTQ ONNX + 校准缓存给 trtexec），一个 conv 能否跑 INT8
取决于：**它的输出张量在缓存里有没有 scale**。没有 scale 的层必须回退 FP16。

而"没有 scale"的根源几乎总是**检测头拓扑**：

- 图的最终输出（s8/s16/s32）永远没有 scale —— Q/DQ 只插在算子之间，图出口后面
  没有算子可插
- 直接产生图输出的算子 → 输出无 scale → FP16（这是设计使然）
- **QAT 导出时未被量化器覆盖的中间路径** → 输出无 scale → FP16（这可能是 bug）

## 2. 占比分析（用到的命令）

分析脚本核心逻辑：数 ONNX 里的 Conv 节点 → 查每个 conv 输出是否在 cache 的
key 集合里 → 按 --layerPrecisions 名单 + scale 覆盖情况分类。

```bash
cd ~/jia/cuDLA-samples
python3 << 'EOF'
import onnx
import numpy as np

def analyze(tag, onnx_path, cache_path, fp16_names):
    g = onnx.load(onnx_path).graph
    inits = {i.name: i for i in g.initializer}
    convs = [n for n in g.node if n.op_type == 'Conv']
    covered = set()
    for line in open(cache_path):
        if ':' in line and not line.startswith('TRT-'):
            covered.add(line.split(':',1)[0].strip())
    def macs(n):  # 用权重形状估算该 conv 的计算量权重
        w = inits.get(n.input[1])
        if w is None: return 1
        oc, ic, kh, kw = onnx.numpy_helper.to_array(w).shape
        return oc*ic*kh*kw
    int8, fp16 = [], []
    for n in convs:
        (fp16 if (n.name in fp16_names or n.output[0] not in covered) else int8).append(n)
    total, fp16_m = sum(macs(n) for n in convs), sum(macs(n) for n in fp16)
    print(f"\n===== {tag} =====")
    print(f"Conv: {len(convs)} | INT8: {len(int8)} ({100*len(int8)/len(convs):.0f}%) "
          f"| FP16: {len(fp16)} ({100*len(fp16)/len(convs):.0f}%) | cache: {len(covered)} 条")
    print(f"FP16 层计算量占比: {100*fp16_m/total:.2f}%")
    for n in fp16:
        w = inits.get(n.input[1])
        s = onnx.numpy_helper.to_array(w).shape if w is not None else '?'
        print(f"  FP16: {n.name}  w={s}")

# v5s 家族（用 9.9 重训产物，v7.0 架构）
analyze('v5s (v7.0 arch) INT8',
        'data/model/yolov5_coco_qat_9.9_noqdq.onnx',
        'data/model/yolov5_coco_qat_9.9_precision_config_calib.cache',
        {'/model.24/m.0/Conv','/model.24/m.1/Conv','/model.24/m.2/Conv'})

# v8s
analyze('v8s INT8',
        'data/model/yolov8s_qat_noqdq.onnx',
        'data/model/yolov8s_qat_precision_config_calib.cache',
        {'/model.22/cv2.0/cv2.0.2/Conv','/model.22/cv2.1/cv2.1.2/Conv','/model.22/cv2.2/cv2.2.2/Conv',
         '/model.22/cv3.0/cv3.0.2/Conv','/model.22/cv3.1/cv3.1.2/Conv','/model.22/cv3.2/cv3.2.2/Conv'})
EOF
```

### 分析结果（修复前）

| 模型 | Conv 总数 | INT8 | FP16 | FP16 计算量占比 | 推理 |
|---|---|---|---|---|---|
| v5s (v7.0) | 60 | 57 (95%) | **3**（= 3 个头 conv，设计使然） | 3.2% | 3.9ms |
| v8s 原生 | 71 | 59 (83%) | **12**（6 个头 conv + **6 个意外**） | 9.1% | 7.9ms |
| v5su | 75 | 62 (83%) | **13**（6 + **7 个意外**） | 12.7% | 8.9ms |

**异常信号**：v8/v5su 各有 6~7 个构建脚本里**根本没指定**的 FP16 层 —— 全是
`/model.{2,4,6,8}/m.*/cv2/conv/Conv`（C2f/C3 瓶颈块的第二 conv，3×3）。

## 3. 根因定位（逐步排查的命令）

### 3.1 确认意外层缺的是 scale（对比两代缓存的 key 覆盖）

```python
def keys(path):
    s = set()
    for line in open(path):
        if ':' in line and not line.startswith('TRT-'):
            s.add(line.split(':',1)[0].strip())
    return s

c5 = keys('data/model/yolov5_coco_qat_9.9_precision_config_calib.cache')  # v7.0 正常
c8 = keys('data/model/yolov8s_qat_precision_config_calib.cache')          # v8 缺

# v7.0 的瓶颈 cv2 路径三件套全在：
#   cv2/conv/Conv_output_0 ✓  cv2/act/Sigmoid_output_0 ✓  cv2/act/Mul_output_0 ✓
# v8 只有 Sigmoid（且是 1/127 默认值），Conv 和 Mul 都缺！
```

### 3.2 检查 QAT ONNX 里 Q 节点的位置（决定性证据）

```bash
python3 << 'EOF'
import onnx
m = onnx.load('data/model/yolov8s_qat.onnx')   # QAT 图（带 Q/DQ 的那份）
g = m.graph
for n in g.node:
    if n.op_type == 'Mul' and '/model.4/m.0/cv2/act' in (n.name or ''):
        consumers = [c.op_type for c in g.node if any(i == n.output[0] for i in c.input)]
        print('cv2/Mul ->', consumers)
        # 修复前输出: ['Add']           ← 没有 QuantizeLinear！Q 丢了
        # 修复后输出: ['QuantizeLinear'] ← 正常
EOF
```

### 3.3 根因

**类级 monkeypatch 不随 pickle 序列化**。`qat_dla.py` 训练时把
`Bottleneck.forward = bottleneck_forward_quant`（残差走 QuantAdd）打在**类**上；
checkpoint 只保存实例状态。`export_dla.py --qat` 用 `torch.load` 加载后，
Bottleneck 走的是 ultralytics **原版** forward（普通 `x + cv2(cv1(x))`）——
QuantAdd（量化残差）从未被调用 → cv2 链上没有任何 Q 节点 → 翻译后无 scale
→ 意外 FP16。且**训练好的 QuantAdd scale 被静默丢弃**。

v5 路线（yolov5_dla）没踩坑的原因：`cmd_export` 每次导出都重新执行
`replace_bottleneck_forward`；而 ultralytics 的 export_dla.py 只在训练侧打过一次补丁。

## 4. 修复（一行本质，防止遗漏的完整上下文）

`ultralytics/scripts/export_dla.py` 的 `--qat` 加载分支，在 `torch.load` 之前加：

```python
# 类级 forward 补丁不进 pickle —— 加载后必须重应用，否则 Bottleneck 导出的是
# 无量化的原版残差，cv2 链丢掉全部 Q 节点 → 瓶颈 conv 意外回退 FP16
from ultralytics.nn.modules.block import Bottleneck
Bottleneck.forward = qat_dla.bottleneck_forward_quant
```

（提交 `184f7a4`。注意旁边原有的 `__main__` 注入解决的是**反序列化时找不到
QuantAdd 类**的问题，与这个补丁是两件事。）

## 5. 修复后的完整重建链（v8s，按序可抄）

```bash
cd ~/jia/cuDLA-samples/ultralytics

# ① 重新导出（现在 QuantAdd 生效，cv2 链带 Q）
python scripts/export_dla.py --weights qat_v8.pt --qat --size 672 --dynamic \
    --save ../data/model/yolov8s_qat.onnx
#    期望输出：worst relative diff = 0.00e+00（拆分自检），cv2/Mul -> QuantizeLinear

cd ..
# ② 重新翻译
cd export/qdq_translator
python3 qdq_translator.py --input_onnx_models=../../data/model/yolov8s_qat.onnx \
    --output_dir=../../data/model/ --infer_concat_scales --infer_mul_scales
cd ../..

# ③ 核对输入 scale（这次没变：3c010204 → 0.007874015718698502）
grep '^images:' data/model/yolov8s_qat_precision_config_calib.cache

# ④ 重建 INT8 loadable（脚本内含 TRT-10 缓存头改写）
bash data/model/build_dla_standalone_loadable_v8_int8.sh

# ⑤ 按配套参数重建 C++（三件套：HEAD_STYLE / INPUT_SCALE；分辨率同导出 --size）
HEAD_STYLE=v8 bash src/matx_reformat/build_matx_reformat.sh
make clean && make HEAD_STYLE=v8 INPUT_SCALE=0.007874015718698502f

# ⑥ 验证
./build/cudla_yolov5_app \
    --engine data/loadable/yolov8s.int8.int8hwc4in.fp16chw16out.standalone.bin \
    --image data/images/image.jpg --backend cudla_int8          # 单图 → result.jpg
./build/cudla_yolov5_app \
    --engine data/loadable/yolov8s.int8.int8hwc4in.fp16chw16out.standalone.bin \
    --coco_path data/coco --backend cudla_int8 --nms cpu       # COCO 5000 张
python3 test_coco_map.py --predict predict.json --coco data/coco/
```

**排查中踩过的一个小坑**：第一次冒烟出现 946 个乱检 —— 是当时二进制还是 v5 默认
构建（测 NMS 开关后重建的），跑 v8 loadable 属于参数错配。**换模型先确认构建参数
三件套**（HEAD_STYLE/INPUT_SCALE/INPUT_H×W 与 loadable 匹配），是本项目反复强调的纪律。

## 6. 修复后实测

| 指标 | 修复前 | 修复后 |
|---|---|---|
| v8s INT8 conv 占比 | 83%（59/71） | **92%**（65/71） |
| v8s FP16 层 | 12 | **6**（仅头部的 cv2.X.2/cv3.X.2 六个 1×1 conv，设计使然） |
| v8s 推理 | 7.9ms | **3.94ms（2.0×）** |
| v8s COCO mAP50-95 | 44.6（GPU NMS 口径 44.6 / 前次） | **44.6（cpu NMS 实测 0.446）** |
| mAP50 | — | 0.618 |
| AR@100 | — | 0.637 |
| 缓存 `images:` | 3c010204 | 3c010204（不变，C++ 无需改） |

意外层只占 ~9% 计算量，但在 DLA 上 FP16 = 2× 成本 + 逐层开销，吃掉了**一半**推理时间。

## 7. 剩余 6 个 FP16 层的定性（为什么不再往下压）

`cv2.X.2`（box 分支末 1×1，→64ch）与 `cv3.X.2`（cls 分支末 1×1，→80ch）直供图输出
Concat。Option#1 量化只给 conv **输入**配量化器，这些层的**输出没有训练出的 scale**；
人工补 scale 等于在检测头最敏感处做 PTQ —— 正是 v2 脚本花速度换精度（37.1→37.3）
的反方向操作。3 个头部 Concat 则受 `fp16:chw16` 输出格式约束必须 FP16。
**结论：6+3 是此头型的合理下限，不再压缩。**

## 8. 最终模型矩阵

| 模型 | mAP50-95 | 推理 | INT8 占比 | 备注 |
|---|---|---|---|---|
| v5s (v7.0) INT8 | 37.1 | 3.9ms | 95% | 官方基线 |
| v8s 原生 INT8 | **44.6** | **3.94ms** | 92% | 速度精度双优，新主力 |
| v5su INT8 | 42.4（待重导出复核） | 预计 ~4.5ms | 预计 ~92% | 重新导出后同步受益 |

## 9. 经验清单

1. **类级 monkeypatch 不进 pickle** —— 训练时打在类上的补丁，换进程加载即失效；
   症状极其隐蔽（不报错，只静默多几个 FP16 层）。凡是"训练侧打补丁 + checkpoint
   中转 + 另一进程导出"的流程，导出侧必须重应用全部类级补丁
2. **占比分析要同时看层数和计算量** —— 意外层只占 9% 计算量却吃 50% 时间，DLA 的
   逐层开销不可小觑
3. **构建脚本 `--layerPrecisions` 名单不是 FP16 全集** —— 真全集 = 名单 ∪ 缺 scale
   的层；分析时必须从 cache 反推
4. 换模型/重建后先跑单图冒烟核对检测数量级，再上 COCO 全量
