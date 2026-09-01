# QAT 量化原理与代码剖析（yolov5_dla 实现笔记）

本文剖析 `quantization/quantize.py`（416 行）、`quantization/rules.py`（94 行）与 `scripts/qat.py` 的协作，
回答一个核心问题：**一个训练好的 FP32 模型，是如何变成"量化后几乎不掉点"的模型的？原理是什么？**

> 阅读前提：知道 PTQ/QAT 的区别（PTQ = 只统计 scale 不动权重；QAT = 带量化误差重训权重）。
> 本文所有行号均可在本仓库直接跳转核对。

---

## 0. 全景：`qat.py quantize` 一条命令的四个阶段

```
best.pt (FP32)
   │
   │ ① 模块替换      initialize() + replace_bottleneck_forward()
   │                 + replace_to_quantization_module()        [quantize.py:129/196/206]
   │    —— 把 nn.Conv2d 等换成 QuantConv2d 等"可量化模块"（权重原封保留）
   ▼
   │ ② 校准          calibrate_model()                         [quantize.py:265]
   │    —— 25 batch 数据喂入，每个量化器收集直方图 → 按 MSE 选 amax
   ▼  （此刻 = PTQ 模型，--ptq 保存的就是它）
   │ ③ QAT 微调      finetune()                                [quantize.py:312]
   │    —— 冻结的 FP 副本当教师，逐层输出 MSE 对齐，Adam lr=1e-5 训练
   ▼  （每 epoch 评 mAP，最优存 qat.pt）
   │ ④ 导出          export_onnx()                             [quantize.py:409]
   │    —— use_fb_fake_quant=True，Q/DQ 变成真正的 ONNX 节点
   ▼
QAT ONNX（显式量化图）→ qdq_translator → INT8 校准缓存 → trtexec → DLA loadable
```

---

## 1. 原理基础

### 1.1 INT8 对称量化与 scale

INT8 推理时，张量值要映射到 `[-127, 127]`：

```
量化：   q  = round(x / scale)
反量化： x' = q × scale
scale  = amax / 127        （amax = 该张量数值范围的截断上界）
```

- **scale 是每个张量一份的**（每个卷积的输入激活一份、每个权重一份），它的选取直接决定量化误差
- amax 取真实最大值不一定最优 —— 长尾离群值会把 scale 撑大、反而牺牲大量小数值精度，所以要从**分布**上选截断点（见 §3 校准）
- DLA 的硬性要求：图中每个算子（包括 Concat/Add/SiLU）都要有 scale，否则该层只能跑 FP16（GPU 上 TensorRT 则允许例外）—— 这是 `models/common.py` 改动和 QuantConcat/QuantAdd 存在的根本原因

### 1.2 伪量化（fake quant）：训练时如何"模拟"量化

`TensorQuantizer`（pytorch-quantization 提供）就是插在张量流上的 **Q+DQ 对**：

```
前向：  y = conv( quantize_dequantize(x), quantize_dequantize(W) )
```

注意：**网络其余计算仍是 FP32** —— 伪量化只是让每个被量化的边界"带上量化误差"，
于是训练时模型感知到的就是 INT8 推理将遭遇的真实误差分布。

### 1.3 为什么"训练"能救精度：STE（直通估计器）

量化里的 `round()` 不可导，梯度无法穿过。**STE（Straight-Through Estimator）** 的处理：
反向传播时把 quantize-dequantize 近似为恒等映射，梯度直接放行：

```
∂loss/∂x ≈ ∂loss/∂x'   （穿过量化器直通）
```

于是两样东西可以被优化：

1. **权重**：在"带量化误差"的前向里正常接收梯度 → 逐渐适应误差、避开易出错的数值区域
2. **amax（scale）**：校准器给出初值，训练中作为约束内的可调量微调

**QAT 与 PTQ 的本质区别**就在这里：PTQ 只做 §3（scale 统计），权重冻结；
QAT 让权重带着误差重新适应 —— 我们的实测代价因此只有 0.4 mAP 点（FP32 0.482 → QAT 0.478）。

---

## 2. 阶段①：把 FP32 模型变成"可量化"结构

### 2.1 替换机制（`replace_to_quantization_module`，quantize.py:206）

pytorch-quantization 维护一张**类→类**的替换表 `_DEFAULT_QUANT_MAP`
（`nn.Conv2d→QuantConv2d`、`nn.Linear→QuantLinear`、`nn.MaxPool2d→QuantMaxPool2d`）。
函数递归遍历模型，命中即替换。

`transfer_torch_to_quantization`（:143）的替换手法值得一看：
用 `quantmodule.__new__` 绕过构造器创建空对象 → 把原模块的 `__dict__` 整体拷过去
（**权重、BN、超参数原封不动**）→ 再调 `init_quantizer` 挂上输入/权重量化器。
所以"替换"不改变任何数值，只加装量化边界。

`--ignore-policy "model\.24\.m\.(.*)"` 就是在这里起作用（:221 `quantization_ignore_match`），
路径匹配的模块跳过替换（例如想让检测头 conv 保持 FP）。

### 2.2 Option#1 vs Option#2（`initialize`，quantize.py:129）

- **Option#1（默认）**：只用 pytorch-quantization 自带的替换表 —— 只有 Conv/Linear/MaxPool 边界有量化器，符合 TensorRT 推荐的 Q/DQ 布局（GPU 融合友好）
- **Option#2（`--all-node-with-qdq`）**：往表里追加两条（:136-140）：
  - `nn.SiLU → QuantSiLU`（:57：两个独立量化器，分别量化 `x` 和 `sigmoid(x)` 后再相乘）
  - `models.common.Concat → QuantConcat`（:46：单一量化器统一量化所有输入再拼接）

  于是**每个算子边界**都有训练出的 scale —— DLA 的 INT8 覆盖可以更完整。
  这也是 `models/common.py` 必须把函数式 `torch.cat` 模块化的原因：替换表按**类**替换，函数永远换不了。

### 2.3 残差 Add 的处理（`replace_bottleneck_forward`，quantize.py:190-203）

`x + cv2(cv1(x))` 里的 `+` 是函数式算子，无法被替换表命中。做法：给每个带残差的
`Bottleneck` 挂一个 `addop = QuantAdd`（:69），再替换整个类的 `forward`：

```python
def bottleneck_quant_forward(self, x):                      # :190
    return self.addop(x, self.cv2(self.cv1(x))) if self.add else ...
```

`QuantAdd` 用**同一个** `_input0_quantizer` 量化两路输入（:81）—— 相加的两个张量必须同 scale，
否则 INT8 加法无法对齐。

---

## 3. 阶段②：校准（`calibrate_model`，quantize.py:265-309）

两步走：

1. **collect_stats**（:278）：把所有量化器切到 `enable_calib + disable_quant` 模式，前向 25 个 batch
   （图片 `/255`、无增强）。直方图校准器在每个量化边界收集数值分布（`_torch_hist=True` 走 torch 加速）
2. **compute_amax(method="mse")**（:267）：对每个直方图，遍历候选截断点 amax，
   选使**量化 MSE 最小**的那个 → `module._amax` 固化为 scale 依据

要点：
- 为什么只要 25 batch —— 统计分布足够稳，不需要全量数据
- `method` 可换 `"entropy"`（TensorRT 默认 KL 散度法）；这里选 mse 是作者的经验选择
- **校准完成的瞬间就是 PTQ 模型**（`--ptq` 保存的它）—— scale 已定、权重未动。
  我们的基线数据里 PTQ 通常已比 FP32 掉 1~2 点，QAT 微调把其中大部分追回来

---

## 4. 阶段③：QAT 微调（`finetune`，quantize.py:312-406）—— 核心

### 4.1 总体设计：蒸馏式逐层对齐，而不是检测损失

```python
origin_model = deepcopy(model).eval()          # :317 冻结教师
disable_quantization(origin_model).apply()     # :318 教师永远输出"无量化误差"的结果
```

- **教师** = 同一份权重的 FP32 副本（量化关闭，:85 的 `disable_quantization` 把所有
  `TensorQuantizer._disabled=True`，前向直接恒等放行）
- **学生** = 量化模型（伪量化生效）
- **损失**（:383-385）：

```python
quant_loss += MSELoss(学生某层输出, 教师同名层输出)   # 对所有被监督层求和
```

为什么用逐层 MSE 而不是原本的检测损失（置信度+框回归）？
- 目标本来就是"INT8 行为逼近 FP 行为"，逐层对齐是该目标**可微且密集**的代理信号；
  检测损失对 scale 的小扰动不敏感、梯度稀疏
- 检测数据仍在用（同样的 train loader 喂入），只是监督信号换了 —— 模型仍面对真实数据分布

### 4.2 监督哪些层：`supervision_policy`（qat.py:221-236）

`cmd_quantize` 传入的策略：`keep_idx = range(0, len-1, supervision_stride) + 最后一层`。
即每隔 `stride` 层取一层监督（默认 stride=3，`--supervision-stride` 可调）。
理由：60 层全监督时浅层误差会支配总损失（数值大、层数多），隔层采样让深层（语义层）也拿到有效梯度。
"没被监督 ≠ 不学习"（qat.py 原注释）—— 梯度照样穿过它们。

### 4.3 训练细节

| 项 | 值 | 出处 |
|---|---|---|
| 优化器 | Adam，lr 默认 1e-5 | :324 |
| lr 调度 | `{0:1e-6, 3:1e-5, 8:1e-6}` 预热→训练→退火 | :328-333 |
| epoch | 10（`nepochs=10`） | :314 |
| 每轮 batch 上限 | `early_exit_batchs_per_epoch`（= `--iters`，试跑用） | :314 |
| 混合精度 | amp autocast + GradScaler | :323,377,390 |

### 4.4 每轮验证与保存（qat.py 的 `per_epoch` 回调）

```python
with quantize.disable_quantization(model.model[24]):   # 检测头量化关闭
    ap = evaluate_coco(...)                            # 评 mAP
if ap > best_ap: torch.save({"model": model}, save_qat)  # 只存最优
```

检测头（`model.model[24]`）量化关闭的原因：ONNX 导出 `--noanchor` 会剪掉头部的解码部分，
评测时的图与导出图保持一致才公平。

> 实现细节坑：forward hooks（:335-338 收集各层输出用的）在 `torch.save`/导出前必须 remove（:400-402 注释），
> 否则钩子会被一起序列化，后续 unpickle 或导出行为异常。

---

## 5. 阶段④：导出（`export_onnx`，quantize.py:409-416）

```python
quant_nn.TensorQuantizer.use_fb_fake_quant = True     # 导出开关
torch.onnx.export(...)
quant_nn.TensorQuantizer.use_fb_fake_quant = False
```

- 平时 TensorQuantizer 是"假装"量化；`use_fb_fake_quant=True` 时它导出为**真正的
  QuantizeLinear/DequantizeLinear 节点**（fb = Facebook 风格 ONNX 量化表示，opset 13），
  scale 就是校准/微调得到的那个
- 配合 `qat.py export` 的两个 monkeypatch：`_make_grid` 换成 CPU 常量网格（避免导出动态网格算子）、
  `--noanchor` 替换 Detect.forward 输出原始卷积结果（s8/s16/s32，解码交给 C++）
- **`--noqadd`**：`cmd_export` 默认执行 `replace_bottleneck_forward`（会注入 QuantAdd）。
  对 qat.pt 这些量化器已校准、属于训练图；对 best.pt（纯 FP32）则注入**未校准**的量化器，
  导出图里出现 ~14 个垃圾 scale 的 Q/DQ —— 所以导出 FP32 trimmed ONNX 必须加 `--noqadd`
  （详见 CLAUDE.md「两种导出风味」）

---

## 6. rules.py：量化器共享规则（Option#1 专属）

### 6.1 为什么需要"共享"

TensorRT 编译显式量化图时有条约束：**汇入同一 Concat/MaxPool 的多条 conv 分支，
其输入 scale 必须一致**，否则融合（Q/DQ folding）无法进行（DLA 比 GPU 更严格）。
Option#1 各 conv 独立校准 → scale 天然各不相同 → 需要后处理统一。

### 6.2 怎么做（`apply_custom_rules_to_quantizer`，quantize.py:245）

1. 先把模型导出成临时 ONNX
2. `find_quantizer_pairs`（rules.py:59）做**图分析**：
   - 找每个 `Concat` / `MaxPool` 节点，枚举其上下游的 `QuantizeLinear`
   - 沿 `DQ→Conv` 回溯出 conv 名，生成 `(major, sub)` 对 —— major 是第一个发现的分支，sub 是其余
3. 回到 PyTorch 侧：`sub._input_quantizer = major._input_quantizer`（:253）
   —— **直接替换 Python 对象指针**，两个模块从此共用同一量化器（同一 scale）
4. 同样地把 `Bottleneck.addop` 的两个量化器绑到 `cv1.conv` 的输入量化器上（:256-262）

Option#2（`--all-node-with-qdq`）不走这套 —— 每个边界自己训练 scale，无需共享。

---

## 7. 与部署链路的衔接（scale 的一生）

```
直方图校准 → amax → scale ──→ TensorQuantizer._amax
                              │ finetune 中可微调
                              ▼
                    导出为 Q/DQ 节点的 scale（QAT ONNX）
                              │ qdq_translator：剥 Q/DQ
                              ▼
                    INT8 校准缓存（TRT-xxxx-EntropyCalibration2 文本格式）
                              │ trtexec --int8 --calib
                              ▼
                    DLA loadable（INT8 权重 + 各层 scale 固化）
```

- C++ 侧唯一手工同步的 scale：`mInputScale` = 缓存里 `images:` 条目（输入量化；
  `mOutputScale1-3` 是死代码）
- 实测两次重训练 `images:` scale 逐位相同（`3c00f9f4`）—— 输入分布稳定时校准结果是确定的
- 三分类归因基线（同验证集、pycocotools）：**FP32 0.482 → QAT 0.478 → DLA INT8 0.466**

---

## 8. 常用旋钮速查

| 旋钮 | 位置 | 作用 |
|---|---|---|
| `--iters` | `finetune(early_exit_batchs_per_epoch=)` | 每 epoch batch 上限，试跑提速 |
| `--supervision-stride` | `cmd_quantize` 的 keep_idx | 监督层采样密度（默认 3） |
| `--ignore-policy` | `replace_to_quantization_module` | 正则跳过指定模块不量化 |
| `--all-node-with-qdq` | `initialize()` | Option#2：SiLU/Concat 也量化（不走 rules） |
| `nepochs / lrschedule` | `finetune` 签名 | 训练轮数与 lr 调度（改源码） |
| `num_batch` | `calibrate_model` | 校准 batch 数（默认 25） |
| `method="mse"` | `compute_amax` | amax 选取准则（可改 "entropy"） |
| `qat.py sensitive` | `cmd_sensitive_analysis` | 逐层关闭量化评 mAP → 找最怕量化的层（配 `--ignore-policy` 用） |
