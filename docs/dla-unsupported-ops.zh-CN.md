# DLA 量化适配中的算子障碍总账（截至 2026-09-09）

本文汇总整个项目从开始至今（v5 v7.0 → 自定义数据集 → Ultralytics 家族）在 DLA 量化
部署中遇到的**所有算子级障碍**：哪些算子 standalone DLA 不支持、出现在哪、怎么解决的；
未解决的列入 TODO。分两类：**硬障碍**（构建直接失败）与**软约束**（构建能过但要求
量化覆盖，处理不当则精度受损或层被挤出 INT8）。

---

## 1. 总表

| # | 算子/结构 | 出现位置 | 类型 | 解决状态 |
|---|---|---|---|---|
| 1 | **Shape → Slice 链** | C2f/C3k2 的 `chunk(2,1)`（yolov8/11/26 骨干） | 硬 | ✅ 已解决 |
| 2 | **Constant 节点**（Slice 参数物化） | 同上（onnxsim 处理后残留） | 硬 | ✅ 随 #1 消失 |
| 3 | **注意力算子群**（MatMul/Softmax/Einsum 型，C2PSA/PSABlock） | yolo11/yolo26 的 neck（model.10） | 硬 | ❌ **TODO** |
| 4 | **Concat 无 INT8 scale** | 全部模型的拼接点 | 软 | ✅ 已解决 |
| 5 | **Mul（SiLU）无 scale** | 全部模型的激活点 | 软 | ✅ 已解决 |
| 6 | **残差 Add 无 scale / 双输入 scale 不齐** | 全部模型的 Bottleneck | 软 | ✅ 已解决（含一个导出 bug） |
| 7 | **图输出 conv 无 scale**（s8/s16/s32 生产者） | 全部模型的检测头 | 软（设计使然） | ✅ 按 FP16 处理 |
| 8 | **INT8 输入布局**（kHWC4/kCHW32/kLINEAR） | 部署侧 I/O | 格式约束 | ✅ MatX 重排 |
| 9 | **FP16 CHW16 布局** | 部署侧 I/O | 格式约束 | ✅ MatX 重排 |

注：Conv、Sigmoid、Mul、Add、Concat、MaxPool、Resize(Upsample-nearest)、Sub 等常规算子
standalone DLA **原生支持**（v5s/v5su/v8s 三个模型全图跑通即为实证）。障碍集中在
**结构性算子（Slice/注意力）** 和**量化覆盖要求**两类。

---

## 2. 硬障碍详情

### 2.1 Shape → Slice 链（C2f 的 chunk）— 已解决

- **出处**：Ultralytics C2f/C3k2 的 `y = self.cv1(x).chunk(2, 1)`；C3k2 继承 C2f 的
  forward，所以 yolo11/yolo26 同样命中
- **报错**（trtexec）：
  ```
  Layer '/model.2/Shape' is not supported on DLA but GPU fallback is not enabled.
  ```
- **排查过程中的发现**：
  - `onnxsim` 能把 Shape 消掉（24→0），但 **Slice 本身 standalone DLA 不支持**——
    Slice 的常量参数会被 TRT 物化成 Constant 层，同样被拒：
    ```
    layer '/model.2/Constant_1_output_0' is not supported on DLA
    ```
  - 即：图里只要残留 Slice，无论参数形式如何都过不了 standalone 校验
- **解法**（P1.5，`ultralytics/scripts/export_dla.py` 的 `make_dla_friendly`）：
  导出前把每个 C2f/C3k2 的 `cv1: Conv(c1→2c)` 沿输出通道**切成两个独立的
  `Conv(c1→c)`**（conv 权重、融合 bias、BN 四参数、逐通道 `_amax` 全部按半切开）。
  两个独立 Conv 节点数学上与"大 conv + chunk"完全等价，图里从此没有 Slice。
  内置导出前数值等价自检（**相对阈值** 1e-2，实测 ~4e-4 噪声级）。
- **附带收获**：v5su 用 C3 骨干（无 chunk），天然没有此问题——这是 v5su 在 DLA 上
  比原生 v8 还快 14% 的原因（无拆分 = 更少的层边界开销）。

### 2.2 注意力算子群（C2PSA / PSABlock）— 未解决

- **出处**：yolo11 / yolo26 的 `C2PSA`（第 10 层，SPPF 之后）及 `C3k2[..., True]`
  的 attn=True 瓶颈
- **报错**：
  ```
  layer '/model.10/m/m.0/attn/Constant_1_output_0' is not supported on DLA
  ```
  （注意力内部的 MatMul/Softmax/转置链在 standalone DLA 无实现；Constant 只是第一个
  被点名的）
- **影响**：yolo11s / yolo26s 至今无法上 standalone DLA
- **TODO（两条路，见下）**：
  1. `attn=False` 重训：改模型 yaml（C2PSA → 删除或换 C3k2、attn 标志置 False），
     服务器 finetune 10~50 epoch 恢复精度。代价 1~3 GPU 天，预期损失 ~0.5-1 AP
  2. 接受限制：v8 头的 DFL 解码和 `REG_MAX=1`（yolo26）在 C++ 侧**已经就绪**
     （decode_dfl.cu），纯差训练侧
- **不选 GPU fallback 的原因**：`--buildDLAStandalone` 禁止 GPU 回退（这正是 standalone
  的语义）；混合模式（`--useDLACore` + 允许 fallback）理论可过，但注意力块会落在 GPU
  上，违背 DLA 独占/并行的部署目标，且本项目未验证该路径（可作为 TODO-3 评估）

---

## 3. 软约束详情（量化覆盖）

DLA 与 GPU TensorRT 的关键差异：**图中每个算子都要有 INT8 scale**，GPU 上 TRT 允许
个别算子（如 Concat）以高精度运行，DLA 不允许。缺 scale 的层要么被推去 FP16（精度/
速度代价），要么构建失败（配合 `--precisionConstraints=obey`）。

### 3.1 Concat 无 scale — 已解决

- **原因**：Option#1（TensorRT 推荐 Q/DQ 布局）不给 Concat 配量化器
- **解法**（双轨）：
  - `qdq_translator --infer_concat_scales`：从邻层推导（v5 家族主线）
  - v5 覆盖层把 `torch.cat` 模块化成 `Concat`（common.py 的 35 行改动），使 Option#2
    （`--all-node-with-qdq`）能注册 `Concat → QuantConcat` 直接量化
- **特例**：检测头部的最终 Concat（输出 s8/s16/s32）永远没有 scale（图出口），
  连同其供给 conv 一起按 `--layerPrecisions` 强制 FP16 —— 设计使然，见 3.4

### 3.2 Mul / SiLU 无 scale — 已解决

- `--infer_mul_scales` 从邻层推导；v5 覆盖层同样准备了 `QuantSiLU`（Option#2）

### 3.3 残差 Add — 已解决（含一个隐蔽 bug）

- **要求**：INT8 加法要求两个输入**共享同一 scale**（否则无法对齐）
- **解法**：`QuantAdd`（单量化器同时量化两路）；v5 路线由 rules.py 把 addop 的量化器
  绑到 cv1 的（TRT Q/DQ folding 要求）
- **踩过的 bug**（P2 后期才暴露）：类级 `Bottleneck.forward` 补丁**不随 pickle 序列化**，
  导出进程加载 checkpoint 后 QuantAdd 静默失效 → cv2 链丢 Q → 瓶颈 conv 意外 FP16
  （v8s 6 层 / v5su 7 层），推理慢一倍。修复 = 导出时重应用类补丁
  （详见 [int8-fp16-ratio-analysis.zh-CN.md](int8-fp16-ratio-analysis.zh-CN.md)）

### 3.4 图输出 conv 无 scale — 按 FP16 处理（正确设计，勿"优化"）

- 检测头末层 1×1 conv（v5 的 m.X/Conv、v8 的 cv2.X.2/cv3.X.2）直供图输出，Option#1
  不会给输出配量化器
- 人工补 scale = 在网络最敏感处做 PTQ，是 v2 脚本（37.1→37.3）的反向操作；
  实测把 6 个头 conv + 3 个 Concat 留在 FP16 是精度-速度的正确平衡点
- 3 个头部 Concat 另受 `fp16:chw16` 输出格式约束，必须 FP16

### 3.5 I/O 布局约束 — 已解决（MatX 重排层）

| 约束 | 要求 | 解法 |
|---|---|---|
| INT8 输入 | `kDLA_LINEAR`/`kDLA_HWC4`/`kCHW32` | `ReformatImageV2`（CHW→HWC4）+ `convert_float_to_int8` |
| FP16 输入/输出 | `kCHW16` | `ReformatImage` / `Run`（CHW16↔平面，通道补齐到 16/32 倍数） |

这就是 `src/matx_reformat/` 存在的全部原因——算子本身 DLA 支持，是**内存布局**不匹配。

---

## 4. TODO 清单

| # | 事项 | 前置条件 | 预估 |
|---|---|---|---|
| 1 | **yolo11/yolo26 注意力**：`attn=False` + 去 C2PSA 的 yaml，finetune 恢复精度后走现有全链路 | 服务器 GPU | 配置 0.5h + 训练 1~3 GPU 天 |
| 2 | yolo26 e2e/one2one 头验证：当前走 one2many+NMS（C++ 解码已就绪，含 REG_MAX=1 分支）| 无（随 #1） | — |
| 3 | （可选）评估混合模式 GPU fallback 跑注意力块的可行性与性能代价 | 无 | 0.5 天 |
| 4 | （可选）验证超采样/量化无关的算子边界：Transpose/Reshape 在 head 内的表现（当前由 MatX 侧吸收，ONNX 侧未单测） | 无 | 低优先 |

---

## 5. 各模型最终算子清单（修复后实测可跑通的图）

| 模型 | 图内算子 | 备注 |
|---|---|---|
| yolov5s v7.0 | Conv, Sigmoid, Mul, Add, Concat, MaxPool, Resize | 全部 DLA 原生支持 |
| yolov5su | 同上 | C3 骨干，无 Slice/注意力 |
| yolov8s 原生 | 同上（chunk 拆分后无 Slice） | C2f 拆分 = 层数↑、单层宽↓ |
| yolo11s / yolo26s | 同上 + **注意力群** | ❌ 被注意力阻断 |
