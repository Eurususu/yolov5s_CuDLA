# Ultralytics 系列模型支持方案（设计文档）

目标：在现有 cuDLA 部署链路上支持 **Ultralytics 现代模型家族**（v5u / v8 / v11 / v12 / v26 等，
已验证源码 `ultralytics/` @ 8.4.142），同时**完全兼容**现有的 yolov5 v7.0 路线。

> 事实均核对于本仓库 `ultralytics/` 源码（行号可跳转）；
> 现有链路知识见根 CLAUDE.md 两条流水线与 yolov5_dla/docs/qat-internals.zh-CN.md。

---

## 1. 为什么不能直接用：三种检测头的本质差异

| | **v5 v7.0**（现支持） | **v8 / v11 / v5u**（主流） | **v26**（end2end） |
|---|---|---|---|
| 检测范式 | anchor-based | **anchor-free + DFL** | anchor-free + DFL + one2one |
| 每层头通道 | `3 × (nc+5)` | **`64 + nc`**（`4×reg_max(16)` 框分布 + nc 类别，[head.py:120](../ultralytics/nn/modules/head.py)） | 同左 ×2 套头 |
| objectness | 有（×obj） | **无** | 无 |
| 框解码 | sigmoid(tx,ty)×2−0.5+grid；tw×2 平方×anchor | **DFL**：16-bin softmax 加权和 → ltrb 距离 → dist2bbox（[block.py:60](../ultralytics/nn/modules/block.py)、[tal.py:441](../ultralytics/utils/tal.py)） | 图内完成 + postprocess |
| 类别分数 | sigmoid×obj | 纯 sigmoid | 图内完成 |
| NMS | 需要 | 需要 | **不需要**（one2one 训练免 NMS） |

**DFL 解码数学（C++ kernel 需要实现的全部）**，已从源码验证：

```
每层原始输出:  box分支 [bs,64,H,W]（4 边 × 16 bin） + cls分支 [bs,nc,H,W]
DFL:          d_side = Σ_i softmax(16bin)[i] × i          （block.py:78，conv 权重=0..15）
dist2bbox:    x1=(gx+0.5−l)·s, y1=(gy+0.5−t)·s, x2=(gx+0.5+r)·s, y2=(gy+0.5+b)·s
              （gx,gy 为网格坐标，s 为 stride 8/16/32；tal.py dist2bbox + make_anchors offset 0.5）
score:        sigmoid(cls)，无 objectness 相乘
```

对比 v5 解码（现 [decode_nms.cu](../src/decode_nms.cu)）：无 anchor 表、无 obj、
通道布局不同 —— **解码 kernel 必须分家**，其余全部环节可复用。

## 2. 各环节兼容性盘点

| 环节 | v5 v7.0 | v8 家族 | 改动量 |
|---|---|---|---|
| 预处理（letterbox/RGB//255/32 倍数） | ✓ | **完全相同** | 0 |
| cuDLA 上下文 / 双模式 / NvSci | ✓ | 无关模型 | 0 |
| trtexec INT8 HWC4 + FP16 CHW16 约定 | ✓ | 相同 | 0（仅 `--shapes` 参数化，已有） |
| qdq_translator（图级 Q/DQ→PTQ） | ✓ | 通用 | 0 |
| **ONNX 导出（原始多头输出）** | `--noanchor` 补丁 | 需新补丁：monkeypatch `Detect.forward` 输出 per-level `[64+nc,H,W]`（等价于 noanchor；官方导出默认是解码后单输出 `[1,4+nc,8400]`） | 中 |
| **matx 重排通道公式** | `3×(nc+5)` | `64+nc` | 小（公式按 head 风格切换） |
| **decode kernel** | decode_nms.cu | 新 decode_dfl.cu（上文数学） | **大**（核心工作） |
| prior 表 | 9 组 anchor | 仅 stride/grid | 小 |
| `--layerPrecisions` 节点名 | `/model.24/m.X/Conv` | 因模型而异（如 `/model.22/cv2.X/...`），从 translator 的 precision_config 里取 | 小（流程不变） |
| QAT 工具链（pytorch-quantization 替换表） | yolov5_dla（对 v7.0 模块集） | ultralytics 模块集（Conv/C2f/C2k2/SPPF/Concat/SiLU…），需新建 `ultralytics_dla` | **大** |
| make_coco_json / 评估链路 | ✓ | 通用 | 0 |

## 3. 关键技术风险（提前排雷）

1. **DFL conv 的量化敏感性**：cv2 末层 1×1 输出的 16-bin 分布对量化误差敏感（softmax 会放大）。
   惯例：`cv2.*` 末层 + DFL 保持 FP16（对应现 `--layerPrecisions` 的扩展，由 translator 的
   missing-scale 报告自动给出候选清单 —— 现有机制正好复用）。
2. **yolo26 end2end**：`one2one` 头 + 图内 postprocess（topk）。topk 类算子 DLA 支持存疑，
   建议 v26 走"导出 one2many 原始头 + C++ 解码"路线（与 v8 同路径），或评估 TRT 层面 DLA+GPU
   混合。**放到最后做**。
3. **C2f/C3k2 的 Split/Concat**：DLA 要求全算子有 scale —— 即 Option#2 风格
   （`--all-node-with-qdq`）在 ultralytics 模块集上的等价实现，替换表要覆盖 `C2f` 内部的 cat。
4. **reg_max 可能非 16**（配置可改）：kernel 里按宏参数化 `YOLO_REG_MAX`，默认 16。

## 4. C++ 侧设计（兼容两种风格）

```
make HEAD_STYLE=v5    （默认，现 anchor-based 路线，零改动）
make HEAD_STYLE=v8    （新：anchor-free DFL）
    ├─ YOLO_HEAD_STYLE 宏 → 通道公式切换:
    │     v5: kChPerAnchor = nc+5, kAnchorsPerPos = 3
    │     v8: kChPerAnchor = reg_max*4+nc, kAnchorsPerPos = 1
    ├─ decode 分派: decode_nms.cu (v5) / decode_dfl.cu (新)
    ├─ prior 表: v5 带 anchor；v8 仅 (gx, gy, stride)
    └─ matx 通道分组推导同步切换（kChw16Groups 公式输入变）
```

## 5. 分期落地计划

| 阶段 | 内容 | 交付物 | 预估 |
|---|---|---|---|
| **P0 导出验证** | 给 ultralytics 写 raw-head 导出补丁（monkeypatch Detect.forward，输出 s8/s16/s32 = [64+nc,H,W]）；用官方 yolov8s.pt 导出并用 trtexec 编 FP16 loadable | 720p/672 v8 FP16 loadable + 导出脚本 | 半天 |
| **P1 C++ v8 解码** | HEAD_STYLE 参数化 + decode_dfl kernel + matx 通道公式 → **v8s FP16 端到端**（FP16 路不需要 QAT，最快见到检测结果） | v8 FP16 检测跑通 + 回归 v5 不受影响 | 1~2 天 |
| **P2 v8 QAT** | `ultralytics_dla` 工具包（替换表 + QuantAdd/Concat 模块化 + qat.py 对 ultralytics trainer 的适配）→ INT8 全链路 | v8 INT8 + mAP 归因 | 3~5 天 |
| **P3 yolo26 / v11 / v12** | 复用 P1/P2（同头风格，仅 yaml 差异）；v26 end2end 单独评估 | 各家族验证记录 | 按需 |

## 6. P0 实测结果（2026-09-07）

**导出**：`ultralytics/scripts/export_dla.py`（monkeypatch `Detect.forward` 输出 per-level 原始头）已验证四家族：
v8s / v11s / v5su 输出 `[1,144,H,W]`（reg_max=16，64+80），**yolo26s 为 `[1,84,H,W]`（reg_max=1，无 DFL，直接回归 4 距离）** —— 解码 kernel 需按 reg_max 参数化。

**DLA 构建（FP16, 672, standalone）**：

| 模型 | 结果 | 原因 |
|---|---|---|
| **yolov5su** | ✅ **PASSED**（19.4MB） | C3 骨干无 Shape/Slice，算子集与 v5 v7.0 同级 DLA 友好 |
| yolov8s / yolo11s / yolo26s | ❌ 被 `C2f/C3k2` 的 `chunk(2,1)` 挡住 | 导出成 Shape→Slice 链；onnxsim 可消掉 Shape，但 **Slice 本身不被 standalone DLA 支持**（常量参数会被 TRT 物化为 Constant 层） |

**v8 系的解法（P1 议题）**：① 图手术 —— 把通道维 Slice 折进后继 Conv 的权重（沿输入通道拆分 weight）；② 导出时补丁 C2f（cv1 拆成两个独立 Conv）；③ 直接用 v5su（v8 头 + C3 骨干，**已可部署**）。短期用 ③，中期做 ①。

## 7. 兼容性保证

- `HEAD_STYLE` 默认 v5：现有全部脚本/文档/验证过的模型**零变化**
- 新代码全部走新文件（decode_dfl.cu 等）+ 宏分派，不触碰 v5 路径
- 每阶段以"现有 672 三分类模型回归不变"为验收门槛
