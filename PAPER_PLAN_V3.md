# 桥梁点云构件分割论文框架 v0.3

日期：2026-09-18

状态：以构件分割精度为核心、以桥型适用范围为边界的实验驱动版本。

---

## 一、核心目标

本研究的首要目标不是提出一个通用增强模块，而是：

> 在常规梁桥点云上，准确分割 `girder`、`pier/support`、`deck`
> 三类构件，并在 0%、25%、50%、75% 受控点缺失条件下保持可用精度。

镜像 TTA、拓扑顺序损失和跨 family 评价都只服务于这个目标。

主指标：

- mIoU；
- `girder` IoU；
- `pier/support` IoU；
- `deck` IoU；
- 混淆矩阵。

任何方法是否保留，都以是否改善三类构件分割为准，而不是以 gate
是否复杂或是否看起来新颖为准。

---

## 二、适用桥型

第一阶段只处理：

- 常规、近直梁桥；
- 横桥向近似对称；
- 构件集合包含 `girder / pier-support / deck`；
- 不包含悬索桥、斜拉桥、复杂拱桥、曲线桥、大斜交桥；
- 不包含拉索、支座、栏杆等长尾构件。

第一版 label-free scope manifest：

- mirror eligible：`c-bridge4`、`c-bridge5`
- mirror ineligible：`c-bridge1`、`c-bridge2`、`c-bridge3`、`s-bridge1`
- evidence threshold：`0.01`
- eligible scene fraction threshold：`0.25`

文件：

- `work/brpcd/family_scope_manifest_v1.json`
- `work/brpcd/family_scope_manifest_v1.csv`

该 manifest 必须在外测前冻结，后续不能根据测试标签重新调类。

---

## 三、当前主方法

### 3.1 主干与训练

- 主干：轻量 PointNet。
- 类别均衡采样 + weighted focal loss。
- 可选拓扑顺序损失：`deck > girder > pier`。
- PointNet++ 当前实现和训练配方是负结果，不作为主线。

### 3.2 镜像 TTA 的角色

镜像 TTA 不是通用增强，而是适用桥型上的辅助推理模块：

- 在 mirror-eligible family 上可以启用固定镜像融合；
- 在 mirror-ineligible family 上关闭镜像 TTA，使用原视图预测；
- 不把 scene-level geometry gate、confidence gate、agreement gate
  或 classwise calibration 作为主方法。

原因：

- 固定 TTA 在非适用 family 上明显有害；
- 复杂 gate 没有超过同分布随机对照；
- 桥型级适用范围比 per-scene 黑箱门控更符合当前实验证据。

### 3.3 推荐推理策略

主方法固定为：

1. 使用 topology-trained PointNet 作为统一分割模型；
2. 当桥型属于 `mirror_eligible` 时，使用 fixed 50/50 镜像 TTA；
3. 当桥型属于 `mirror_ineligible` 时，关闭镜像 TTA，使用原视图。

该策略的目标不是让所有 family 都提升，而是：

- 在适用桥型上保持较高的构件分割精度；
- 在不适用桥型上避免镜像 TTA 造成的明显退化；
- 让方法边界和工程适用范围清晰可审计。

---

## 四、当前最关键结果

### 4.1 P1 主划分

测试 family：`c-bridge4`。

| 方法 | 0% | 25% | 50% | 75% |
|---|---:|---:|---:|---:|
| 拓扑训练，无 TTA | 0.5096 | 0.4235 | 0.3484 | 0.2617 |
| 拓扑训练 + 固定镜像 TTA | 0.5200 | 0.4765 | 0.4319 | 0.3878 |

这说明在适用桥型上，镜像 TTA 可以明显改善缺失点云下的构件分割。

### 4.2 桥型适用范围

| Scope | 0% | 25% | 50% | 75% |
|---|---:|---:|---:|---:|
| mirror eligible，fixed 相对 none | +0.0068 | +0.0028 | -0.0001 | +0.0008 |
| mirror ineligible，fixed 相对 none | -0.0439 | -0.0348 | -0.0341 | -0.0272 |

这说明固定镜像 TTA 应只用于适用桥型。

### 4.3 适用 family 上的构件分割主表

适用范围：`c-bridge4`、`c-bridge5`。

| 方法 | 0% mIoU | 25% | 50% | 75% |
|---|---:|---:|---:|---:|
| augmentation fixed | 0.3855 | 0.3734 | 0.3547 | 0.3225 |
| augmentation geometry | 0.3847 | 0.3725 | 0.3549 | 0.3195 |
| topology fixed | 0.4123 | 0.3821 | 0.3504 | 0.3133 |
| topology geometry | 0.4121 | 0.3821 | 0.3508 | 0.3158 |
| topology none | 0.4055 | 0.3793 | 0.3505 | 0.3126 |

解释：

- 0% 和 25% 缺失时，topology + fixed TTA 的 mIoU 最好；
- 50% 缺失时，topology + geometry TTA 略好；
- 75% 缺失时，augmentation fixed 的 mIoU 更高，但 topology 方法在
  `deck` 和 `girder` 上不差，trade-off 需要按任务需求报告。

### 4.4 三类构件 IoU

0% 缺失时：

| 方法 | girder | pier | deck |
|---|---:|---:|---:|
| augmentation fixed | 0.1919 | 0.9074 | 0.0572 |
| topology fixed | 0.2264 | 0.8746 | 0.1359 |
| topology geometry | 0.2267 | 0.8730 | 0.1367 |

75% 缺失时：

| 方法 | girder | pier | deck |
|---|---:|---:|---:|
| augmentation fixed | 0.1633 | 0.7770 | 0.0272 |
| topology fixed | 0.1826 | 0.7039 | 0.0536 |
| topology geometry | 0.1823 | 0.7029 | 0.0622 |
| topology none | 0.1797 | 0.6896 | 0.0683 |

这比只看 mIoU 更重要：如果目标是准确分割所有构件，
必须同时报告 pier 和 deck 的 trade-off。

### 4.5 In-Scope 与 Out-of-Scope 的推理选择

在 mirror-ineligible family 上：

- `topology_none` 和 `topology_geometry` 结果相同，因为几何判据关闭了镜像；
- `topology_fixed` 明显低于两者；
- 因此 out-of-scope 的推荐策略是 no mirror TTA。

在 mirror-eligible family 上：

- `topology_fixed` 在 0%、25% 缺失率最好；
- `topology_geometry` 在 50% 缺失率略好；
- 75% 缺失率下存在 `pier` 与 `deck` 的 trade-off；
- 因此主策略采用 `topology_fixed`，并把 geometry TTA 作为消融。

### 4.6 总体推理策略结果

按推荐策略复算：

`scope_fixed = eligible family 用 topology + fixed TTA，ineligible family 用 topology + no TTA`

| Strategy | 0% | 25% | 50% | 75% |
|---|---:|---:|---:|---:|
| always_none | 0.3801 | 0.3582 | 0.3305 | 0.2911 |
| always_fixed | 0.3531 | 0.3360 | 0.3077 | 0.2733 |
| scope_fixed | 0.3823 | 0.3592 | 0.3304 | 0.2914 |
| scope_geometry | 0.3823 | 0.3592 | 0.3305 | 0.2922 |

`scope_fixed` 相对 `always_none` 的变化为
`+0.0023 / +0.0009 / -0.0000 / +0.0003`。

虽然总体平均增益很小，但它的意义是：在适用桥型上保留构件分割收益，
同时避免在非适用桥型上使用错误镜像视图。该策略应作为最终可执行方案，
而不是继续寻找更复杂的 scene-level gate。

### 4.7 配对 bootstrap 统计

按 family/seed 配对并做 family-level bootstrap：

| Comparison | 0% | 25% | 50% | 75% |
|---|---:|---:|---:|---:|
| in-scope fixed - none | +0.0068 | +0.0028 | -0.0001 | +0.0008 |
| out-of-scope fixed - none | -0.0439 | -0.0348 | -0.0341 | -0.0272 |
| in-scope geometry - none | +0.0066 | +0.0028 | +0.0002 | +0.0032 |
| scope fixed - always none | +0.0023 | +0.0009 | -0.0000 | +0.0003 |

95% bootstrap CI：

- in-scope fixed - none 在 0%、25% 下为正；
- out-of-scope fixed - none 在四个缺失率下均为负；
- in-scope geometry - none 在四个缺失率下均为正，但 in-scope 只有两个
  family，CI 只能作为探索性证据；
- scope fixed - always none 在 0%、25% 下为正，50%、75% 基本持平。

因此主结论应采用 scope-aware fixed TTA，而不是 unconditional fixed TTA。

---

## 五、论文主 claim

可以写：

1. 在常规梁桥的适用范围内，所提流程能准确分割
   `girder / pier / deck` 三类构件。
2. 镜像 TTA 在适用桥型上提高缺失点云下的分割鲁棒性；
   在非适用桥型上会造成明显退化。
3. 桥型级适用性比 scene-level 或 point-level 复杂门控更符合实验证据。
4. 拓扑顺序是可选辅助模块，在部分适用桥型和少数类构件上有帮助，
   但不能作为跨桥型通用创新。

不能写：

- 通用桥梁分割；
- 所有桥型都适用镜像 TTA；
- geometry/pointwise/classwise gate 是独立新机制；
- 拓扑顺序跨 family 稳定有效；
- 2D-3D、生成或对称增强本身是本文首次提出。

---

## 六、需要补的实验

1. 按冻结 manifest 重算 in-scope / out-of-scope 主表。
2. 补齐 girder、pier、deck 三类 IoU 和混淆矩阵。
3. 对适用 family 做配对统计和 bootstrap 置信区间。
4. 比较 BridgeNetv2、weighted superpoint graph、view consensus。
5. 在独立桥梁数据上验证适用范围定义是否仍成立。
6. 若 75% 缺失率下 pier/deck trade-off 增大，单独报告两类失败案例。

---

## 七、论文组织

### Abstract

强调构件分割精度、跨 family 适用边界和缺失点云鲁棒性。

### Introduction

从桥梁数字孪生和构件分割需求切入，不先讲 TTA。

### Related Work

按以下顺序：

1. bridge point-cloud component segmentation；
2. 2D-3D multimodal fusion；
3. synthetic/generative bridge point clouds；
4. symmetry, invariance and test-time augmentation。

### Method

1. 桥梁坐标规范；
2. PointNet 分割主干；
3. 可选构件顺序损失；
4. applicability-aware mirror TTA；
5. in-scope / out-of-scope 推理流程。

### Experiments

1. P1 component segmentation；
2. LOBO family holdout；
3. scope-aware main table；
4. per-class IoU and confusion matrix；
5. external baseline。

### Discussion

重点讨论：

- 为什么镜像 TTA 不能通用；
- 哪些桥型可以准确分割；
- pier 与 deck 的 trade-off；
- 模型适用范围和工程部署边界。

---

## 八、当前结论

当前最可信的研究主线是：

> 在常规梁桥构件分割任务中，优先保证 `girder / pier / deck` 三类构件
> 的准确分割；镜像 TTA 只在符合适用条件的桥型上启用，不适用桥型退回
> 原视图。拓扑顺序和几何门控降为辅助或消融模块。
