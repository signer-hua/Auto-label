# Auto-label：人机协同图像自动标注工具

基于 **Grounding DINO + SAM3 + DINOv3** 三大核心算法的工业级单人标注系统。
采用 **FastAPI + Celery + Redis** 异步架构，GPU 推理不阻塞 API 请求。
前端基于 **React + TypeScript + Zustand + react-konva** 实现三层 Canvas 交互。

## 单人闭环工作流

```
上传图片集 → 选择多参考图(2~5张) → 批量自动标注 → 按置信分数筛选
    → 手动修正低分/错误标注(矩形→SAM3/负向框选/撤销/重做) → 一键导出(COCO/VOC/YOLO)
```

## 三大标注模式

| 模式 | 名称 | 链路 | 适用场景 |
|------|------|------|----------|
| 模式1 | 文本提示标注 | 文本 → Grounding DINO → SAM3 | 已知类别名称，一键全图标注 |
| 模式2 | 框选批量标注 | 框选 → SAM3-PCS → MRF融合 → DINOv3 匹配 → 批量 SAM3 | 未知类别，标一个扩展到全部 |
| 模式3 | 选实例跨图标注 | DINOv3 聚类 → 选实例 → MRF融合 → DINOv3 匹配 → SAM3 | 视觉选择，跨图批量扩展 |

---

## v9.1 橡皮擦实时反馈修复 + 重置确认弹窗

### 橡皮擦修复

| 修复项 | 说明 |
|--------|------|
| **实时视觉反馈** | 拖动擦除时画布实时绘制粉色圆圈轨迹，清晰可见擦除区域 |
| **Mask 强制刷新** | 擦除完成后通过 URL 缓存击破（`?v=N`）强制浏览器重载 Mask 图片 |
| **全 Mask 擦除** | 遍历当前图片所有 Mask 逐一擦除，不再仅处理最后一个 |
| **依赖数组修复** | `handleMouseDown/Move/Up` 补全 `canErase`/`imageFit`/`isErasing` 依赖 |
| **图标区分** | 橡皮擦用 `HighlightOutlined`，清空用 `DeleteOutlined`，不再混淆 |
| **模式指示** | 画布左上角橡皮擦模式显示粉色"橡皮擦"标签 |

### 重置确认弹窗

| 功能 | 说明 |
|------|------|
| **二次确认** | 点击「重置全部」弹出 Popconfirm 确认框，防止误触 |
| **说明提示** | 弹窗显示"将清空所有标注、类别、参考图，并删除缓存文件。不可撤销" |
| **确认后执行** | 点击"确认重置"才执行清理，点击"取消"无操作 |

---

## v9.0 精细化橡皮擦 + 导出去重 + 缓存清理

### 精细化橡皮擦

| 功能 | 说明 |
|------|------|
| **像素级擦除** | 鼠标拖动沿轨迹以指定半径圆形区域擦除 Mask 像素 |
| **大小可调** | Slider 调节 3~50px 橡皮擦半径，实时预览 |
| **后端处理** | `POST /api/annotate/erase` 接收轨迹坐标，OpenCV 圆形区域置零 |
| **非破坏性** | 仅修改 Alpha 通道像素，保留其余有效区域 |

### 导出重叠去重

| 功能 | 说明 |
|------|------|
| **同类别 IoU 检测** | 导出时对同图片同类别标注计算 IoU |
| **自动合并** | IoU ≥ 0.3 的标注合并为最小外接矩形，仅保留一条记录 |
| **全格式生效** | COCO/VOC/YOLO 导出均经过去重，避免重复标注 |

### 缓存清理

| 功能 | 说明 |
|------|------|
| **重置即清理** | 点击「重置全部」自动调用 `POST /api/clean_cache` |
| **精准清理** | 递归删除 `data/exports/` 和 `data/masks/` 所有文件，保留 `data/images/` |
| **Redis 清理** | 同时删除所有 `task:*` 前缀的 Redis 键 |
| **异常安全** | 文件被占用时跳过并记录日志，不中断流程 |

### 新增/修改文件

| 文件 | 说明 |
|------|------|
| `backend/api/routes.py` | `POST /api/clean_cache` + `POST /api/annotate/erase` + 导出去重 |
| `frontend/src/api/index.ts` | `cleanCache` + `eraseMaskRegion` API |
| `frontend/src/components/Toolbar.tsx` | 橡皮擦按钮 + 大小调节 + 重置清理 |
| `frontend/src/components/MainCanvas.tsx` | 橡皮擦交互（轨迹收集+提交） |

---

## v8.0 ProMerge 实例分割 + LoRA 微调 + 效果验证

### ProMerge 谱聚类实例分割（替换 K-Means）

| 功能 | 说明 |
|------|------|
| **亲和矩阵计算** | 基于 DINOv3 patch 特征余弦相似度构建亲和矩阵 |
| **谱聚类** | `SpectralClustering`（Normalized Cut）替代 K-Means，保留空间连续性 |
| **scikit-image 后处理** | `label` + `regionprops` 过滤面积 < 1% 的碎片，提升实例完整性 |
| **K-Means 兜底** | ProMerge 失败自动降级至原 K-Means 聚类，功能不降级 |
| **新增依赖** | `scikit-image>=0.22.0`（PyPI 安装） |

### SAM3 LoRA 参数高效微调

| 功能 | 说明 |
|------|------|
| **仅微调 mask_decoder** | 骨干网络冻结，LoRA 参数约 0.5M，峰值显存 ≤ 4GB |
| **配置** | `r=8, lora_alpha=32, dropout=0.05`，适配消费级 GPU |
| **训练优化** | `batch_size=1, grad_accum=4, fp16=True`，RTX3060 可训练 |
| **前端入口** | 高级工具面板：选类别 → 选图片（≥5张）→ 设轮次 → 启动微调 |
| **推理加载** | `load_lora_weights` 自动检测类别 LoRA 权重并加载 |
| **新增依赖** | `peft>=0.6.0, accelerate>=0.20.0`（PyPI 安装，可选） |

### 效果验证工具

| 方法 | 说明 |
|------|------|
| `calculate_instance_completeness` | 实例数、总覆盖率、平均面积、碎片率统计 |
| `calculate_mask_miou` | 单 Mask IoU 计算 |
| `calculate_batch_miou` | 批量 mIoU + 逐实例 IoU 列表 |

### 新增/修改文件

| 文件 | 说明 |
|------|------|
| `backend/models/dino_engine.py` | ProMerge 谱聚类 + K-Means 兜底 |
| `backend/models/sam_engine.py` | `load_lora_weights` 方法 |
| `backend/utils/lora_finetune.py` | LoRA 配置 + 训练逻辑 |
| `backend/utils/eval_utils.py` | 完整性 + mIoU 验证 |
| `backend/api/routes.py` | `/api/annotate/start_lora_finetune` 接口 |
| `backend/worker.py` | `process_lora_finetune` Celery 任务 |
| `frontend/src/api/index.ts` | `startLoraFinetune` API |
| `frontend/src/components/AdvancedTools.tsx` | LoRA 微调面板 |
| `backend/requirements.txt` | 新增 scikit-image, peft, accelerate |

---

## v7.5 跨图颜色同步 + 手动标注持久化 + 特征融合

### 跨图标注颜色统一

| 修复项 | 说明 |
|--------|------|
| **图片级解析色传递** | `handleMode3Select` 使用 `getResolvedColor` 传递参考图绑定色（而非全局类别色），目标图 Mask 颜色与参考图完全一致 |
| **多类别逐个解析** | 多类别模式下，每个类别独立调用 `getResolvedColor` 获取图片级映射色 |

### 手动标注 Mask 持久化

| 修复项 | 说明 |
|--------|------|
| **`mergeMaskUrls` 合并操作** | 跨图标注完成时合并新结果而非替换，保留参考图已有的手动标注 Mask |
| **RightPanel 轮询修复** | 任务成功回调使用 `mergeMaskUrls` 替代 `setMaskUrls`，不再清空参考图状态 |

### 手动标注纳入参考特征库

| 功能 | 说明 |
|------|------|
| **`manual_mask_urls` 参数** | 前端传递参考图所有手动 Mask URL 到后端 |
| **特征提取** | Worker 从手动 Mask PNG 提取二值 Mask → DINOv3 特征 |
| **加权融合** | 自动实例权重 0.7 + 手动实例权重 0.3，融合后再跨图匹配 |
| **准确率提升** | 手动标注补充了自动实例未覆盖的区域，跨图匹配更完整 |

---

## v7.4 图片级类别视觉标识色映射

### 核心逻辑

同一图片下，类别的视觉标识色**锁定为首个关联实例的原始发现颜色**，后续同类别手动标注强制复用该色。

```
实例#0 原始色 rgb(255,80,80) → 分配到「cat」→ 当前图片「cat」视觉色锁定为 #FF5050
  → 手动标注选「cat」→ 预览框/Mask 均为 #FF5050
  → 实例#2 也分配到「cat」→ 同样使用 #FF5050（不覆盖）
```

### 图片级映射体系

| 功能 | 说明 |
|------|------|
| **`imageCategoryColorMap`** | `{imageId: {categoryId: hex}}` 图片级类别-视觉色映射 |
| **`originalColor`** | `InstanceItem` 新增属性，持久化实例生成阶段的初始色 |
| **`bindCategoryColor`** | 首个实例分配类别时锁定映射，后续不覆盖 |
| **`getResolvedColor`** | 优先级：图片级映射 > 全局类别色 > 默认 #CCCCCC |
| **手动标注统一** | SAM3 Mask 请求使用 `getResolvedColor` 而非全局类别色 |
| **画布预览统一** | 手动标注矩形、高亮框、实例按钮均使用解析后颜色 |

---

## v7.3 实例-类别-颜色强绑定

### 模式3 实例颜色实时同步

| 功能 | 说明 |
|------|------|
| **InstanceItem 扩展** | 新增 `categoryId`（默认 null）、`categoryColor`（默认 null），与实例 ID 强绑定 |
| **分配即同步** | `setMode3CategoryInstances` 同时更新 `instanceMasks` 的颜色属性，画布图层立即重渲染 |
| **类别颜色高亮框** | 已分配实例在画布叠加类别色边框（选中实线/未选中虚线），与 Mask 预览并存 |
| **未分配灰化** | 未分配类别的实例 opacity=0.35，已分配=0.7，选中=1.0，视觉层级清晰 |

### 选中实例自动同步类别

| 功能 | 说明 |
|------|------|
| **点击即联动** | 画布点击已分配实例 → `activeCategoryId` 自动切换到该实例绑定的类别 |
| **手动标注跟随** | 手动标注类别选择器自动显示为实例所属类别，颜色完全一致 |
| **工具栏联动提示** | 手动标注区显示"手动标注→类别名（实例#N同色）"确认颜色匹配 |

### 手动标注矩形预览颜色

| 功能 | 说明 |
|------|------|
| **类别色矩形** | 手动标注框选预览使用当前类别颜色（替代硬编码橙色），与最终 Mask 颜色一致 |
| **颜色优先级** | 实例绑定类别 ＞ 手动选择类别 ＞ 默认橙色 |

---

## v7.2 坐标修复 + 颜色统一 + 交互逻辑完善

### 模式2 标注失败修复

| 修复项 | 说明 |
|--------|------|
| **preprocess_image 缩放致 bbox 失配** | 移除 `preprocess_image` 中的 `resize_image` 调用。SAM3/DINOv3 各自处理输入尺寸，在预处理阶段缩放会导致后续 bbox（原图坐标）与缩放后图像不匹配，SAM3 生成空 Mask → "No valid category references found" |
| **confirmMode2Bbox 坐标转换** | 确认框选时将画布坐标（Stage 空间）转为原图坐标再存储，消除渲染双重变换偏移 |
| **单框提交坐标修正** | `handleMode2Annotate` 单框模式提交前做画布→原图坐标转换，确保后端收到正确像素坐标 |

### 模式3 分配交互优化

| 修复项 | 说明 |
|--------|------|
| **分配后清空已选** | 点击"分配到类别"后自动清空已选实例列表，按钮自然禁用，需重新选择实例才能再次分配 |
| **已分配状态可视化** | 实例按钮显示 `✓` 标记 + 类别颜色边框 + 悬浮提示已分配类别名 |
| **分配计数提示** | 已分配状态下显示"已分配 N 个实例，可继续选择或手动标注补充" |

### 全模式类别颜色统一

| 修复项 | 说明 |
|--------|------|
| **颜色传递链路** | 前端将用户定义的类别颜色 hex 传入后端（`category_color` 字段） |
| **模式2/3 Mask 颜色** | Worker 使用传入颜色渲染 Mask，不再使用硬编码 `MULTI_CATEGORY_COLORS` |
| **手动标注颜色一致** | 手动标注区显示当前类别色块 + 名称，确保手动/自动标注同类别颜色完全一致 |
| **模式3 工作流** | 分配实例到类别 → 手动标注补充（同色）→ 跨图标注（同色），全链路颜色统一 |

---

## v7.0 核心精度提升 + 交互修复

### SAM3-PCS 范式激活（模式2 精度核心）

| 功能 | 说明 |
|------|------|
| **PCS 混合推理** | `generate_mask_from_exemplars` 方法：图像示例+坐标+文本提示联合推理 |
| **多Mask并集融合** | 启用 `multimask_output`，高分候选取 union 提升完整性 |
| **半精度推理** | `torch.autocast(float16)` 适配消费级 GPU |
| **文本提示补充** | 模式2 新增文本提示输入框，辅助语义引导 |

### MRF 多参考融合器

| 功能 | 说明 |
|------|------|
| **通道加权 MLP** | 学习特征通道重要性，强化判别性通道 |
| **自注意力模块** | `MultiheadAttention` 实现跨参考图特征交互 |
| **低层/高层融合** | DINOv3 层9（低层语义）+ 层12（高层抽象）加权合并 |
| **纯 PyTorch 实现** | 无新增依赖，适配半精度推理 |

### ACT/ACF 动态阈值体系

| 模块 | 说明 |
|------|------|
| **ACT 自适应阈值** | 高斯核密度估计（KDE）分析相似度分布，自动确定前景/背景分界点 |
| **ACF 置信过滤** | 分位数动态过滤低置信结果，适应不同数据集分布 |
| **纯数值计算** | 仅依赖 numpy/scipy，无新增依赖 |

### 全模式显存优化

| 优化项 | 说明 |
|--------|------|
| **输入分辨率限制** | 图片长边统一缩放至 1024px（OpenCV 等比缩放） |
| **混合精度推理** | DINOv3 `.half()` 加载，SAM3 `autocast(float16)` |
| **显存峰值 ≤ 5GB** | 适配 RTX 3060/3090 消费级 GPU |

### 置信度评分重构（三维评分）

| 维度 | 权重 | 说明 |
|------|------|------|
| 特征匹配度 | 40% | DINOv3 余弦相似度（ACT 动态阈值优化） |
| Mask 合理性 | 40% | 覆盖率 + 面积合理性综合评估 |
| 形态完整性 | 20% | OpenCV 计算孔洞率 + 连通域数量 |

### 人机协同负向提示

| 功能 | 说明 |
|------|------|
| **负向框选工具** | 红色框选排除错误区域，SAM3 重新生成修正 Mask |
| **正向+负向组合** | 正向框保留目标，负向框排除噪声，精准修正 |
| **实时更新** | 框选后立即调用后端修正，Mask 实时替换 |
| **`/api/annotate/correct`** | 新增修正接口，传入正向框+负向框+标签 |

### Celery 队列优先级分离

| 队列 | 任务类型 | 说明 |
|------|----------|------|
| `high_priority` | 框选/预览/手动标注/负向修正 | 实时交互，低延迟 |
| `low_priority` | 批量标注（模式1/2/3） | 后台处理，不阻塞交互 |

---

## v6.0 手动标注类别 + 已标注图参考

### 手动标注类别选择

- 手动标注工具区新增**类别选择器**，标注前选择类别后 Mask 颜色与该类别统一
- 手动标注矩形提交时自动携带 `category_color` + `category_name`
- 后端 `process_manual_sam` 使用传入颜色渲染 Mask（不再固定橙色）
- 未选类别时显示黄色警告提示

### 已标注图片作为参考图

- 右侧面板点击**图钉**将任意图片加入参考图库（包括已自动/手动标注过的）
- 模式2/3 标注时，参考图库中**已有 Mask 的图片无需手动框选**，自动使用全图作为参考特征源
- 参考图不包含在标注目标中，仅用于提供特征模板
- 工作流：手动修正低分图 → 加入参考图库 → 重新批量标注 → 准确率提升

---

## v5.0 类别统一 + 图片自适应

### 全模式类别强制选择

- 模式1/2/3 标注前**必须选择全局类别**，下拉框显示颜色色块 + 名称
- 自动/手动标注结果统一使用所选类别的全局绑定颜色渲染
- COCO 导出携带 `category_rgb` 字段，颜色映射与标注结果一一对应

### 图片自适应渲染

| 规则 | 说明 |
|------|------|
| **等比缩放** | 图片按宽/高中较大维度缩放适配画布（20px 内边距） |
| **居中显示** | 缩放后图片在画布中水平/垂直居中 |
| **不放大小图** | 小于画布的图片按原图尺寸居中 |
| **切图重置** | 切换图片时自动重算缩放参数，Stage 缩放/位置还原，临时图层清除 |
| **坐标映射** | `imageAdapter.ts` 提供 `canvasToImage` / `imageToCanvas` / `canvasBboxToImage` 转换 |

### 缩放控制

- 工具栏新增放大/缩小/还原按钮，手动调整视图比例
- 右下角同时显示「原图尺寸 | 适配比例 | 视图比例」

### 工具类

| 文件 | 功能 |
|------|------|
| `frontend/src/utils/categoryColorMap.ts` | 类别-颜色映射、查询、自动分配、同步、统一颜色获取 |
| `frontend/src/utils/imageAdapter.ts` | 图片等比缩放居中计算、画布↔原图坐标双向映射、批量坐标转换 |

---

## v4.0 修复与增强

### Bug 修复

| 修复项 | 说明 |
|--------|------|
| **模式2 框选残留** | `Mode2CategoryRef` 增加 `imageId` 字段，`confirmedBboxes` 仅渲染当前图片的框选；标注提交后 `clearBboxAfterAnnotate` 自动清空 |
| **类别跨模式隔离** | 全局 `CategoryPanel` 统一管理类别，模式1/2/3 共享同一类别颜色映射 |

### 全局类别管理

| 功能 | 说明 |
|------|------|
| **全局类别面板** | 底部 `CategoryPanel` 组件，添加/编辑/删除类别，绑定 RGB 颜色 |
| **localStorage 持久化** | 类别数据保存到浏览器本地存储，刷新页面不丢失 |
| **模式1 类别绑定** | 文本标注可绑定全局类别，Mask 使用该类别颜色渲染 |
| **颜色统一** | 同类别在所有模式下（自动/手动）使用统一颜色 |

---

## v3.0 核心增强

### 1. 多参考图增强标注

| 功能 | 说明 |
|------|------|
| **参考图库** | 右侧面板图钉按钮选择 2~5 张参考图 |
| **权重设置** | 核心参考图 1.0、辅助 0.8，滑块可视化调节 |
| **MRF 融合** | `MultiReferFuser` 低层/高层特征加权融合（v7 升级） |
| **完全兼容** | 不选参考图库时自动回退到单参考图模式 |

### 2. 置信度评分（v7 三维重构）

| 维度 | 权重 | 说明 |
|------|------|------|
| 特征匹配度 | 40% | DINOv3 余弦相似度 + ACT 动态阈值 |
| Mask 合理性 | 40% | 覆盖率 + 面积合理性综合 |
| 形态完整性 | 20% | 孔洞率 + 连通域数量（OpenCV） |

- 图片列表显示 0~100 分，颜色区分：≥85 绿色、70~85 黄色、<70 红色
- 支持按分数筛选（高/中/低）和排序（升序/降序）
- 悬浮分数 Tag 显示分项评分

### 3. 手动标注兜底

| 工具 | 说明 |
|------|------|
| **矩形框选** | 画矩形 → 触发 SAM3 生成精准 Mask |
| **负向框选** | 红色矩形 → 排除错误区域，修正 Mask |
| **撤销/重做** | Ctrl+Z / Ctrl+Shift+Z，支持 10 步操作历史 |
| **清空标注** | 一键清除当前图片所有 Mask |

### 4. 算法优化

| 优化项 | 说明 |
|--------|------|
| 多尺度特征融合 | 0.8x/1.0x/1.2x 三尺度提取后加权平均 |
| 背景抑制 | 裁剪到 mask bbox + 15% padding |
| 通道方差注意力 | softmax 加权强化判别性通道 |
| SAM3-PCS 范式 | 图像示例+坐标+文本提示混合推理 |
| MRF 多参考融合 | 低层/高层特征自注意力交互融合 |
| ACT 动态阈值 | KDE 自适应前景/背景分界 |
| SAM3 多 Mask 并集 | 高分候选 Mask 取 union |
| 形态学后处理 | 闭运算 + 高斯平滑 |
| 输入分辨率限制 | 长边 ≤ 1024px（显存优化） |
| 肘部法则聚类 | 自动 K=2~10 |
| 图像预处理 | CLAHE + 高斯去噪 |

## 架构设计

```
┌─────────────────┐     ┌──────────────┐     ┌──────────────┐
│   React 前端     │────▶│   FastAPI    │────▶│    Redis     │
│ Zustand + Konva  │◀────│  API 接收层  │◀────│  任务/评分   │
│ 三模式+负向提示  │     │  (≤100ms)   │     │  状态存储    │
└─────────────────┘     └──────────────┘     └──────┬───────┘
                                                    │
                                             ┌──────▼───────┐
                                             │ Celery Worker │
                                             │ ┌───────────┐ │
                                             │ │high_priority│ │ ← 实时交互
                                             │ │low_priority │ │ ← 批量标注
                                             │ └───────────┘ │
                                             │ SAM3-PCS     │
                                             │ MRF + DINOv3 │
                                             │ ACT/ACF 阈值 │
                                             │ +评分计算     │
                                             └──────────────┘
```

## 环境要求

| 项目 | 最低要求 |
|------|----------|
| Python | >= 3.12 |
| PyTorch | >= 2.7 |
| CUDA | >= 12.6 |
| GPU 显存 | >= 8GB（优化后） |
| Node.js | >= 18 |
| Redis | >= 5.0 |

## 快速开始

### 手动启动（4 个终端）

```powershell
# 终端1: Redis
redis-server

# 终端2: Celery Worker（多队列）
cd Auto-label
$env:PYTHONPATH = (Get-Location).Path
celery -A backend.worker worker --concurrency=1 --pool=solo -l info -Q high_priority,low_priority,celery

# 终端3: FastAPI
$env:PYTHONPATH = (Get-Location).Path
uvicorn backend.main:app --host 0.0.0.0 --port 8000

# 终端4: 前端
cd frontend && npm run dev
```

访问 **http://localhost:5173** 开始使用。

## 使用指南

### 完整标注流程

1. **上传图片**：拖拽或点击上传
2. **选择参考图**：星标设为主参考图，图钉加入参考图库（可选 2~5 张，滑块设权重）
3. **选择模式**：文本/框选/实例
4. **批量自动标注**：一键提交异步任务（MRF 融合 + ACT 动态阈值）
5. **查看评分**：图片列表显示置信分数（绿/黄/红，三维评分）
6. **筛选低分图**：按分数排序或筛选，定位需修正的图片
7. **手动修正**：矩形框选触发 SAM3 补充 Mask
8. **负向修正**：红色负向框选排除错误区域
9. **撤销/重做**：Ctrl+Z / Ctrl+Shift+Z
10. **导出**：COCO / VOC / YOLO 格式

### 多参考图使用

1. 在右侧图片列表点击 **图钉** 按钮将图片加入参考图库
2. 参考图库面板显示在右下角，拖动滑块设置权重
3. 切换到对应参考图，在画布上框选目标
4. 点击「批量标注」，后端自动通过 MRF 融合多张参考图特征

### 负向提示修正

1. 在手动标注区域点击 **负向框选** 按钮（红色图标）
2. 在画布上用红色框选需要排除的区域
3. 系统自动调用 SAM3 生成修正后的 Mask（排除负向区域）
4. 可叠加多个负向框精细修正

## API 接口

| 接口 | 方法 | 说明 |
|------|------|------|
| `/api/upload` | POST | 上传图像 |
| `/api/images` | GET | 获取图片列表 |
| `/api/annotate/mode1` | POST | 文本提示标注 |
| `/api/annotate/mode2` | POST | 框选标注（MRF+PCS） |
| `/api/annotate/mode3` | POST | 实例发现 |
| `/api/annotate/mode3/select` | POST | 跨图标注（MRF融合） |
| `/api/annotate/manual` | POST | 手动框选触发 SAM3 |
| `/api/annotate/correct` | POST | 负向提示修正 Mask |
| `/api/tasks/{id}` | GET | 查询状态（含评分） |
| `/api/export/{id}/{fmt}` | GET | 导出标注 |

## 技术栈

| 模块 | 技术 |
|------|------|
| 后端 | FastAPI + Celery（多队列） + Redis + PyTorch 2.7 |
| 前端 | React 18 + TypeScript + Zustand + react-konva |
| 检测 | Grounding DINO (transformers) |
| 分割 | SAM3（PCS范式 + 多Mask并集 + 形态学 + 负向提示） |
| 特征 | DINOv3 ViT-S/16（多尺度 + MRF多层融合 + ACT动态阈值） |
| 融合 | MRF（通道MLP + 自注意力 + 低层/高层特征融合） |
| 评分 | 三维置信度量化（特征匹配40% + Mask合理性40% + 形态完整性20%） |
| 显存 | 混合精度 + 分辨率限制（峰值 ≤ 5GB） |

## 项目结构

```
Auto-label/
├── backend/
│   ├── main.py                      # FastAPI 应用入口
│   ├── worker.py                    # Celery Worker（多队列）
│   ├── api/
│   │   └── routes.py                # API 路由（含 /correct 接口）
│   ├── core/
│   │   ├── config.py                # 全局配置
│   │   └── exceptions.py            # 异常处理
│   ├── models/
│   │   ├── sam_engine.py            # SAM3 引擎（PCS + 负向提示）
│   │   ├── dino_engine.py           # DINOv3 引擎（多层特征 + ACT）
│   │   ├── grounding_dino_engine.py # Grounding DINO 引擎
│   │   └── mrf_engine.py            # MRF 多参考融合器
│   ├── services/
│   │   ├── storage.py               # 文件存储
│   │   ├── mask_utils.py            # Mask 工具
│   │   └── image_utils.py           # 图像分辨率预处理
│   ├── utils/
│   │   ├── score_calculator.py      # 三维置信评分
│   │   └── threshold_utils.py       # ACT/ACF 动态阈值
│   └── libs/                        # SAM3/DINOv3 内置库
├── frontend/
│   ├── src/
│   │   ├── api/index.ts             # API 接口（含 correct）
│   │   ├── components/
│   │   │   ├── Toolbar.tsx          # 工具栏（含负向框选+文本提示）
│   │   │   ├── MainCanvas.tsx       # 画布（负向框+图层强绑定）
│   │   │   ├── CategoryPanel.tsx    # 类别管理
│   │   │   └── RightPanel.tsx       # 右侧面板
│   │   ├── store/useAppStore.ts     # Zustand 状态
│   │   └── utils/
│   │       ├── imageAdapter.ts      # 坐标映射增强
│   │       └── categoryColorMap.ts  # 颜色统一管理
│   └── package.json
└── README.md
```

## License

MIT License
