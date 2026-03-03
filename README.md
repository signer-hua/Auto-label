# Auto-label：人机协同图像自动标注工具

基于 **Grounding DINO + SAM3 + DINOv3** 三大核心算法的工业级人机协同标注系统。
采用 **FastAPI + Celery + Redis** 异步架构，GPU 推理不阻塞 API 请求。
前端基于 **React + TypeScript + Zustand + react-konva** 实现三层 Canvas 交互。

## 三大标注模式

| 模式 | 名称 | 链路 | 适用场景 |
|------|------|------|----------|
| 模式1 | 文本提示标注 | 文本 → Grounding DINO 检测 bbox → SAM3 精准分割 | 已知类别名称，一键标注全部图片 |
| 模式2 | 框选批量标注 | 框选 bbox → SAM3 生成 Mask → DINOv3 特征匹配 → 批量 SAM3 | 未知类别，标一个扩展到全部 |
| 模式3 | 选实例跨图标注 | DINOv3 聚类粗分割 → 用户选实例 → DINOv3 特征匹配 → SAM3 批量分割 | 视觉选择，跨图批量扩展 |

## v2.0 增强特性

### 算法优化（标注精度 70% → ≥85%）

| 优化项 | 说明 |
|--------|------|
| **DINOv3 多尺度特征融合** | 0.8x/1.0x/1.2x 三尺度提取后加权平均，提升跨分辨率匹配鲁棒性 |
| **背景抑制** | 裁剪目标区域 + 15% padding，排除背景噪声对特征的干扰 |
| **通道方差注意力加权** | 高方差通道包含更多判别性信息，softmax 加权强化核心特征 |
| **SAM3 多 Mask 并集融合** | 取多个高分候选 Mask 并集，提升目标完整性 |
| **形态学后处理** | 闭运算填充空洞 + 高斯平滑边缘，减少锯齿 |
| **动态余弦阈值** | 大目标阈值 0.72、小目标阈值 0.65，按面积比自适应插值 |
| **肘部法则聚类** | 自动选择 K=2~10 最优聚类数，替代固定 K=8 |
| **图像预处理** | 高斯去噪 + CLAHE 对比度增强，适配低质量/曝光异常图片 |

### 功能增强

| 功能 | 说明 |
|------|------|
| **多类别并行标注** | 模式2/3 支持定义多个类别，每个类别独立参考、独立匹配、独立 Mask |
| **多实例合并标注** | 模式3 支持 Ctrl/Shift 多选粗分割实例，融合特征后统一匹配 |
| **多框选参考融合** | 模式2 每个类别支持多个框选参考，取特征均值降低人工误差 |
| **COCO 多类别映射** | 导出时保留类别 ID + 名称，完全兼容 COCO 训练流程 |

### 交互修复

| 修复项 | 说明 |
|--------|------|
| **模式2 框选隔离** | 框选矩形与 imageId 强绑定，切图不残留、不跨图污染 |
| **模式3 实例隔离** | 实例图层与 imageId 强绑定，切图自动清除、不重叠 |
| **模式切换清理** | 切换模式时自动清空非当前模式的所有画布元素 |

## 架构设计

```
┌─────────────────┐     ┌──────────────┐     ┌──────────────┐
│   React 前端     │────▶│   FastAPI    │────▶│    Redis     │
│ Zustand + Konva  │◀────│  API 接收层  │◀────│  任务队列    │
│ 三模式交互       │     │  (≤100ms)   │     │  状态存储    │
└─────────────────┘     └──────────────┘     └──────┬───────┘
                                                    │
                                             ┌──────▼───────┐
                                             │ Celery Worker │
                                             │  异步计算层   │
                                             │ Grounding DINO│
                                             │ + SAM3 + DINO │
                                             │ (GPU float16) │
                                             └──────────────┘
```

## 目录结构

```
auto-label/                            # 工作区根目录
├── weights/                           # 模型权重文件（手动放置）
│   ├── sam3.pt                        # SAM3 权重
│   ├── dinov3_vits16_pretrain_lvd1689m-08c60483.pth  # DINOv3 权重
│   └── grounding-dino-base/           # Grounding DINO 本地权重（可选）
│
└── Auto-label/                        # 项目代码目录
    ├── backend/
    │   ├── main.py                    # FastAPI 入口
    │   ├── worker.py                  # Celery Worker：三模式异步任务 + 图像预处理
    │   ├── core/
    │   │   ├── config.py              # 全局配置（算法参数/阈值/预处理开关）
    │   │   └── exceptions.py          # 全局异常处理
    │   ├── api/
    │   │   └── routes.py              # API 路由（含多类别请求模型）
    │   ├── models/
    │   │   ├── sam_engine.py          # SAM3 单例：多 Mask 并集 + 形态学后处理
    │   │   ├── dino_engine.py         # DINOv3 单例：多尺度 + 背景抑制 + 注意力
    │   │   └── grounding_dino_engine.py  # Grounding DINO 单例
    │   ├── services/
    │   │   ├── storage.py             # 文件存储
    │   │   └── mask_utils.py          # Mask 格式转换
    │   └── libs/                      # 内置算法库
    │       ├── dinov3/
    │       └── sam3/
    ├── frontend/
    │   └── src/
    │       ├── App.tsx
    │       ├── store/useAppStore.ts    # Zustand 状态（图层隔离 + 多类别）
    │       ├── api/index.ts           # API 客户端（多类别接口）
    │       └── components/
    │           ├── Toolbar.tsx         # 工具栏（多类别 UI + 多选）
    │           ├── MainCanvas.tsx      # 画布（图层隔离 + Ctrl 多选）
    │           └── RightPanel.tsx      # 右侧面板
    └── data/                          # 运行时数据（自动创建）
```

## 环境要求

| 项目 | 最低要求 |
|------|----------|
| Python | >= 3.12 |
| PyTorch | >= 2.7 |
| CUDA | >= 12.6 |
| GPU 显存 | >= 12GB (RTX 3060/3090) |
| Node.js | >= 18 |
| Redis | >= 5.0 |

## 快速开始（Windows）

### 1. 安装依赖

```powershell
cd Auto-label
.\scripts\install.ps1
```

或手动安装：

```powershell
# 安装 PyTorch (CUDA 12.6)
pip install torch==2.7.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126

# 安装后端依赖
pip install -r backend\requirements.txt

# 安装内置算法库
pip install -e backend\libs\dinov3
pip install -e backend\libs\sam3

# 安装前端依赖
cd frontend; npm install; cd ..
```

### 2. 配置权重文件

将以下权重文件放入 `weights/` 目录：

| 文件名 | 模型 | 获取方式 |
|--------|------|----------|
| `sam3.pt` | SAM3 | 从 HuggingFace 下载 |
| `dinov3_vits16_pretrain_lvd1689m-08c60483.pth` | DINOv3 | 从 Meta AI 下载 |
| `grounding-dino-base/` | Grounding DINO | 可选，手动下载放入；否则首次启动自动在线加载 |

### 3. 一键启动所有服务

```powershell
cd Auto-label
.\scripts\start.ps1
```

### 4. 手动启动（分 4 个终端）

```powershell
# 终端1: Redis
redis-server

# 终端2: Celery Worker（在 Auto-label 目录下执行）
$env:PYTHONPATH = (Get-Location).Path
celery -A backend.worker worker --concurrency=1 --pool=solo -l info

# 终端3: FastAPI
$env:PYTHONPATH = (Get-Location).Path
uvicorn backend.main:app --host 0.0.0.0 --port 8000

# 终端4: 前端
cd frontend; npm run dev
```

访问 **http://localhost:5173** 开始使用。

## 快速开始（Linux/macOS）

```bash
cd Auto-label
bash scripts/install.sh

# 启动（4 个终端）
redis-server
celery -A backend.worker worker --concurrency=1 --pool=solo -l info
uvicorn backend.main:app --host 0.0.0.0 --port 8000
cd frontend && npm run dev
```

## 使用指南

### 模式1：文本提示标注

1. 上传图片（支持拖拽批量上传）
2. 输入文本提示（如 `person, car, dog`，逗号分隔多个类别）
3. 点击「一键标注」
4. 等待标注完成，切换图片查看 Mask 结果
5. 导出 COCO/VOC/YOLO 格式

### 模式2：框选批量标注

**单类别模式：**
1. 上传图片，选择参考图（星标图标）
2. 使用框选工具在参考图上框选目标
3. 点击「批量标注」

**多类别模式：**
1. 在「多类别管理」区域添加类别（如 cat、dog）
2. 选择活跃类别 → 框选对应目标 → 点击「确认框选」
3. 切换类别 → 框选另一目标 → 确认
4. 每个类别可添加多个框选参考（取均值降低误差）
5. 点击「批量标注」

### 模式3：选实例跨图标注

**单实例模式：**
1. 选择参考图 → 点击「生成实例」
2. 在画布上点击选中目标实例（支持 Ctrl+点击多选）
3. 点击「跨图标注」

**多类别多实例模式：**
1. 添加类别 → 选择活跃类别
2. 多选实例 → 点击「分配到类别」
3. 切换类别 → 多选其他实例 → 分配
4. 点击「跨图标注」

## 环境配置

主要配置项（环境变量或 `.env` 文件）：

| 环境变量 | 默认值 | 说明 |
|----------|--------|------|
| `WEIGHTS_DIR` | `../weights` | 权重文件目录 |
| `SAM3_CHECKPOINT` | `weights/sam3.pt` | SAM3 权重路径 |
| `DINO_WEIGHTS_PATH` | `weights/dinov3_...pth` | DINOv3 权重路径 |
| `GROUNDING_DINO_MODEL_NAME` | `IDEA-Research/grounding-dino-base` | Grounding DINO 模型 |
| `REDIS_HOST` / `REDIS_PORT` | `127.0.0.1` / `6379` | Redis 连接 |
| `COSINE_THRESHOLD` | `0.75` | 余弦相似度基准阈值 |
| `COSINE_SIM_LARGE_THRESH` | `0.72` | 大目标动态阈值 |
| `COSINE_SIM_SMALL_THRESH` | `0.65` | 小目标动态阈值 |
| `PREPROCESS_CLAHE` | `true` | CLAHE 对比度增强开关 |
| `PREPROCESS_DENOISE` | `true` | 高斯去噪开关 |
| `SAM3_MULTIMASK_UNION` | `true` | 多 Mask 并集融合开关 |
| `DINO_ELBOW_MAX_K` | `10` | 肘部法则最大聚类数 |

## API 接口

| 接口 | 方法 | 说明 |
|------|------|------|
| `/api/upload` | POST | 上传图像 |
| `/api/images` | GET | 获取图片列表 |
| `/api/images/{id}` | DELETE | 删除图片 |
| `/api/annotate/mode1` | POST | 模式1：文本提示标注 |
| `/api/annotate/mode2` | POST | 模式2：框选批量标注（支持多类别） |
| `/api/annotate/mode3` | POST | 模式3阶段1：实例发现 |
| `/api/annotate/mode3/select` | POST | 模式3阶段2：跨图标注（支持多类别+多实例） |
| `/api/tasks/{task_id}` | GET | 查询任务状态 |
| `/api/tasks/{task_id}/pause` | POST | 暂停任务 |
| `/api/tasks/{task_id}/resume` | POST | 恢复任务 |
| `/api/tasks/{task_id}/cancel` | POST | 取消任务 |
| `/api/export/{task_id}/{fmt}` | GET | 导出标注（coco/voc/yolo 格式） |

## 性能指标

| 指标 | 目标 |
|------|------|
| API 响应时间 | ≤ 100ms |
| 模式1 单图标注 | ≤ 2s |
| 模式2 单图匹配+Mask | ≤ 3s |
| GPU 显存（三模型共存） | ≤ 10GB |
| 标注精度（Mask mIoU） | ≥ 80% |

## 技术栈

| 模块 | 技术 |
|------|------|
| 后端 | FastAPI + Celery + Redis + PyTorch 2.7 |
| 前端 | React 18 + TypeScript + Zustand + react-konva |
| 检测 | Grounding DINO Base (HuggingFace transformers) |
| 分割 | SAM3 (Meta, 多 Mask 并集 + 形态学后处理) |
| 特征 | DINOv3 ViT-S/16 (多尺度 + 背景抑制 + 通道注意力) |
| 部署 | Docker + NVIDIA CUDA 12.6 + Redis |

## License

MIT License
