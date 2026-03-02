# Auto-label：人机协同图像自动标注工具

基于 **Grounding DINO + SAM3 + DINOv3** 三大核心算法的高并发人机协同标注系统。
采用 **FastAPI + Celery + Redis** 异步架构，GPU 推理不阻塞 API 请求。
前端基于 **React + TypeScript + Zustand + react-konva** 实现三层 Canvas 交互。

## 三大标注模式

| 模式 | 名称 | 链路 | 适用场景 |
|------|------|------|----------|
| 模式1 | 文本提示标注 | 文本 → Grounding DINO 检测 bbox → SAM3 精准分割 | 已知类别名称，一键标注全部图片 |
| 模式2 | 框选批量标注 | 框选 bbox → SAM3 生成 Mask → DINOv3 特征匹配 → 批量 SAM3 | 未知类别，标一个扩展到全部 |
| 模式3 | 选实例跨图标注 | DINOv3 聚类粗分割 → 用户选实例 → DINOv3 特征匹配 → SAM3 批量分割 | 视觉选择，跨图批量扩展 |

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
│   └── (Grounding DINO 首次启动自动下载到 HuggingFace 缓存)
├── .npm-cache/                        # npm 本地缓存
│
└── Auto-label/                        # 项目代码目录
    ├── backend/
    │   ├── main.py                    # FastAPI 入口
    │   ├── worker.py                  # Celery Worker：三模式异步任务 + 三模型预热
    │   ├── core/
    │   │   ├── config.py              # 全局配置（Redis/模型路径/阈值）
    │   │   └── exceptions.py          # 全局异常处理
    │   ├── api/
    │   │   └── routes.py              # API 路由
    │   ├── models/
    │   │   ├── sam_engine.py          # SAM3 单例：bbox → Mask
    │   │   ├── dino_engine.py         # DINOv3 单例：特征提取 + 余弦匹配
    │   │   └── grounding_dino_engine.py  # Grounding DINO 单例：文本 → bbox 检测
    │   ├── services/
    │   │   ├── storage.py             # 文件存储
    │   │   └── mask_utils.py          # Mask → 透明 PNG / COCO / VOC / YOLO 格式转换
    │   ├── libs/                      # 内置算法库
    │   │   ├── dinov3/
    │   │   └── sam3/
    │   └── requirements.txt
    ├── frontend/
    │   ├── src/
    │   │   ├── App.tsx
    │   │   ├── store/useAppStore.ts
    │   │   ├── api/index.ts
    │   │   └── components/
    │   │       ├── Toolbar.tsx
    │   │       ├── MainCanvas.tsx
    │   │       └── RightPanel.tsx
    │   └── package.json
    ├── data/                          # 运行时数据（自动创建）
    ├── scripts/
    │   ├── start.ps1                  # Windows 一键启动脚本
    │   ├── install.ps1                # Windows 安装脚本
    │   ├── install.sh                 # Linux/macOS 安装脚本
    │   └── test_pipeline.py           # 验证测试脚本
    ├── .env.example                   # 环境配置示例
    ├── Dockerfile
    ├── docker-compose.yml
    └── README.md
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

# 安装前端依赖（使用本地 npm 缓存）
npm config set cache "E:\auto-label\.npm-cache"
cd frontend; npm install; cd ..
```

### 2. 配置权重文件

将以下权重文件放入 `E:\auto-label\weights\` 目录：

| 文件名 | 模型 | 获取方式 |
|--------|------|----------|
| `sam3.pt` | SAM3 | 从 [HuggingFace](https://huggingface.co/facebook/sam3) 下载 |
| `dinov3_vits16_pretrain_lvd1689m-08c60483.pth` | DINOv3 | 从 [Meta AI](https://ai.meta.com/resources/models-and-libraries/dinov3-downloads/) 下载 |
| Grounding DINO | Grounding DINO | **无需手动下载**，首次启动自动从 HuggingFace Hub 下载 |

### 3. 一键启动所有服务

```powershell
cd Auto-label
.\scripts\start.ps1
```

该脚本会自动启动 4 个服务：
1. Redis（`E:\app\Redis\Redis-x64-5.0.14.1\redis-server.exe`）
2. Celery Worker（GPU 推理，并发=1）
3. FastAPI（后端 API，端口 8000）
4. Vite（前端开发服务器，端口 5173）

### 4. 手动启动（分 4 个终端）

```powershell
# 终端1: Redis
E:\app\Redis\Redis-x64-5.0.14.1\redis-server.exe

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

## 环境配置

复制 `.env.example` 为 `.env`，根据实际环境修改：

```bash
cp .env.example .env
```

主要配置项：

| 环境变量 | 默认值 | 说明 |
|----------|--------|------|
| `WEIGHTS_DIR` | `../weights` | 权重文件目录 |
| `SAM3_CHECKPOINT` | `weights/sam3.pt` | SAM3 权重路径 |
| `DINO_WEIGHTS_PATH` | `weights/dinov3_...pth` | DINOv3 权重路径 |
| `GROUNDING_DINO_MODEL_NAME` | `IDEA-Research/grounding-dino-base` | Grounding DINO 模型 |
| `REDIS_HOST` | `127.0.0.1` | Redis 地址 |
| `REDIS_PORT` | `6379` | Redis 端口 |
| `GROUNDING_DINO_SCORE_THR` | `0.3` | 检测置信度阈值 |
| `GROUNDING_DINO_SCORE_THR_LOW` | `0.2` | 低置信度重试阈值 |

## 验证测试

```bash
python scripts/test_pipeline.py --image path/to/test.jpg
```

## API 接口

| 接口 | 方法 | 说明 |
|------|------|------|
| `/api/upload` | POST | 上传图像 |
| `/api/images` | GET | 获取图片列表 |
| `/api/images/{id}` | DELETE | 删除图片 |
| `/api/annotate/mode1` | POST | 模式1：文本提示标注 |
| `/api/annotate/mode2` | POST | 模式2：框选批量标注 |
| `/api/annotate/mode3` | POST | 模式3阶段1：实例发现 |
| `/api/annotate/mode3/select` | POST | 模式3阶段2：跨图标注 |
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
| GPU 显存（三模型共存，float16） | ≤ 10GB |

## 技术栈

| 模块 | 技术 |
|------|------|
| 后端 | FastAPI + Celery + Redis + PyTorch 2.7 |
| 前端 | React 18 + TypeScript + Zustand + react-konva |
| 检测 | Grounding DINO Base (HuggingFace transformers) |
| 分割 | SAM3 (Meta, 精准实例分割) |
| 特征 | DINOv3 ViT-S/16 (Meta, 自监督特征匹配) |
| 部署 | Docker + NVIDIA CUDA 12.6 + Redis |

## License

MIT License
