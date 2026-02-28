# Auto-label：人机协同图像自动标注工具

基于 **SAM3 + DINOv3 + YOLO-World** 三大核心算法的高并发人机协同标注系统。
采用 **FastAPI + Celery + Redis** 异步架构，GPU 推理不阻塞 API 请求。
前端基于 **React + TypeScript + Zustand + react-konva** 实现三层 Canvas 交互。

## 双核心标注模式

| 模式 | 名称 | 链路 | 适用场景 |
|------|------|------|----------|
| 模式1 | 文本提示标注 | 文本 → YOLO-World 检测 bbox → SAM3 精准分割 | 已知类别名称，一键标注全部图片 |
| 模式2 | 框选批量标注 | 框选 bbox → SAM3 生成 Mask → DINOv3 特征匹配 → 批量 SAM3 | 未知类别，标一个扩展到全部 |

## 架构设计

```
┌─────────────────┐     ┌──────────────┐     ┌──────────────┐
│   React 前端     │────▶│   FastAPI    │────▶│    Redis     │
│ Zustand + Konva  │◀────│  API 接收层  │◀────│  任务队列    │
│ 双模式交互       │     │  (≤100ms)   │     │  状态存储    │
└─────────────────┘     └──────────────┘     └──────┬───────┘
                                                    │
                                             ┌──────▼───────┐
                                             │ Celery Worker │
                                             │  异步计算层   │
                                             │ YOLO-World    │
                                             │ + SAM3 + DINO │
                                             │ (GPU half)    │
                                             └──────────────┘
```

## 目录结构

```
Auto-label/
├── backend/
│   ├── main.py                 # FastAPI 入口
│   ├── worker.py               # Celery Worker：双模式异步任务 + 三模型预热
│   ├── core/
│   │   ├── config.py           # Redis/模型路径/阈值配置（含 YOLO-World）
│   │   └── exceptions.py       # 全局异常（含 YOLO 检测失败、文本为空）
│   ├── api/
│   │   └── routes.py           # /upload, /annotate/mode1, /annotate/mode2, /tasks/{id}
│   ├── models/
│   │   ├── sam_engine.py       # SAM3 单例：bbox → Mask，半精度
│   │   ├── dino_engine.py      # DINOv3 单例：特征提取 + 余弦匹配
│   │   └── yolo_engine.py      # YOLO-World 单例：文本 → bbox 检测
│   ├── services/
│   │   ├── storage.py          # 文件存储：路径生成、URL 拼接
│   │   └── mask_utils.py       # Mask → 透明 PNG 转换
│   ├── libs/                   # 内置算法库
│   │   ├── dinov3/
│   │   ├── sam3/
│   │   └── yolo_world/
│   └── requirements.txt
├── frontend/
│   ├── src/
│   │   ├── App.tsx             # 主布局：Toolbar | Canvas | Panel
│   │   ├── store/useAppStore.ts # Zustand（双模式 + 文本提示 + 任务状态）
│   │   ├── api/index.ts        # 后端 API 封装（mode1 + mode2）
│   │   └── components/
│   │       ├── Toolbar.tsx     # 模式切换 + 文本输入 + 框选工具 + 标注触发
│   │       ├── MainCanvas.tsx  # 三层 konva 画布（双模式兼容）
│   │       └── RightPanel.tsx  # 图片列表 + 进度 + 导出 + 删除
│   └── package.json
├── data/                       # 运行时数据（自动创建）
│   ├── images/
│   ├── masks/
│   └── exports/
├── Dockerfile
├── docker-compose.yml
└── README.md
```

## 模式1 全链路数据流转

```
用户输入文本提示 "person, car"
        │
        ▼
POST /api/annotate/mode1
{text_prompt, image_ids, image_paths}
        │ (≤100ms)
        ▼
FastAPI → Redis: status=pending → Celery delay()
返回 {task_id}
        │
        ▼
Celery Worker 执行：
  解析文本 → ["person", "car"]
  for 每张图:
    ① YOLO-World("person","car", image) → detections [{box, label, score}]
    ② for 每个 detection:
       SAM3(image, box) → precise_mask
       mask → 透明 PNG (蓝色 alpha=140)
    ③ 更新 Redis: progress++
  生成 COCO JSON
  Redis: status=success, mask_urls={...}
        │
        ▼
前端 1s 轮询 → 进度条 → Mask PNG 叠加渲染（蓝色）
```

## 模式2 全链路数据流转

```
用户框选 bbox → Zustand 存储坐标
        │
        ▼
POST /api/annotate/mode2
{ref_image_id, bbox, target_images}
        │ (≤100ms)
        ▼
Celery Worker 执行：
  ① SAM3(首图, bbox) → ref_mask
  ② DINOv3(首图, ref_mask) → template_feature [384]
  ③ for 每张目标图:
     DINOv3 余弦匹配 → matched_bboxes
     SAM3(目标图, bbox) → precise_mask
     mask → 透明 PNG (红色 alpha=128)
     更新 Redis: progress++
  生成 COCO JSON
  Redis: status=success, mask_urls={...}
        │
        ▼
前端 1s 轮询 → 进度条 → Mask PNG 叠加渲染（红色）
```

## 部署说明

### 环境要求
- Python >= 3.11, CUDA >= 12.1, GPU >= 12GB (RTX 3060/3090)
- Node.js >= 18
- Redis >= 7.0

### 安装步骤

```bash
cd Auto-label

# 1. 安装 PyTorch
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# 2. 安装 MMCV 生态（YOLO-World 依赖，必须按顺序）
pip install openmim
mim install mmengine
mim install mmcv==2.0.0
mim install mmdet>=3.0.0

# 3. 安装后端依赖
pip install -r backend/requirements.txt

# 4. 安装内置算法库
pip install -e backend/libs/dinov3
pip install -e backend/libs/sam3
pip install -e backend/libs/yolo_world

# 5. 下载 YOLO-World 预训练权重
# 从 https://github.com/AILab-CVC/YOLO-World/releases 下载 yolo_world_v2_s 权重
# 设置环境变量：export YOLOWORLD_WEIGHTS=/path/to/yolo_world_v2_s.pth

# 6. 安装前端依赖
cd frontend && npm install && cd ..
```

### 启动服务（4 个终端）

```bash
# 终端1: Redis
redis-server

# 终端2: Celery Worker（并发数=1，避免 GPU OOM）
celery -A backend.worker worker --concurrency=1 --pool=solo -l info

# 终端3: FastAPI
uvicorn backend.main:app --host 0.0.0.0 --port 8000

# 终端4: 前端
cd frontend && npm run dev
# 访问 http://localhost:5173
```

## API 接口

| 接口 | 方法 | 说明 |
|------|------|------|
| `/api/upload` | POST | 上传图像 |
| `/api/images` | GET | 获取图片列表 |
| `/api/images/{id}` | DELETE | 删除图片 |
| `/api/annotate/mode1` | POST | 模式1：文本提示标注 |
| `/api/annotate/mode2` | POST | 模式2：框选批量标注 |
| `/api/tasks/{task_id}` | GET | 查询任务状态/进度/Mask URL |
| `/api/health` | GET | 健康检查 |

## 性能指标

| 指标 | 目标 |
|------|------|
| API 响应时间 | ≤ 100ms |
| 模式1 单图标注 | ≤ 2s |
| 模式2 单图匹配+Mask | ≤ 3s |
| Mask PNG 大小 | ≤ 100KB |
| GPU 显存 | ≤ 12GB |

## 视觉区分

- 模式1 Mask：**蓝色** (0, 120, 255) alpha=140
- 模式2 Mask：**红色** (255, 0, 0) alpha=128
- 画布左上角显示当前模式标签

## 技术栈

| 模块 | 技术 |
|------|------|
| 后端 | FastAPI + Celery + Redis + PyTorch |
| 前端 | React 18 + TypeScript + Zustand + react-konva |
| 检测 | YOLO-World-v2-S (文本驱动开放词汇检测) |
| 分割 | SAM3 (精准实例分割) |
| 特征 | DINOv3 ViT-S/16 (自监督特征匹配) |
| 部署 | Docker + NVIDIA CUDA 12.1 + Redis 7 |
