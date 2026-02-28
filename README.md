# Auto-label：人机协同图像自动标注工具

基于 **SAM3 + DINOv3** 的高并发、低阻塞人机协同标注系统。
采用 **FastAPI + Celery + Redis** 异步架构，GPU 推理不阻塞 API 请求。
前端基于 **React + TypeScript + react-konva** 实现三层 Canvas 交互。

## 架构设计

```
┌─────────────┐     ┌──────────────┐     ┌──────────────┐
│   React     │────▶│   FastAPI    │────▶│    Redis     │
│  前端 UI    │◀────│  API 接收层  │◀────│  任务队列    │
│ (react-konva)│     │  (≤100ms)   │     │  状态存储    │
└─────────────┘     └──────────────┘     └──────┬───────┘
                                                │
                                         ┌──────▼───────┐
                                         │ Celery Worker │
                                         │  异步计算层   │
                                         │ SAM3 + DINOv3 │
                                         │ (GPU half)    │
                                         └──────────────┘
```

### 三层架构
1. **API 接收层** (FastAPI)：接收请求，触发异步任务，立即返回 task_id（≤100ms）
2. **异步计算层** (Celery Worker)：GPU 推理，Mask 生成，实时更新 Redis 进度
3. **结果后处理层**：Mask → 透明 PNG 存储，COCO 格式导出

### 前端三层 Canvas
- **Layer0 原图层**：显示当前图像
- **Layer1 交互层**：鼠标框选矩形
- **Layer2 结果层**：Mask 透明 PNG 叠加渲染

## 目录结构

```
Auto-label/
├── backend/
│   ├── main.py                 # FastAPI 入口：/upload、/annotate/mode2、/tasks/{task_id}
│   ├── worker.py               # Celery Worker：异步任务 + 模型预热
│   ├── core/
│   │   ├── config.py           # Redis/模型路径/阈值/存储路径配置
│   │   └── exceptions.py       # 全局异常处理
│   ├── api/
│   │   └── routes.py           # 业务接口：上传、标注、查询
│   ├── models/
│   │   ├── sam_engine.py       # SAM3 单例：bbox → Mask，半精度推理
│   │   └── dino_engine.py      # DINOv3 单例：特征提取 + 余弦匹配
│   ├── services/
│   │   ├── storage.py          # 文件存储：路径生成、URL 拼接
│   │   └── mask_utils.py       # Mask → 透明 PNG 转换
│   ├── libs/                   # 内置算法库（推理最小集）
│   │   ├── dinov3/
│   │   └── sam3/
│   └── requirements.txt
├── frontend/
│   ├── src/
│   │   ├── App.tsx             # 主布局：Toolbar | Canvas | Panel
│   │   ├── store/
│   │   │   └── useAppStore.ts  # Zustand 全局状态管理
│   │   ├── api/
│   │   │   └── index.ts        # 后端 API 封装
│   │   └── components/
│   │       ├── Toolbar.tsx     # 左侧工具栏
│   │       ├── MainCanvas.tsx  # 三层 konva 画布
│   │       └── RightPanel.tsx  # 右侧面板（图片列表 + 进度 + 导出）
│   └── package.json
├── data/                       # 运行时数据（自动创建）
│   ├── images/                 # 上传的原始图像
│   ├── masks/                  # 生成的透明 PNG Mask
│   └── exports/                # COCO 格式导出
├── scripts/
│   ├── install.sh
│   └── test_pipeline.py
└── docker-compose.yml
```

## 模式2 全链路数据流转

```
用户框选 bbox ──▶ 前端 Zustand 存储坐标
                      │
                      ▼
              POST /api/annotate/mode2
              {ref_image_id, bbox, target_images}
                      │
                      ▼ (≤100ms)
              FastAPI 生成 task_id
              写入 Redis: status=pending
              调用 Celery delay()
              返回 {task_id}
                      │
                      ▼
              Celery Worker 执行：
              ① SAM3(首图, bbox) → ref_mask
              ② DINOv3(首图, ref_mask) → template_feature
              ③ for 每张目标图:
                 DINOv3 余弦匹配 → matched_bboxes
                 SAM3(目标图, bbox) → precise_mask
                 mask → 透明 PNG 保存
                 更新 Redis: progress++
              ④ 生成 COCO JSON
              ⑤ Redis: status=success, mask_urls={...}
                      │
                      ▼
              前端每 1s 轮询 GET /api/tasks/{task_id}
              更新进度条: "批量标注中 (50/100)"
              status=success → 获取 mask_urls
                      │
                      ▼
              切换右侧图片 → Layer2 加载对应 Mask PNG
              与原图精准叠加（同步缩放/平移）
```

## 部署说明

### 环境要求
- Python >= 3.11, CUDA >= 12.1, GPU >= 12GB
- Node.js >= 18
- Redis >= 7.0

### 分步启动

```bash
# 1. 启动 Redis
redis-server

# 2. 安装后端依赖
cd Auto-label
pip install -r backend/requirements.txt
pip install -e backend/libs/dinov3
pip install -e backend/libs/sam3

# 3. 启动 Celery Worker（并发数=1，避免 GPU OOM）
celery -A backend.worker worker --concurrency=1 --pool=solo -l info

# 4. 启动 FastAPI（另一个终端）
uvicorn backend.main:app --host 0.0.0.0 --port 8000

# 5. 启动前端（另一个终端）
cd frontend
npm install
npm run dev
# 访问 http://localhost:5173
```

### Docker 部署

```bash
docker-compose up --build
```

## 性能指标

| 指标 | 目标 | 说明 |
|------|------|------|
| API 响应时间 | ≤ 100ms | 仅触发异步任务，不执行推理 |
| 单张 Mask 生成 | ≤ 2s | SAM3 + DINOv3 推理 |
| Mask PNG 大小 | ≤ 100KB | 1008×1008 透明 PNG |
| 进度更新频率 | 1s | 前端轮询间隔 |
| GPU 显存 | ≤ 12GB | 适配 RTX 3060/3090 |

## 技术栈

| 模块 | 技术 |
|------|------|
| 后端 | FastAPI + Celery + Redis + PyTorch |
| 前端 | React 18 + TypeScript + Zustand + react-konva |
| 算法 | SAM3 (Meta) + DINOv3 ViT-S/16 (Meta) |
| 部署 | Docker + NVIDIA CUDA 12.1 |
