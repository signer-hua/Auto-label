# Auto-label：人机协同图像自动标注工具

基于 **Grounding DINO + SAM3 + DINOv3** 三大核心算法的高并发人机协同标注系统。
采用 **FastAPI + Celery + Redis** 异步架构，GPU 推理不阻塞 API 请求。
前端基于 **React + TypeScript + Zustand + react-konva** 实现三层 Canvas 交互。

> **v2.0 技术栈升级**：将原 YOLO-World（依赖 MMDetection/MMEngine）替换为 **Grounding DINO**（基于 HuggingFace transformers），彻底解决环境冲突问题，三模型在 PyTorch/HuggingFace 生态下 100% 兼容。

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
Auto-label/
├── backend/
│   ├── main.py                        # FastAPI 入口
│   ├── worker.py                      # Celery Worker：三模式异步任务 + 三模型预热
│   ├── core/
│   │   ├── config.py                  # Redis/模型路径/阈值配置
│   │   └── exceptions.py             # 全局异常处理
│   ├── api/
│   │   └── routes.py                  # API 路由
│   ├── models/
│   │   ├── sam_engine.py              # SAM3 单例：bbox → Mask，半精度
│   │   ├── dino_engine.py             # DINOv3 单例：特征提取 + 余弦匹配
│   │   └── grounding_dino_engine.py   # Grounding DINO 单例：文本 → bbox 检测
│   ├── services/
│   │   ├── storage.py                 # 文件存储：路径生成、URL 拼接
│   │   └── mask_utils.py             # Mask → 透明 PNG 转换
│   ├── libs/                          # 内置算法库
│   │   ├── dinov3/
│   │   └── sam3/
│   └── requirements.txt
├── frontend/
│   ├── src/
│   │   ├── App.tsx                    # 主布局
│   │   ├── store/useAppStore.ts       # Zustand 状态管理
│   │   ├── api/index.ts              # 后端 API 封装
│   │   └── components/
│   │       ├── Toolbar.tsx            # 模式切换 + 文本输入
│   │       ├── MainCanvas.tsx         # 三层 konva 画布
│   │       └── RightPanel.tsx         # 图片列表 + 进度 + 导出
│   └── package.json
├── data/                              # 运行时数据（自动创建）
│   ├── images/
│   ├── masks/
│   └── exports/
├── scripts/
│   ├── install.sh                     # Linux/macOS 安装脚本
│   ├── install.ps1                    # Windows 安装脚本
│   └── test_pipeline.py              # 验证测试脚本
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
  Grounding DINO 文本格式化 → "person . car ."
  for 每张图:
    ① Grounding DINO(image, "person . car .") → detections [{box, label, score}]
    ② 无结果时自动降低阈值（0.3→0.2）重试
    ③ for 每个 detection:
       SAM3(image, box) → precise_mask
       mask → 透明 PNG (多类别颜色区分)
    ④ 更新 Redis: progress++
  生成 COCO JSON
  Redis: status=success, mask_urls={...}
        │
        ▼
前端 1s 轮询 → 进度条 → Mask PNG 叠加渲染
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

## 模式3 全链路数据流转

```
阶段1 (实例发现):
  POST /api/annotate/mode3/discover
  → DINOv3 全图 patch 特征 → K-Means 聚类
  → SAM3 对每个聚类区域生成粗分割 Mask
  → 返回实例列表供用户选择

阶段2 (跨图标注):
  POST /api/annotate/mode3/annotate
  → DINOv3 提取选中实例特征模板
  → for 每张目标图:
     DINOv3 余弦匹配 → SAM3 精准 Mask
  → 生成 COCO JSON
```

## 环境要求

| 项目 | 最低要求 |
|------|----------|
| Python | >= 3.12 |
| PyTorch | >= 2.7 |
| CUDA | >= 12.6 |
| GPU 显存 | >= 12GB (RTX 3060/3090) |
| Node.js | >= 18 |
| Redis | >= 7.0 |

## 安装步骤

### 方式一：一键安装

```bash
cd Auto-label

# Linux/macOS
bash scripts/install.sh

# Windows (PowerShell)
.\scripts\install.ps1
```

### 方式二：手动安装

```bash
cd Auto-label

# 1. 安装 PyTorch (CUDA 12.6)
pip install torch==2.7.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126

# 2. 安装后端依赖（无需 MMCV/MMDetection）
pip install -r backend/requirements.txt

# 3. 安装内置算法库
pip install -e backend/libs/dinov3
pip install -e backend/libs/sam3

# 4. 安装前端依赖
cd frontend && npm install && cd ..
```

### 模型权重下载

- **Grounding DINO**：首次启动 Worker 时自动从 HuggingFace Hub 下载 `IDEA-Research/grounding-dino-base`，无需手动操作
- **SAM3**：首次启动时自动从 HuggingFace Hub 下载（需先 `huggingface-cli login` 获取访问权限）
- **DINOv3**：通过 `torch.hub` 自动加载预训练权重

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

### Docker 部署

```bash
docker-compose up -d
```

## 环境兼容性验证

### 验证三模型共存

```bash
cd Auto-label
python scripts/test_pipeline.py --image path/to/test.jpg
```

该脚本将依次验证：
1. DINOv3 特征提取 + K-Means 聚类
2. SAM3 bbox → Mask 分割
3. Grounding DINO 文本检测
4. 完整模式1流水线 (Grounding DINO → SAM3)

### 显存占用检测

```python
import torch
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"显存占用: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB")
print(f"显存缓存: {torch.cuda.memory_reserved(0) / 1024**3:.2f} GB")
```

预期显存占用（三模型全部加载，float16 半精度）：
- Grounding DINO Base: ~1.5 GB
- SAM3: ~3.5 GB
- DINOv3 ViT-S/16: ~0.5 GB
- **总计: ~5.5 GB**（远低于 12GB 上限）

## API 接口

| 接口 | 方法 | 说明 |
|------|------|------|
| `/api/upload` | POST | 上传图像 |
| `/api/images` | GET | 获取图片列表 |
| `/api/images/{id}` | DELETE | 删除图片 |
| `/api/annotate/mode1` | POST | 模式1：文本提示标注 |
| `/api/annotate/mode2` | POST | 模式2：框选批量标注 |
| `/api/annotate/mode3/discover` | POST | 模式3阶段1：实例发现 |
| `/api/annotate/mode3/annotate` | POST | 模式3阶段2：跨图标注 |
| `/api/tasks/{task_id}` | GET | 查询任务状态/进度/Mask URL |
| `/api/health` | GET | 健康检查 |

## 性能指标

| 指标 | 目标 |
|------|------|
| API 响应时间 | ≤ 100ms |
| 模式1 单图标注 | ≤ 2s |
| 模式2 单图匹配+Mask | ≤ 3s |
| 模式3 实例发现 | ≤ 5s |
| Mask PNG 大小 | ≤ 100KB |
| GPU 显存（三模型共存） | ≤ 10GB |

## 视觉区分

- 模式1 Mask：**多类别颜色区分**（蓝/黄/粉/青/紫等 10 色循环）
- 模式2 Mask：**红色** (255, 0, 0) alpha=128
- 模式3 Mask：**绿色** (0, 255, 0) alpha=130
- 模式3 实例发现：**多色调色板**区分不同实例

## 技术栈

| 模块 | 技术 |
|------|------|
| 后端 | FastAPI + Celery + Redis + PyTorch |
| 前端 | React 18 + TypeScript + Zustand + react-konva |
| 检测 | Grounding DINO (HuggingFace transformers，开放词汇检测) |
| 分割 | SAM3 (精准实例分割) |
| 特征 | DINOv3 ViT-S/16 (自监督特征匹配) |
| 部署 | Docker + NVIDIA CUDA 12.6 + Redis 7 |

## 配置项

所有配置均可通过环境变量覆盖：

| 环境变量 | 默认值 | 说明 |
|----------|--------|------|
| `GROUNDING_DINO_MODEL_NAME` | `IDEA-Research/grounding-dino-base` | Grounding DINO 模型名称 |
| `GROUNDING_DINO_SCORE_THR` | `0.3` | 文本检测置信度阈值 |
| `GROUNDING_DINO_SCORE_THR_LOW` | `0.2` | 低置信度重试阈值 |
| `GROUNDING_DINO_BOX_THR` | `0.3` | 边界框阈值 |
| `REDIS_HOST` | `127.0.0.1` | Redis 地址 |
| `REDIS_PORT` | `6379` | Redis 端口 |
| `SAM3_DEVICE` | `cuda` | SAM3 推理设备 |
| `DINO_DEVICE` | `cuda` | DINOv3 推理设备 |

## 从 YOLO-World 迁移说明

本项目 v2.0 将 YOLO-World 替换为 Grounding DINO，主要变更：

1. **移除所有 MM 系列依赖**：不再需要 `openmim`、`mmengine`、`mmcv`、`mmdet`、`mmyolo`
2. **模型加载方式变更**：从 mmdet 的 `init_detector` 改为 transformers 的 `AutoModelForZeroShotObjectDetection`
3. **文本提示格式**：Grounding DINO 使用 ` . ` 分隔类别（自动处理，对外接口不变）
4. **输出格式不变**：检测结果仍为 `[{"box": [x1,y1,x2,y2], "label": str, "score": float, "label_id": int}]`
5. **前端无需修改**：所有 API 接口、请求/响应格式完全不变

## License

MIT License
