"""
全局配置模块
集中管理 Redis 地址、模型路径、阈值参数、文件存储路径等。
所有配置项均可通过环境变量覆盖。
"""
import os
from pathlib import Path

# ==================== 路径配置 ====================
# 项目根目录: Auto-label/
PROJECT_ROOT = Path(__file__).parent.parent.parent
# 后端根目录: Auto-label/backend/
BACKEND_ROOT = Path(__file__).parent.parent
# 内置算法库根目录
LIBS_ROOT = BACKEND_ROOT / "libs"

# 数据存储目录
DATA_DIR = PROJECT_ROOT / "data"
IMAGE_DIR = DATA_DIR / "images"       # 上传的原始图像
MASK_DIR = DATA_DIR / "masks"         # 生成的透明 PNG Mask
EXPORT_DIR = DATA_DIR / "exports"     # COCO 格式导出

# 自动创建目录
for d in [IMAGE_DIR, MASK_DIR, EXPORT_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ==================== Redis 配置 ====================
REDIS_HOST = os.getenv("REDIS_HOST", "127.0.0.1")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
REDIS_DB = int(os.getenv("REDIS_DB", "0"))
REDIS_URL = f"redis://{REDIS_HOST}:{REDIS_PORT}/{REDIS_DB}"

# ==================== Celery 配置 ====================
CELERY_BROKER_URL = os.getenv("CELERY_BROKER_URL", REDIS_URL)
CELERY_RESULT_BACKEND = os.getenv("CELERY_RESULT_BACKEND", REDIS_URL)

# ==================== 模型配置 ====================
# SAM3
SAM3_CHECKPOINT = os.getenv("SAM3_CHECKPOINT", "")  # 留空自动从 HuggingFace 下载
SAM3_DEVICE = os.getenv("SAM3_DEVICE", "cuda")

# DINOv3
DINO_MODEL_NAME = os.getenv("DINO_MODEL_NAME", "dinov3_vits16")
DINO_DEVICE = os.getenv("DINO_DEVICE", "cuda")

# YOLO-World
YOLOWORLD_CONFIG = os.getenv(
    "YOLOWORLD_CONFIG",
    str(LIBS_ROOT / "yolo_world" / "configs" / "pretrain" /
        "yolo_world_v2_s_vlpan_bn_2e-3_100e_4x8gpus_obj365v1_goldg_train_lvis_minival.py"),
)
YOLOWORLD_WEIGHTS = os.getenv("YOLOWORLD_WEIGHTS", "")  # 需手动下载并指定路径
YOLOWORLD_DEVICE = os.getenv("YOLOWORLD_DEVICE", "cuda")
YOLOWORLD_SCORE_THR = float(os.getenv("YOLOWORLD_SCORE_THR", "0.3"))  # 文本检测置信度阈值
YOLOWORLD_NMS_THR = float(os.getenv("YOLOWORLD_NMS_THR", "0.7"))     # NMS IoU 阈值

# ==================== 标注参数 ====================
# 余弦相似度阈值（模式2 特征匹配）
COSINE_SIMILARITY_THRESHOLD = float(os.getenv("COSINE_THRESHOLD", "0.75"))
# Mask 透明 PNG 颜色配置（模式2：红色）
MASK_COLOR_R = 255
MASK_COLOR_G = 0
MASK_COLOR_B = 0
MASK_ALPHA = 128

# 模式1 Mask 颜色（蓝色，与模式2 视觉区分）
MASK_MODE1_COLOR = (0, 120, 255)
MASK_MODE1_ALPHA = 140

# ==================== 服务器配置 ====================
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "8000"))
# 静态文件 URL 前缀（前端通过此 URL 访问图片和 Mask）
STATIC_URL_PREFIX = os.getenv("STATIC_URL_PREFIX", "http://localhost:8000")
CORS_ORIGINS = [
    "http://localhost:3000",
    "http://localhost:5173",
    "http://127.0.0.1:3000",
    "http://127.0.0.1:5173",
]
