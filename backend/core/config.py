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

# 权重文件目录（默认在项目同级的 weights/ 目录下）
WEIGHTS_DIR = Path(os.getenv("WEIGHTS_DIR", str(PROJECT_ROOT.parent / "weights")))

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
# 本地权重路径：设置为 weights/sam3.pt 则从本地加载，留空则自动从 HuggingFace 下载
SAM3_CHECKPOINT = os.getenv("SAM3_CHECKPOINT", str(WEIGHTS_DIR / "sam3.pt"))
SAM3_DEVICE = os.getenv("SAM3_DEVICE", "cuda")

# DINOv3
DINO_MODEL_NAME = os.getenv("DINO_MODEL_NAME", "dinov3_vits16")
DINO_DEVICE = os.getenv("DINO_DEVICE", "cuda")
# DINOv3 本地权重路径（如 weights/dinov3_vits16_pretrain_lvd1689m-08c60483.pth）
DINO_WEIGHTS_PATH = os.getenv("DINO_WEIGHTS_PATH", str(WEIGHTS_DIR / "dinov3_vits16_pretrain_lvd1689m-08c60483.pth"))

# Grounding DINO（使用 transformers 库加载，无 MM 系列依赖）
# 可以是 HuggingFace Hub 模型名称，也可以是本地目录路径
GROUNDING_DINO_MODEL_NAME = os.getenv(
    "GROUNDING_DINO_MODEL_NAME",
    # 优先使用本地目录（手动下载后放到 weights/grounding-dino-base/），
    # 如果本地目录不存在则回退到 HuggingFace Hub 在线加载
    str(WEIGHTS_DIR / "grounding-dino-base")
    if (WEIGHTS_DIR / "grounding-dino-base" / "config.json").exists()
    else "IDEA-Research/grounding-dino-base"
)
GROUNDING_DINO_DEVICE = os.getenv("GROUNDING_DINO_DEVICE", "cuda")
GROUNDING_DINO_SCORE_THR = float(os.getenv("GROUNDING_DINO_SCORE_THR", "0.3"))
GROUNDING_DINO_BOX_THR = float(os.getenv("GROUNDING_DINO_BOX_THR", "0.3"))
GROUNDING_DINO_SCORE_THR_LOW = float(os.getenv("GROUNDING_DINO_SCORE_THR_LOW", "0.2"))

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

# 模式3 Mask 颜色（绿色，与模式1/2 视觉区分）
MASK_MODE3_COLOR = (0, 255, 0)
MASK_MODE3_ALPHA = 130

# 模式1 多类别颜色调色板（最多 10 个类别循环使用）
MODE1_CATEGORY_COLORS = [
    (0, 120, 255),   # 蓝色
    (255, 200, 0),   # 黄色
    (255, 0, 150),   # 粉色
    (0, 220, 180),   # 青色
    (180, 80, 255),  # 紫色
    (255, 120, 0),   # 橙色
    (0, 200, 80),    # 绿色
    (200, 200, 0),   # 橄榄
    (100, 150, 255), # 浅蓝
    (255, 100, 100), # 浅红
]

# ==================== DINOv3 聚类参数（模式3） ====================
DINO_CLUSTER_NUM = int(os.getenv("DINO_CLUSTER_NUM", "8"))  # K-Means 聚类数

# ==================== DINOv3 算法优化参数 ====================
DINO_MULTI_SCALES = [0.8, 1.0, 1.2]  # 多尺度特征提取（不同视野范围）
DINO_ELBOW_MAX_K = int(os.getenv("DINO_ELBOW_MAX_K", "10"))
DINO_MIN_CLUSTER_RATIO = float(os.getenv("DINO_MIN_CLUSTER_RATIO", "0.01"))
DINO_BG_PAD_RATIO = float(os.getenv("DINO_BG_PAD_RATIO", "0.15"))

# 动态余弦相似度阈值（按目标面积比自适应调整）
COSINE_SIM_LARGE_THRESH = float(os.getenv("COSINE_SIM_LARGE_THRESH", "0.72"))
COSINE_SIM_SMALL_THRESH = float(os.getenv("COSINE_SIM_SMALL_THRESH", "0.65"))
COSINE_SIM_AREA_CUTOFF = float(os.getenv("COSINE_SIM_AREA_CUTOFF", "0.05"))

# 框选质量校验（模式2）
BBOX_MIN_AREA = int(os.getenv("BBOX_MIN_AREA", "100"))
BBOX_MAX_RATIO = float(os.getenv("BBOX_MAX_RATIO", "0.9"))

# ==================== SAM3 优化参数 ====================
SAM3_MULTIMASK_UNION = os.getenv("SAM3_MULTIMASK_UNION", "true").lower() == "true"
SAM3_SCORE_THRESHOLD = float(os.getenv("SAM3_SCORE_THRESHOLD", "0.6"))
SAM3_MORPH_KERNEL = int(os.getenv("SAM3_MORPH_KERNEL", "5"))

# ==================== 图像预处理参数 ====================
PREPROCESS_CLAHE = os.getenv("PREPROCESS_CLAHE", "true").lower() == "true"
PREPROCESS_DENOISE = os.getenv("PREPROCESS_DENOISE", "true").lower() == "true"
PREPROCESS_CLAHE_CLIP = float(os.getenv("PREPROCESS_CLAHE_CLIP", "2.0"))
PREPROCESS_CLAHE_GRID = int(os.getenv("PREPROCESS_CLAHE_GRID", "8"))

# ==================== 模式2/3 多类别颜色调色板 ====================
MULTI_CATEGORY_COLORS = [
    (255, 80, 80),    # 红色
    (80, 200, 80),    # 绿色
    (80, 120, 255),   # 蓝色
    (255, 200, 0),    # 黄色
    (200, 80, 255),   # 紫色
    (0, 220, 180),    # 青色
    (255, 120, 0),    # 橙色
    (180, 180, 0),    # 橄榄
]

# ==================== 多参考图配置 ====================
MAX_REF_IMAGES = int(os.getenv("MAX_REF_IMAGES", "5"))
DEFAULT_REF_WEIGHT = float(os.getenv("DEFAULT_REF_WEIGHT", "1.0"))
AUX_REF_WEIGHT = float(os.getenv("AUX_REF_WEIGHT", "0.8"))

# ==================== 置信度评分权重 ====================
SCORE_WEIGHT_SIMILARITY = float(os.getenv("SCORE_W_SIM", "0.40"))
SCORE_WEIGHT_MASK_COVERAGE = float(os.getenv("SCORE_W_COV", "0.35"))
SCORE_WEIGHT_AREA = float(os.getenv("SCORE_W_AREA", "0.15"))
SCORE_WEIGHT_DETECTION = float(os.getenv("SCORE_W_DET", "0.10"))

# ==================== Redis 任务过期时间 ====================
REDIS_TASK_EXPIRE = int(os.getenv("REDIS_TASK_EXPIRE", "86400"))  # 默认 24 小时

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
