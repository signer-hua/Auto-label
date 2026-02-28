"""
项目全局配置文件
所有算法库已内置于 backend/libs/ 目录中，无需外部依赖
支持通过环境变量或配置文件覆盖默认参数
"""
import os
from pathlib import Path
from pydantic import BaseModel


# ==================== 路径配置 ====================
# 项目根目录: Auto-label/
PROJECT_ROOT = Path(__file__).parent.parent
# 后端根目录: Auto-label/backend/
BACKEND_ROOT = Path(__file__).parent
# 内置算法库根目录: Auto-label/backend/libs/
LIBS_ROOT = BACKEND_ROOT / "libs"
# 上传/输出目录
UPLOAD_DIR = BACKEND_ROOT / "uploads"
OUTPUT_DIR = BACKEND_ROOT / "outputs"
UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

# 内置算法库路径
DINOV3_LIB_PATH = LIBS_ROOT / "dinov3"
SAM3_LIB_PATH = LIBS_ROOT / "sam3"
YOLOWORLD_LIB_PATH = LIBS_ROOT / "yolo_world"

# YOLO-World 配置文件路径（内置）
YOLOWORLD_CONFIG_PATH = YOLOWORLD_LIB_PATH / "configs" / "pretrain" / \
    "yolo_world_v2_s_vlpan_bn_2e-3_100e_4x8gpus_obj365v1_goldg_train_lvis_minival.py"


class ModelConfig(BaseModel):
    """模型配置"""
    # DINOv2/v3 配置
    dino_model_name: str = "dinov3_vits16"  # 使用 ViT-S/16，384维特征
    dino_device: str = "cuda"

    # SAM3 配置
    sam3_device: str = "cuda"
    sam3_checkpoint: str = ""  # 留空则自动从 HuggingFace 下载

    # YOLO-World 配置
    yoloworld_config: str = str(YOLOWORLD_CONFIG_PATH)
    yoloworld_weights: str = ""  # 需要手动下载并指定路径
    yoloworld_device: str = "cuda"
    yoloworld_score_thr: float = 0.3  # 检测置信度阈值
    yoloworld_nms_thr: float = 0.7  # NMS 阈值


class AnnotationConfig(BaseModel):
    """标注参数配置"""
    # 特征匹配阈值（模式2/3使用）
    similarity_threshold: float = 0.8
    # K-Means 聚类数（模式3使用）
    kmeans_clusters: int = 10
    # 单批次最大处理图像数（显存优化）
    max_batch_size: int = 5
    # 输出格式
    export_format: str = "coco"  # coco / voc


class ServerConfig(BaseModel):
    """服务器配置"""
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 1
    cors_origins: list = ["http://localhost:3000", "http://localhost:5173"]


# 全局配置实例
model_config = ModelConfig()
annotation_config = AnnotationConfig()
server_config = ServerConfig()
