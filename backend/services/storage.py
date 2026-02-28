"""
文件存储服务
负责：本地路径生成、文件保存、URL 拼接。
图片存储在 /data/images/，Mask 存储在 /data/masks/。
"""
import uuid
import shutil
from pathlib import Path
from fastapi import UploadFile

from backend.core.config import IMAGE_DIR, MASK_DIR, EXPORT_DIR, STATIC_URL_PREFIX

# 支持的图像格式
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}


def generate_image_id() -> str:
    """生成唯一图像 ID（UUID hex）"""
    return uuid.uuid4().hex


def get_image_path(image_id: str, ext: str = ".jpg") -> Path:
    """
    根据图像 ID 获取本地存储路径。

    Args:
        image_id: 唯一图像 ID
        ext: 文件扩展名
    Returns:
        本地文件路径
    """
    return IMAGE_DIR / f"{image_id}{ext}"


def get_mask_path(image_id: str, instance_idx: int = 0) -> Path:
    """
    根据图像 ID 和实例索引获取 Mask PNG 存储路径。

    Args:
        image_id: 图像 ID
        instance_idx: 实例索引（同一张图可能有多个 Mask）
    Returns:
        Mask PNG 文件路径
    """
    return MASK_DIR / f"{image_id}_mask_{instance_idx}.png"


def get_image_url(image_id: str, ext: str = ".jpg") -> str:
    """
    获取图像的相对 URL（前端通过 Vite proxy 或 Nginx 代理访问）。
    使用相对路径，避免硬编码 host:port。
    """
    return f"/data/images/{image_id}{ext}"


def get_mask_url(image_id: str, instance_idx: int = 0) -> str:
    """
    获取 Mask PNG 的相对 URL。
    """
    return f"/data/masks/{image_id}_mask_{instance_idx}.png"


async def save_upload_file(file: UploadFile) -> tuple[str, str, Path]:
    """
    保存上传的图像文件到本地。

    Args:
        file: FastAPI UploadFile 对象
    Returns:
        (image_id, original_filename, saved_path)
    Raises:
        ValueError: 不支持的文件格式
    """
    original_name = file.filename or "unknown.jpg"
    ext = Path(original_name).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise ValueError(f"Unsupported format: {ext}")

    image_id = generate_image_id()
    save_path = get_image_path(image_id, ext)

    # 写入本地文件
    content = await file.read()
    save_path.write_bytes(content)

    return image_id, original_name, save_path


def get_export_path(task_id: str) -> Path:
    """获取 COCO 导出文件路径"""
    return EXPORT_DIR / f"{task_id}_coco.json"


def get_export_url(task_id: str) -> str:
    """获取 COCO 导出文件相对 URL"""
    return f"/data/exports/{task_id}_coco.json"
