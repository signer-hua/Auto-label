"""
图像预处理工具模块
基于 OpenCV 实现输入分辨率统一限制，适配消费级 GPU 显存。
"""
import cv2
import numpy as np
from PIL import Image
import logging

logger = logging.getLogger(__name__)

MAX_LONG_EDGE = 1024


def resize_image(
    image: Image.Image,
    max_long_edge: int = MAX_LONG_EDGE,
) -> Image.Image:
    """
    将图片长边统一缩放至指定像素（等比例缩放），
    控制模型输入分辨率，降低显存占用。

    若图片长边已 <= max_long_edge，则返回原图不做处理。

    Args:
        image: PIL Image (RGB)
        max_long_edge: 长边最大像素数，默认1024

    Returns:
        resized: 等比缩放后的 PIL Image
    """
    w, h = image.size
    long_edge = max(w, h)

    if long_edge <= max_long_edge:
        return image

    scale = max_long_edge / long_edge
    new_w = int(w * scale)
    new_h = int(h * scale)

    img_np = np.array(image)
    resized_np = cv2.resize(img_np, (new_w, new_h), interpolation=cv2.INTER_AREA)
    resized = Image.fromarray(resized_np)

    logger.info("[ImageUtils] Resized %dx%d -> %dx%d (scale=%.3f)",
                w, h, new_w, new_h, scale)
    return resized


def get_resize_scale(image: Image.Image, max_long_edge: int = MAX_LONG_EDGE) -> float:
    """
    计算图片缩放比例（不实际缩放）。
    用于将缩放后的坐标映射回原图坐标。

    Returns:
        scale: 缩放比例（<= 1.0）
    """
    w, h = image.size
    long_edge = max(w, h)
    if long_edge <= max_long_edge:
        return 1.0
    return max_long_edge / long_edge
