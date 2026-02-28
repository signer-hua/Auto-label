"""
Mask 工具模块
核心功能：将 Tensor/NumPy 布尔矩阵转换为带 Alpha 通道的透明 PNG。
Mask 区域设为红色 (255,0,0) + Alpha 128，非目标区域 Alpha 0。
单张 1008×1008 Mask PNG 大小 ≤ 100KB。
"""
import numpy as np
from pathlib import Path
from PIL import Image
from backend.core.config import MASK_COLOR_R, MASK_COLOR_G, MASK_COLOR_B, MASK_ALPHA


def mask_to_transparent_png(
    mask: np.ndarray,
    save_path: str | Path,
    color: tuple[int, int, int] = (MASK_COLOR_R, MASK_COLOR_G, MASK_COLOR_B),
    alpha: int = MASK_ALPHA,
) -> Path:
    """
    将布尔 Mask 矩阵转换为带 Alpha 通道的透明 PNG 并保存。

    Args:
        mask: 布尔 numpy 数组，shape [H, W]，True 表示目标区域
        save_path: PNG 文件保存路径
        color: 目标区域 RGB 颜色，默认红色 (255, 0, 0)
        alpha: 目标区域 Alpha 值 (0-255)，默认 128（半透明）

    Returns:
        save_path: 保存后的文件路径

    Raises:
        ValueError: mask 维度不正确
    """
    save_path = Path(save_path)

    # 确保 mask 是 2D 布尔数组
    if mask.ndim != 2:
        raise ValueError(f"Mask must be 2D, got shape {mask.shape}")
    mask = mask.astype(bool)

    h, w = mask.shape
    # 创建 RGBA 图像，初始全透明
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    # 目标区域填充颜色 + Alpha
    rgba[mask, 0] = color[0]  # R
    rgba[mask, 1] = color[1]  # G
    rgba[mask, 2] = color[2]  # B
    rgba[mask, 3] = alpha     # Alpha

    # 保存为 PNG（自动压缩，1008×1008 通常 ≤ 100KB）
    img = Image.fromarray(rgba, mode="RGBA")
    save_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(str(save_path), format="PNG", optimize=True)

    return save_path


def mask_to_polygon(mask: np.ndarray) -> list[list[float]]:
    """
    将布尔 Mask 转换为多边形坐标（COCO segmentation 格式）。

    Args:
        mask: 布尔 numpy 数组，shape [H, W]
    Returns:
        polygons: [[x1,y1,x2,y2,...], ...]
    """
    import cv2
    contours, _ = cv2.findContours(
        mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    polygons = []
    for contour in contours:
        if len(contour) < 3:
            continue
        polygon = contour.flatten().tolist()
        if len(polygon) >= 6:
            polygons.append(polygon)
    return polygons


def mask_to_bbox(mask: np.ndarray) -> list[float]:
    """
    从 Mask 计算 COCO 格式边界框 [x, y, width, height]。

    Args:
        mask: 布尔 numpy 数组
    Returns:
        bbox: [x, y, w, h]
    """
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return [0.0, 0.0, 0.0, 0.0]
    x1, y1, x2, y2 = float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())
    return [x1, y1, x2 - x1, y2 - y1]
