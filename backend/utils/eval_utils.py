"""
标注效果验证工具
仅依赖 numpy/cv2，提供实例完整性和 mIoU 统计。
"""
import numpy as np
import logging

logger = logging.getLogger(__name__)


def calculate_instance_completeness(
    instance_masks: list[np.ndarray],
    image_shape: tuple[int, int],
) -> dict:
    """
    统计实例分割完整性指标。

    Args:
        instance_masks: 实例 Mask 列表，每个 shape [H, W] bool
        image_shape: (height, width)

    Returns:
        {
            "instance_count": int,
            "total_coverage": float,  # 所有实例覆盖的总面积占比
            "avg_area_ratio": float,  # 平均单实例面积占比
            "fragmentation": float,   # 碎片率（连通域数/实例数）
        }
    """
    import cv2

    h, w = image_shape
    total_pixels = h * w
    if not instance_masks or total_pixels == 0:
        return {"instance_count": 0, "total_coverage": 0.0,
                "avg_area_ratio": 0.0, "fragmentation": 0.0}

    union_mask = np.zeros((h, w), dtype=bool)
    area_ratios = []
    total_components = 0

    for mask in instance_masks:
        if mask.shape != (h, w):
            mask = cv2.resize(mask.astype(np.uint8), (w, h),
                              interpolation=cv2.INTER_NEAREST).astype(bool)
        union_mask |= mask
        area_ratios.append(float(mask.sum()) / total_pixels)

        num_labels, _ = cv2.connectedComponents(mask.astype(np.uint8) * 255)
        total_components += max(1, num_labels - 1)

    return {
        "instance_count": len(instance_masks),
        "total_coverage": float(union_mask.sum()) / total_pixels,
        "avg_area_ratio": float(np.mean(area_ratios)) if area_ratios else 0.0,
        "fragmentation": total_components / max(len(instance_masks), 1),
    }


def calculate_mask_miou(
    pred_mask: np.ndarray,
    gt_mask: np.ndarray,
) -> float:
    """
    计算预测 Mask 与 GT Mask 的交并比（IoU）。

    Args:
        pred_mask: 预测 Mask [H, W] bool
        gt_mask: GT Mask [H, W] bool

    Returns:
        iou: 0~1
    """
    pred = pred_mask.astype(bool)
    gt = gt_mask.astype(bool)

    intersection = (pred & gt).sum()
    union = (pred | gt).sum()

    if union == 0:
        return 1.0 if intersection == 0 else 0.0
    return float(intersection) / float(union)


def calculate_batch_miou(
    pred_masks: list[np.ndarray],
    gt_masks: list[np.ndarray],
) -> dict:
    """
    批量计算 mIoU。

    Returns:
        {"miou": float, "per_instance_iou": list[float]}
    """
    if not pred_masks or not gt_masks:
        return {"miou": 0.0, "per_instance_iou": []}

    n = min(len(pred_masks), len(gt_masks))
    ious = [calculate_mask_miou(pred_masks[i], gt_masks[i]) for i in range(n)]

    return {
        "miou": float(np.mean(ious)) if ious else 0.0,
        "per_instance_iou": ious,
    }
