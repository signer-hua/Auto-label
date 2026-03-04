"""
置信度评分模块（v2 重构版）
为每张标注图片生成 0~100 分综合置信分数。

评分维度及权重（三维重构）：
    - 特征匹配度（40%）：DINOv3 余弦相似度
    - Mask 合理性（40%）：覆盖率 + 面积合理性综合
    - 形态完整性（20%）：孔洞率 + 连通域数量（OpenCV 计算）
"""
import numpy as np
import logging

logger = logging.getLogger(__name__)

WEIGHT_FEATURE_MATCH = 0.40
WEIGHT_MASK_QUALITY = 0.40
WEIGHT_MORPHOLOGY = 0.20

AREA_RATIO_MIN = 0.001
AREA_RATIO_MAX = 0.50
AREA_RATIO_IDEAL_MIN = 0.005
AREA_RATIO_IDEAL_MAX = 0.30


def _feature_match_score(similarities: list[float]) -> float:
    """
    特征匹配度评分。
    将相似度 [0.5, 1.0] 映射到 [0, 100]。
    """
    if not similarities:
        return 0.0
    avg = float(np.mean(similarities))
    score = max(0.0, min(100.0, (avg - 0.5) / 0.5 * 100.0))
    return round(score, 1)


def _mask_quality_score(
    coverages: list[float],
    area_ratios: list[float],
) -> float:
    """
    Mask 合理性评分（覆盖率 + 面积合理性综合）。

    覆盖率 coverage = mask_area / bbox_area，理想值 0.6~0.9
    面积合理性：过小或过大均扣分
    """
    cov_score = 0.0
    if coverages:
        avg_cov = float(np.mean(coverages))
        if 0.6 <= avg_cov <= 0.9:
            cov_score = 100.0
        elif avg_cov >= 0.4:
            cov_score = 60.0 + (avg_cov - 0.4) / 0.2 * 40.0
        elif avg_cov > 0.9:
            cov_score = max(50.0, 100.0 - (avg_cov - 0.9) / 0.1 * 50.0)
        else:
            cov_score = max(0.0, avg_cov / 0.4 * 60.0)

    area_score = 50.0
    if area_ratios:
        scores = []
        for ratio in area_ratios:
            if AREA_RATIO_IDEAL_MIN <= ratio <= AREA_RATIO_IDEAL_MAX:
                scores.append(100.0)
            elif ratio < AREA_RATIO_MIN:
                scores.append(20.0)
            elif ratio > AREA_RATIO_MAX:
                scores.append(30.0)
            elif ratio < AREA_RATIO_IDEAL_MIN:
                t = (ratio - AREA_RATIO_MIN) / (AREA_RATIO_IDEAL_MIN - AREA_RATIO_MIN)
                scores.append(20.0 + t * 80.0)
            else:
                t = (ratio - AREA_RATIO_IDEAL_MAX) / (AREA_RATIO_MAX - AREA_RATIO_IDEAL_MAX)
                scores.append(100.0 - t * 70.0)
        area_score = float(np.mean(scores))

    combined = cov_score * 0.6 + area_score * 0.4
    return round(min(100.0, max(0.0, combined)), 1)


def _morphology_score(masks: list[np.ndarray] | None = None) -> float:
    """
    形态完整性评分。
    通过 OpenCV 计算孔洞率和连通域数量评估 Mask 形态质量。

    评分规则：
    - 孔洞率 < 5%：满分
    - 连通域数量 == 1：满分
    - 孔洞率/连通域越多，扣分越重
    """
    if not masks:
        return 70.0

    import cv2

    scores = []
    for mask in masks:
        if not isinstance(mask, np.ndarray) or mask.sum() < 10:
            scores.append(50.0)
            continue

        mask_uint8 = mask.astype(np.uint8) * 255

        num_labels, _ = cv2.connectedComponents(mask_uint8)
        n_components = max(1, num_labels - 1)

        filled = cv2.morphologyEx(
            mask_uint8,
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)),
            iterations=3,
        )
        holes = (filled > 0) & (mask_uint8 == 0)
        hole_ratio = float(holes.sum()) / max(1, float(mask.sum()))

        hole_score = 100.0 if hole_ratio < 0.05 else max(30.0, 100.0 - hole_ratio * 500)
        conn_score = 100.0 if n_components == 1 else max(30.0, 100.0 - (n_components - 1) * 20)

        scores.append(hole_score * 0.5 + conn_score * 0.5)

    return round(float(np.mean(scores)), 1)


def compute_comprehensive_score(
    similarity_values: list[float],
    mask_coverages: list[float],
    area_ratios: list[float],
    masks: list[np.ndarray] | None = None,
) -> dict:
    """
    计算单张图片的综合置信分数（v2 三维重构版，0~100）。

    Args:
        similarity_values: 每个检测目标的 DINOv3 余弦相似度
        mask_coverages: 每个目标的 Mask 覆盖率
        area_ratios: 每个目标的面积占比
        masks: 每个目标的 Mask 数组列表（可选，用于形态评分）

    Returns:
        综合评分字典
    """
    feat = _feature_match_score(similarity_values)
    quality = _mask_quality_score(mask_coverages, area_ratios)
    morph = _morphology_score(masks)

    total = round(
        feat * WEIGHT_FEATURE_MATCH
        + quality * WEIGHT_MASK_QUALITY
        + morph * WEIGHT_MORPHOLOGY,
        1,
    )

    if total >= 85:
        level = "high"
    elif total >= 70:
        level = "medium"
    else:
        level = "low"

    return {
        "total": total,
        "feature_match": feat,
        "mask_quality": quality,
        "morphology": morph,
        "similarity": feat,
        "mask_coverage": quality,
        "area": round(quality * 0.4 / 0.4, 1) if area_ratios else 50.0,
        "detection": 100.0 if similarity_values else 0.0,
        "level": level,
    }


def compute_image_score(
    similarity_values: list[float],
    mask_coverages: list[float],
    area_ratios: list[float],
    detected_count: int,
    expected_count: int | None = None,
) -> dict:
    """
    兼容原有调用的评分接口（向后兼容）。
    内部调用重构后的 compute_comprehensive_score。
    """
    return compute_comprehensive_score(
        similarity_values=similarity_values,
        mask_coverages=mask_coverages,
        area_ratios=area_ratios,
        masks=None,
    )


def compute_mask_coverage(mask_area: float, bbox: list[float]) -> float:
    """
    计算 Mask 覆盖率 = mask_area / bbox_area。

    Args:
        mask_area: Mask 像素面积
        bbox: COCO 格式 [x, y, w, h]

    Returns:
        coverage: 0~1 之间的覆盖率
    """
    if len(bbox) < 4:
        return 0.0
    bbox_area = bbox[2] * bbox[3]
    if bbox_area <= 0:
        return 0.0
    return min(1.0, mask_area / bbox_area)
