"""
置信度评分模块
为每张标注图片生成 0~100 分综合置信分数。

评分维度及权重：
    - DINOv3 特征匹配相似度均值（40%）
    - SAM3 Mask 与目标外接矩形的覆盖占比均值（35%）
    - 目标区域面积合理性（15%，过滤过小/过大异常目标）
    - 漏检风险反向计分（10%，无漏检得满分）

使用方式：
    在 worker.py 标注循环中收集每张图的评分原始数据，
    标注完成后调用 compute_image_score 计算综合分数。
"""
import numpy as np
import logging

logger = logging.getLogger(__name__)

WEIGHT_SIMILARITY = 0.40
WEIGHT_MASK_COVERAGE = 0.35
WEIGHT_AREA = 0.15
WEIGHT_DETECTION = 0.10

AREA_RATIO_MIN = 0.001   # 面积占比下限（<此值视为过小）
AREA_RATIO_MAX = 0.50    # 面积占比上限（>此值视为过大）
AREA_RATIO_IDEAL_MIN = 0.005
AREA_RATIO_IDEAL_MAX = 0.30


def _similarity_score(similarities: list[float]) -> float:
    """
    特征匹配相似度评分。
    相似度越高 → 匹配越准确。将 [0.5, 1.0] 映射到 [0, 100]。
    """
    if not similarities:
        return 0.0
    avg = float(np.mean(similarities))
    score = max(0.0, min(100.0, (avg - 0.5) / 0.5 * 100.0))
    return round(score, 1)


def _mask_coverage_score(coverages: list[float]) -> float:
    """
    Mask 覆盖率评分。
    coverage = mask_area / bbox_area，理想值接近 0.6~0.9。
    过低说明 Mask 不完整，过高可能包含背景。
    """
    if not coverages:
        return 0.0
    avg = float(np.mean(coverages))
    if avg >= 0.6 and avg <= 0.9:
        score = 100.0
    elif avg >= 0.4:
        score = 60.0 + (avg - 0.4) / 0.2 * 40.0
    elif avg > 0.9:
        score = max(50.0, 100.0 - (avg - 0.9) / 0.1 * 50.0)
    else:
        score = max(0.0, avg / 0.4 * 60.0)
    return round(min(100.0, max(0.0, score)), 1)


def _area_score(area_ratios: list[float]) -> float:
    """
    目标面积合理性评分。
    过小（< AREA_RATIO_MIN）或过大（> AREA_RATIO_MAX）扣分。
    """
    if not area_ratios:
        return 50.0
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
    return round(float(np.mean(scores)), 1)


def _detection_score(detected_count: int, expected_count: int | None = None) -> float:
    """
    检测完整性评分（漏检风险反向计分）。
    检测到目标 → 满分；未检测到 → 0 分。
    若已知预期数量，按检出比例计分。
    """
    if detected_count <= 0:
        return 0.0
    if expected_count is None or expected_count <= 0:
        return 100.0
    ratio = min(1.0, detected_count / expected_count)
    return round(ratio * 100.0, 1)


def compute_image_score(
    similarity_values: list[float],
    mask_coverages: list[float],
    area_ratios: list[float],
    detected_count: int,
    expected_count: int | None = None,
) -> dict:
    """
    计算单张图片的综合置信分数（0~100）。

    Args:
        similarity_values: 每个检测目标的 DINOv3 余弦相似度
        mask_coverages: 每个目标的 Mask覆盖率 = mask_area / bbox_area
        area_ratios: 每个目标的面积占比 = mask_area / image_area
        detected_count: 本图检测到的目标数量
        expected_count: 预期目标数量（可选，用于漏检评估）

    Returns:
        {
            "total": float,         # 综合分 0~100
            "similarity": float,    # 匹配度分项
            "mask_coverage": float, # Mask完整性分项
            "area": float,          # 面积合理性分项
            "detection": float,     # 检测完整性分项
            "level": str,           # "high" / "medium" / "low"
        }
    """
    sim = _similarity_score(similarity_values)
    cov = _mask_coverage_score(mask_coverages)
    area = _area_score(area_ratios)
    det = _detection_score(detected_count, expected_count)

    total = round(
        sim * WEIGHT_SIMILARITY
        + cov * WEIGHT_MASK_COVERAGE
        + area * WEIGHT_AREA
        + det * WEIGHT_DETECTION,
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
        "similarity": sim,
        "mask_coverage": cov,
        "area": area,
        "detection": det,
        "level": level,
    }


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
