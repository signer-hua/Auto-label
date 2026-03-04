"""
动态阈值体系（ACT/ACF）
基于统计分布的自适应匹配阈值与置信度过滤。

ACT（Adaptive Change Threshold）：
    基于相似度分布的高斯核密度估计，自动确定最佳分割阈值。

ACF（Adaptive Confidence Filter）：
    基于置信度分位数动态过滤低置信结果。

所有计算仅依赖 numpy/scipy。
"""
import numpy as np
import logging

logger = logging.getLogger(__name__)


def adaptive_change_threshold(
    similarities: np.ndarray,
    fallback_threshold: float = 0.70,
    min_threshold: float = 0.55,
    max_threshold: float = 0.90,
) -> float:
    """
    基于相似度分布的高斯核密度估计，计算动态匹配阈值。

    策略：
    1. 对所有 patch 的相似度值做 KDE
    2. 在密度曲线上找到"前景/背景"的分界谷值
    3. 谷值即为自适应阈值

    Args:
        similarities: 所有 patch 的相似度值数组
        fallback_threshold: KDE 失败时的回退阈值
        min_threshold: 阈值下限
        max_threshold: 阈值上限

    Returns:
        threshold: 0~1 区间的自适应阈值
    """
    if len(similarities) < 10:
        return fallback_threshold

    try:
        from scipy.stats import gaussian_kde

        sims = similarities.flatten()
        valid = sims[(sims > 0.1) & (sims < 1.0)]
        if len(valid) < 10:
            return fallback_threshold

        kde = gaussian_kde(valid, bw_method='silverman')

        x_grid = np.linspace(
            max(valid.min(), 0.3),
            min(valid.max(), 0.95),
            200,
        )
        density = kde(x_grid)

        peaks = []
        for i in range(1, len(density) - 1):
            if density[i] > density[i - 1] and density[i] > density[i + 1]:
                peaks.append(i)

        if len(peaks) >= 2:
            p1, p2 = peaks[0], peaks[-1]
            valley_region = density[p1:p2 + 1]
            valley_idx = p1 + np.argmin(valley_region)
            threshold = float(x_grid[valley_idx])
        else:
            mean_sim = float(valid.mean())
            std_sim = float(valid.std())
            threshold = mean_sim + 0.5 * std_sim

        threshold = max(min_threshold, min(max_threshold, threshold))
        logger.info("[ACT] Adaptive threshold: %.4f (n=%d, mean=%.3f, std=%.3f)",
                    threshold, len(valid), float(valid.mean()), float(valid.std()))
        return threshold

    except Exception as e:
        logger.warning("[ACT] KDE failed, using fallback: %s", str(e))
        return fallback_threshold


def adaptive_confidence_filter(
    scores: list[float],
    quantile_low: float = 0.15,
    min_confidence: float = 0.3,
) -> float:
    """
    基于置信度分位数的动态过滤阈值。

    根据当前批次的置信度分布，计算下分位数作为过滤阈值，
    自动适应不同数据集的置信度分布特征。

    Args:
        scores: 置信度分数列表（0~1）
        quantile_low: 下分位数比例（过滤低于此分位数的结果）
        min_confidence: 最低置信度门限

    Returns:
        filter_threshold: 0~1 区间的过滤阈值
    """
    if not scores or len(scores) < 3:
        return min_confidence

    arr = np.array(scores)
    q_val = float(np.quantile(arr, quantile_low))

    iqr = float(np.quantile(arr, 0.75) - np.quantile(arr, 0.25))
    adaptive_floor = float(np.median(arr)) - 1.5 * iqr

    threshold = max(min_confidence, max(q_val, adaptive_floor))
    threshold = min(threshold, float(np.median(arr)))

    logger.info("[ACF] Confidence filter: %.4f (n=%d, median=%.3f, q15=%.3f)",
                threshold, len(scores), float(np.median(arr)), q_val)
    return threshold
