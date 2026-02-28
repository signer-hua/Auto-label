"""
Celery Worker 入口
注册异步任务，实现模型预热。

启动命令（强制并发数为 1，避免 GPU OOM）：
    celery -A backend.worker worker --concurrency=1 --pool=solo -l info

核心任务：
    process_batch_annotation: 模式2 批量标注异步任务
"""
import json
import time
import logging
import numpy as np
from pathlib import Path
from PIL import Image
from celery import Celery
from celery.signals import worker_process_init

from backend.core.config import (
    CELERY_BROKER_URL, CELERY_RESULT_BACKEND,
    REDIS_URL, COSINE_SIMILARITY_THRESHOLD,
)

logger = logging.getLogger(__name__)

# ==================== Celery 应用配置 ====================
celery_app = Celery(
    "auto_label",
    broker=CELERY_BROKER_URL,
    backend=CELERY_RESULT_BACKEND,
)
celery_app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    task_track_started=True,
    # 任务超时：单张图 ≤ 2s，100 张图 ≤ 300s
    task_time_limit=600,
    task_soft_time_limit=300,
)


def _get_redis():
    """获取 Redis 连接（延迟导入避免循环依赖）"""
    import redis
    return redis.Redis.from_url(REDIS_URL, decode_responses=True)


# ==================== Worker 启动时预热模型 ====================
@worker_process_init.connect
def warmup_models(**kwargs):
    """
    Celery Worker 进程启动时自动调用。
    预热 SAM3 和 DINOv3 模型到 GPU (half precision)。
    """
    logger.info("=" * 60)
    logger.info("[Worker] Warming up models...")
    logger.info("=" * 60)

    from backend.models.sam_engine import SAMEngine
    from backend.models.dino_engine import DINOEngine

    sam = SAMEngine.get_instance()
    sam.warmup()

    dino = DINOEngine.get_instance()
    dino.warmup()

    logger.info("[Worker] All models warmed up. Ready to process tasks.")


# ==================== 异步任务：模式2 批量标注 ====================
@celery_app.task(name="process_batch_annotation", bind=True)
def process_batch_annotation(
    self,
    ref_image_path: str,
    bbox: list[float],
    target_image_paths: list[str],
    target_image_ids: list[str],
    task_id: str,
):
    """
    模式2 批量标注异步任务。

    全链路流程：
    1. SAM3 对首图 bbox 区域生成 Mask
    2. DINOv3 提取 Mask 区域特征 → 特征模板
    3. 遍历目标图：
       a. DINOv3 特征匹配 → 找到相似区域 bbox
       b. SAM3 对匹配 bbox 生成精准 Mask
       c. Mask → 透明 PNG 保存
    4. 实时更新 Redis 进度/状态

    Args:
        ref_image_path: 首图（参考图）本地路径
        bbox: 用户框选的 [x1, y1, x2, y2] 像素坐标
        target_image_paths: 待标注图像路径列表
        target_image_ids: 待标注图像 ID 列表
        task_id: 任务 ID（用于 Redis 状态更新）
    """
    from backend.models.sam_engine import SAMEngine
    from backend.models.dino_engine import DINOEngine
    from backend.services.mask_utils import mask_to_transparent_png, mask_to_bbox, mask_to_polygon
    from backend.services.storage import get_mask_path, get_mask_url, get_export_path, get_export_url

    r = _get_redis()
    total = len(target_image_paths)

    try:
        # 更新状态为 processing
        r.hset(f"task:{task_id}", mapping={
            "status": "processing",
            "progress": 0,
            "total": total,
            "message": "Generating reference mask...",
        })

        sam = SAMEngine.get_instance()
        dino = DINOEngine.get_instance()

        # ========== Step 1: SAM3 生成首图 Mask ==========
        ref_image = Image.open(ref_image_path).convert("RGB")
        ref_mask = sam.generate_mask(ref_image, bbox)
        logger.info("[Task %s] Reference mask generated, area=%d", task_id, ref_mask.sum())

        if ref_mask.sum() < 10:
            raise ValueError("Reference mask is too small. Please adjust the bounding box.")

        # ========== Step 2: DINOv3 提取特征模板 ==========
        r.hset(f"task:{task_id}", "message", "Extracting feature template...")
        template_feature = dino.extract_mask_feature(ref_image, ref_mask)
        logger.info("[Task %s] Template feature extracted, dim=%d", task_id, len(template_feature))

        # ========== Step 3: 遍历目标图批量标注 ==========
        mask_urls = {}       # {image_id: [mask_url, ...]}
        coco_annotations = []  # COCO 格式标注
        annotation_id = 1

        for idx, (img_path, img_id) in enumerate(zip(target_image_paths, target_image_ids)):
            t0 = time.time()

            # 更新进度
            r.hset(f"task:{task_id}", mapping={
                "progress": idx,
                "message": f"Processing image {idx + 1}/{total}...",
            })

            target_image = Image.open(img_path).convert("RGB")
            img_w, img_h = target_image.size

            # Step 3a: DINOv3 特征匹配
            match_mask, match_bboxes = dino.match_in_image(
                target_image, template_feature, threshold=COSINE_SIMILARITY_THRESHOLD
            )

            image_mask_urls = []

            if len(match_bboxes) > 0:
                for inst_idx, matched_bbox in enumerate(match_bboxes):
                    # Step 3b: SAM3 对匹配 bbox 生成精准 Mask
                    precise_mask = sam.generate_mask(target_image, matched_bbox)

                    # Step 3c: Mask → 透明 PNG
                    mask_path = get_mask_path(img_id, inst_idx)
                    mask_to_transparent_png(precise_mask, mask_path)
                    mask_url = get_mask_url(img_id, inst_idx)
                    image_mask_urls.append(mask_url)

                    # 构建 COCO 标注
                    coco_annotations.append({
                        "id": annotation_id,
                        "image_id": img_id,
                        "category_id": 1,
                        "bbox": mask_to_bbox(precise_mask),
                        "segmentation": mask_to_polygon(precise_mask),
                        "area": float(precise_mask.sum()),
                        "iscrowd": 0,
                    })
                    annotation_id += 1

            mask_urls[img_id] = image_mask_urls

            elapsed = time.time() - t0
            logger.info("[Task %s] Image %d/%d done (%.2fs), %d instances",
                        task_id, idx + 1, total, elapsed, len(image_mask_urls))

        # ========== Step 4: 保存 COCO 导出 ==========
        # 构建 images 列表（填充实际 width/height）
        coco_images = []
        for img_path, img_id in zip(target_image_paths, target_image_ids):
            try:
                with Image.open(img_path) as _img:
                    w, h = _img.size
            except Exception:
                w, h = 0, 0
            coco_images.append({
                "id": img_id,
                "file_name": Path(img_path).name,
                "width": w,
                "height": h,
            })

        coco_result = {
            "images": coco_images,
            "annotations": coco_annotations,
            "categories": [{"id": 1, "name": "target", "supercategory": "object"}],
        }
        export_path = get_export_path(task_id)
        export_path.write_text(json.dumps(coco_result, ensure_ascii=False, indent=2))
        export_url = get_export_url(task_id)

        # ========== Step 5: 更新 Redis 为 success ==========
        r.hset(f"task:{task_id}", mapping={
            "status": "success",
            "progress": total,
            "total": total,
            "message": f"Completed. {annotation_id - 1} instances found.",
            "mask_urls": json.dumps(mask_urls),
            "export_url": export_url,
        })
        # 设置 24 小时过期
        r.expire(f"task:{task_id}", 86400)

        # 释放显存
        sam.release_memory()
        dino.release_memory()

        logger.info("[Task %s] DONE. Total annotations: %d", task_id, annotation_id - 1)
        return {"status": "success", "task_id": task_id}

    except Exception as e:
        logger.exception("[Task %s] FAILED: %s", task_id, str(e))
        r.hset(f"task:{task_id}", mapping={
            "status": "failed",
            "message": str(e),
        })
        r.expire(f"task:{task_id}", 86400)
        return {"status": "failed", "task_id": task_id, "error": str(e)}
