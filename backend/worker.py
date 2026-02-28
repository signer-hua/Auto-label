"""
Celery Worker 入口
注册异步任务，实现模型预热。

启动命令（强制并发数为 1，避免 GPU OOM）：
    celery -A backend.worker worker --concurrency=1 --pool=solo -l info

核心任务：
    process_batch_annotation:    模式2 框选批量标注
    process_text_annotation:     模式1 文本提示标注
    process_instance_annotation: 模式3 选实例跨图标注
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
    MASK_MODE1_COLOR, MASK_MODE1_ALPHA,
    MASK_MODE3_COLOR, MASK_MODE3_ALPHA,
    MODE1_CATEGORY_COLORS, REDIS_TASK_EXPIRE,
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
    task_time_limit=600,
    task_soft_time_limit=300,
)


def _get_redis():
    """获取 Redis 连接（延迟导入避免循环依赖）"""
    import redis
    return redis.Redis.from_url(REDIS_URL, decode_responses=True)


def _check_task_status(r, task_id: str) -> str:
    """
    检查 Redis 中任务状态，支持暂停/取消控制。

    Returns:
        status: 'processing' | 'paused' | 'canceled'
    """
    status = r.hget(f"task:{task_id}", "status")
    if status == "canceled":
        return "canceled"
    if status == "paused":
        # 暂停时循环等待，直到 resumed 或 canceled
        logger.info("[Task %s] Paused, waiting for resume...", task_id)
        while status == "paused":
            time.sleep(1)
            status = r.hget(f"task:{task_id}", "status")
            if status == "canceled":
                return "canceled"
        # resumed → 恢复为 processing
        r.hset(f"task:{task_id}", "status", "processing")
    return "processing"


def _release_all_models():
    """统一释放所有模型的 GPU 显存"""
    from backend.models.sam_engine import SAMEngine
    from backend.models.dino_engine import DINOEngine
    from backend.models.yolo_engine import YOLOWorldEngine

    SAMEngine.get_instance().release_memory()
    DINOEngine.get_instance().release_memory()
    YOLOWorldEngine.get_instance().release_memory()


def _handle_gpu_oom(r, task_id: str, e: Exception):
    """检测 GPU OOM 并设置友好提示"""
    msg = str(e)
    if "out of memory" in msg.lower() or "CUDA" in msg:
        r.hset(f"task:{task_id}", mapping={
            "status": "failed",
            "message": "GPU 显存不足，建议压缩图片分辨率或关闭其他程序",
            "error_type": "gpu_oom",
        })
    else:
        r.hset(f"task:{task_id}", mapping={
            "status": "failed",
            "message": msg,
        })
    r.expire(f"task:{task_id}", REDIS_TASK_EXPIRE)


# ==================== Worker 启动时预热模型 ====================
@worker_process_init.connect
def warmup_models(**kwargs):
    """
    Celery Worker 进程启动时自动调用。
    预热 SAM3、DINOv3、YOLO-World 三个模型到 GPU (half precision)。
    """
    logger.info("=" * 60)
    logger.info("[Worker] Warming up all models...")
    logger.info("=" * 60)

    from backend.models.sam_engine import SAMEngine
    from backend.models.dino_engine import DINOEngine
    from backend.models.yolo_engine import YOLOWorldEngine

    SAMEngine.get_instance().warmup()
    DINOEngine.get_instance().warmup()
    YOLOWorldEngine.get_instance().warmup()

    logger.info("[Worker] All models (SAM3 + DINOv3 + YOLO-World) warmed up. Ready.")


# PLACEHOLDER: tasks will be added via StrReplace


# ==================== 异步任务：模式2 框选批量标注 ====================
@celery_app.task(name="process_batch_annotation", bind=True, max_retries=2, retry_backoff=5)
def process_batch_annotation(
    self,
    ref_image_path: str,
    bbox: list[float],
    target_image_paths: list[str],
    target_image_ids: list[str],
    task_id: str,
):
    """
    模式2 框选批量标注异步任务。

    链路：SAM3(首图bbox→Mask) → DINOv3(特征模板) → 遍历目标图(DINOv3匹配→SAM3精准Mask)
    支持暂停/取消/重试。
    """
    from backend.models.sam_engine import SAMEngine
    from backend.models.dino_engine import DINOEngine
    from backend.services.mask_utils import mask_to_transparent_png, mask_to_bbox, mask_to_polygon
    from backend.services.storage import get_mask_path, get_mask_url, get_export_path, get_export_url

    r = _get_redis()
    total = len(target_image_paths)

    try:
        r.hset(f"task:{task_id}", mapping={
            "status": "processing", "progress": 0, "total": total,
            "message": "Generating reference mask...", "mode": "mode2",
        })

        sam = SAMEngine.get_instance()
        dino = DINOEngine.get_instance()

        # Step 1: SAM3 生成首图 Mask
        ref_image = Image.open(ref_image_path).convert("RGB")
        ref_mask = sam.generate_mask(ref_image, bbox)
        if ref_mask.sum() < 10:
            raise ValueError("Reference mask is too small. Please adjust the bounding box.")

        # Step 2: DINOv3 提取特征模板
        r.hset(f"task:{task_id}", "message", "Extracting feature template...")
        template_feature = dino.extract_mask_feature(ref_image, ref_mask)

        # Step 3: 遍历目标图批量标注
        mask_urls = {}
        coco_annotations = []
        coco_images = []
        annotation_id = 1

        for idx, (img_path, img_id) in enumerate(zip(target_image_paths, target_image_ids)):
            # 检查暂停/取消
            ctrl = _check_task_status(r, task_id)
            if ctrl == "canceled":
                r.hset(f"task:{task_id}", mapping={"status": "canceled", "message": "Task canceled by user."})
                r.expire(f"task:{task_id}", REDIS_TASK_EXPIRE)
                _release_all_models()
                return {"status": "canceled", "task_id": task_id}

            t0 = time.time()
            r.hset(f"task:{task_id}", mapping={
                "progress": idx, "message": f"Processing image {idx + 1}/{total}...",
            })

            target_image = Image.open(img_path).convert("RGB")
            img_w, img_h = target_image.size

            coco_images.append({
                "id": img_id, "file_name": Path(img_path).name,
                "width": img_w, "height": img_h,
            })

            # DINOv3 特征匹配
            match_mask, match_bboxes = dino.match_in_image(
                target_image, template_feature, threshold=COSINE_SIMILARITY_THRESHOLD
            )

            image_mask_urls = []
            for inst_idx, matched_bbox in enumerate(match_bboxes):
                precise_mask = sam.generate_mask(target_image, matched_bbox)
                mask_path = get_mask_path(img_id, inst_idx)
                mask_to_transparent_png(precise_mask, mask_path)
                mask_url = get_mask_url(img_id, inst_idx)
                image_mask_urls.append(mask_url)

                coco_annotations.append({
                    "id": annotation_id, "image_id": img_id, "category_id": 1,
                    "bbox": mask_to_bbox(precise_mask),
                    "segmentation": mask_to_polygon(precise_mask),
                    "area": float(precise_mask.sum()), "iscrowd": 0,
                })
                annotation_id += 1

            mask_urls[img_id] = image_mask_urls
            elapsed = time.time() - t0
            logger.info("[Task %s] Image %d/%d done (%.2fs), %d instances",
                        task_id, idx + 1, total, elapsed, len(image_mask_urls))

        # COCO 导出
        coco_result = {
            "images": coco_images,
            "annotations": coco_annotations,
            "categories": [{"id": 1, "name": "target", "supercategory": "object"}],
        }
        export_path = get_export_path(task_id)
        export_path.write_text(json.dumps(coco_result, ensure_ascii=False, indent=2))

        r.hset(f"task:{task_id}", mapping={
            "status": "success", "progress": total, "total": total,
            "message": f"Completed. {annotation_id - 1} instances found.",
            "mask_urls": json.dumps(mask_urls),
            "export_url": get_export_url(task_id),
        })
        r.expire(f"task:{task_id}", REDIS_TASK_EXPIRE)
        _release_all_models()
        return {"status": "success", "task_id": task_id}

    except Exception as e:
        logger.exception("[Task %s] Mode2 FAILED: %s", task_id, str(e))
        _handle_gpu_oom(r, task_id, e)
        _release_all_models()
        # 自动重试
        if self.request.retries < self.max_retries and "out of memory" not in str(e).lower():
            raise self.retry(exc=e)
        return {"status": "failed", "task_id": task_id, "error": str(e)}


# ==================== 异步任务：模式1 文本提示标注 ====================
@celery_app.task(name="process_text_annotation", bind=True, max_retries=2, retry_backoff=5)
def process_text_annotation(
    self,
    image_paths: list[str],
    image_ids: list[str],
    text_prompt: str,
    task_id: str,
):
    """
    模式1 文本提示标注异步任务。

    链路：文本 → YOLO-World 检测 bbox → SAM3 精准 Mask → 蓝色透明 PNG
    支持多类别颜色区分、暂停/取消/重试。
    """
    from backend.models.sam_engine import SAMEngine
    from backend.models.yolo_engine import YOLOWorldEngine
    from backend.services.mask_utils import mask_to_transparent_png, mask_to_bbox, mask_to_polygon
    from backend.services.storage import get_mask_path, get_mask_url, get_export_path, get_export_url
    from backend.core.config import YOLOWORLD_SCORE_THR_LOW

    r = _get_redis()
    total = len(image_paths)

    try:
        r.hset(f"task:{task_id}", mapping={
            "status": "processing", "progress": 0, "total": total,
            "message": "Starting text-driven annotation...", "mode": "mode1",
        })

        sam = SAMEngine.get_instance()
        yolo = YOLOWorldEngine.get_instance()

        # 解析文本提示为类别列表
        categories = [c.strip() for c in text_prompt.replace('，', ',').split(',') if c.strip()]
        if not categories:
            raise ValueError("Text prompt is empty after parsing")

        logger.info("[Task %s] Mode1: categories=%s, images=%d", task_id, categories, total)

        mask_urls = {}
        coco_annotations = []
        coco_images = []
        annotation_id = 1

        coco_categories = [
            {"id": i + 1, "name": cat, "supercategory": "object"}
            for i, cat in enumerate(categories)
        ]

        for idx, (img_path, img_id) in enumerate(zip(image_paths, image_ids)):
            # 检查暂停/取消
            ctrl = _check_task_status(r, task_id)
            if ctrl == "canceled":
                r.hset(f"task:{task_id}", mapping={"status": "canceled", "message": "Task canceled by user."})
                r.expire(f"task:{task_id}", REDIS_TASK_EXPIRE)
                _release_all_models()
                return {"status": "canceled", "task_id": task_id}

            t0 = time.time()
            r.hset(f"task:{task_id}", mapping={
                "progress": idx, "message": f"Processing image {idx + 1}/{total}...",
            })

            target_image = Image.open(img_path).convert("RGB")
            img_w, img_h = target_image.size

            coco_images.append({
                "id": img_id, "file_name": Path(img_path).name,
                "width": img_w, "height": img_h,
            })

            # YOLO-World 文本检测
            detections = yolo.detect(target_image, categories)

            # 无检测结果时自动降低阈值重试一次
            if len(detections) == 0:
                logger.info("[Task %s] No detections at default threshold, retrying with low threshold...", task_id)
                detections = yolo.detect(target_image, categories, score_thr=YOLOWORLD_SCORE_THR_LOW)

            image_mask_urls = []
            inst_idx = 0

            for det in detections:
                bbox_det = det["box"]
                label_id = det["label_id"]
                score = det["score"]

                precise_mask = sam.generate_mask(target_image, bbox_det)
                if precise_mask.sum() < 10:
                    continue

                # 多类别颜色区分
                color = MODE1_CATEGORY_COLORS[label_id % len(MODE1_CATEGORY_COLORS)]

                mask_path = get_mask_path(img_id, inst_idx)
                mask_to_transparent_png(precise_mask, mask_path, color=color, alpha=MASK_MODE1_ALPHA)
                mask_url = get_mask_url(img_id, inst_idx)
                image_mask_urls.append(mask_url)

                coco_annotations.append({
                    "id": annotation_id, "image_id": img_id,
                    "category_id": label_id + 1,
                    "bbox": mask_to_bbox(precise_mask),
                    "segmentation": mask_to_polygon(precise_mask),
                    "area": float(precise_mask.sum()),
                    "score": score, "iscrowd": 0,
                })
                annotation_id += 1
                inst_idx += 1

            mask_urls[img_id] = image_mask_urls
            elapsed = time.time() - t0
            logger.info("[Task %s] Image %d/%d done (%.2fs), %d masks",
                        task_id, idx + 1, total, elapsed, len(image_mask_urls))

        # COCO 导出
        coco_result = {
            "images": coco_images,
            "annotations": coco_annotations,
            "categories": coco_categories,
        }
        export_path = get_export_path(task_id)
        export_path.write_text(json.dumps(coco_result, ensure_ascii=False, indent=2))

        total_annotations = annotation_id - 1
        r.hset(f"task:{task_id}", mapping={
            "status": "success", "progress": total, "total": total,
            "message": f"Completed. {total_annotations} instances found across {total} images.",
            "mask_urls": json.dumps(mask_urls),
            "export_url": get_export_url(task_id),
        })
        r.expire(f"task:{task_id}", REDIS_TASK_EXPIRE)
        _release_all_models()
        return {"status": "success", "task_id": task_id}

    except Exception as e:
        logger.exception("[Task %s] Mode1 FAILED: %s", task_id, str(e))
        _handle_gpu_oom(r, task_id, e)
        _release_all_models()
        if self.request.retries < self.max_retries and "out of memory" not in str(e).lower():
            raise self.retry(exc=e)
        return {"status": "failed", "task_id": task_id, "error": str(e)}


# ==================== 异步任务：模式3 选实例跨图标注 ====================
# 模式3 分两阶段：
#   阶段1 (process_instance_discovery): 粗分割实例生成 → 返回实例 Mask URL 列表
#   阶段2 (process_instance_annotation): 用户选中实例后 → 跨图批量标注

@celery_app.task(name="process_instance_discovery", bind=True, max_retries=2, retry_backoff=5)
def process_instance_discovery(
    self,
    ref_image_path: str,
    ref_image_id: str,
    task_id: str,
):
    """
    模式3 阶段1：参考图粗分割实例生成。

    链路：DINOv3 全图 patch 特征 → K-Means 聚类 → SAM3 对每个聚类区域生成 Mask
    结果存入 Redis instance_masks 字段，前端渲染供用户选择。
    """
    from backend.models.sam_engine import SAMEngine
    from backend.models.dino_engine import DINOEngine
    from backend.services.mask_utils import mask_to_transparent_png, mask_to_bbox
    from backend.services.storage import get_mask_path, get_mask_url

    r = _get_redis()

    try:
        r.hset(f"task:{task_id}", mapping={
            "status": "processing", "progress": 0, "total": 1,
            "message": "Generating instance proposals...", "mode": "mode3_discovery",
        })

        dino = DINOEngine.get_instance()
        sam = SAMEngine.get_instance()

        ref_image = Image.open(ref_image_path).convert("RGB")
        img_w, img_h = ref_image.size

        # DINOv3 K-Means 聚类
        cluster_map, instance_masks, h, w = dino.extract_patch_features_clustering(ref_image)

        if len(instance_masks) == 0:
            r.hset(f"task:{task_id}", mapping={
                "status": "failed",
                "message": "参考图未检测到可分割实例，请更换图片或调整聚类参数",
            })
            r.expire(f"task:{task_id}", REDIS_TASK_EXPIRE)
            return {"status": "failed", "task_id": task_id}

        # 为每个聚类实例生成 SAM3 精准 Mask
        # 用彩色调色板区分不同实例
        instance_colors = [
            (255, 80, 80), (80, 200, 80), (80, 120, 255),
            (255, 200, 0), (200, 80, 255), (0, 220, 180),
            (255, 120, 0), (180, 180, 0),
        ]

        instance_data = []  # [{id, mask_url, bbox, color}]
        for i, inst_mask in enumerate(instance_masks):
            # 从聚类 Mask 计算 bbox → SAM3 精准分割
            ys, xs = np.where(inst_mask)
            if len(xs) == 0:
                continue
            inst_bbox = [float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())]

            precise_mask = sam.generate_mask(ref_image, inst_bbox)
            if precise_mask.sum() < 50:
                continue

            color = instance_colors[i % len(instance_colors)]
            mask_path = get_mask_path(f"{ref_image_id}_inst", i)
            mask_to_transparent_png(precise_mask, mask_path, color=color, alpha=150)
            mask_url = get_mask_url(f"{ref_image_id}_inst", i)

            instance_data.append({
                "id": i,
                "mask_url": mask_url,
                "bbox": inst_bbox,
                "color": list(color),
            })

        logger.info("[Task %s] Mode3 discovery: %d instances found", task_id, len(instance_data))

        r.hset(f"task:{task_id}", mapping={
            "status": "instance_ready",
            "progress": 1, "total": 1,
            "message": f"Found {len(instance_data)} instances. Please select one.",
            "instance_masks": json.dumps(instance_data),
        })
        r.expire(f"task:{task_id}", REDIS_TASK_EXPIRE)
        _release_all_models()
        return {"status": "instance_ready", "task_id": task_id}

    except Exception as e:
        logger.exception("[Task %s] Mode3 discovery FAILED: %s", task_id, str(e))
        _handle_gpu_oom(r, task_id, e)
        _release_all_models()
        if self.request.retries < self.max_retries and "out of memory" not in str(e).lower():
            raise self.retry(exc=e)
        return {"status": "failed", "task_id": task_id, "error": str(e)}


@celery_app.task(name="process_instance_annotation", bind=True, max_retries=2, retry_backoff=5)
def process_instance_annotation(
    self,
    ref_image_path: str,
    ref_image_id: str,
    selected_instance_id: int,
    target_image_paths: list[str],
    target_image_ids: list[str],
    task_id: str,
):
    """
    模式3 阶段2：用户选中实例后，跨图批量标注。

    链路：
    1. 加载参考图 + 选中实例的 Mask → DINOv3 提取实例特征模板
    2. 遍历目标图 → DINOv3 余弦匹配 → SAM3 精准 Mask → 绿色透明 PNG
    支持暂停/取消/重试。
    """
    from backend.models.sam_engine import SAMEngine
    from backend.models.dino_engine import DINOEngine
    from backend.services.mask_utils import mask_to_transparent_png, mask_to_bbox, mask_to_polygon
    from backend.services.storage import get_mask_path, get_mask_url, get_export_path, get_export_url

    r = _get_redis()
    total = len(target_image_paths)

    try:
        r.hset(f"task:{task_id}", mapping={
            "status": "processing", "progress": 0, "total": total,
            "message": "Extracting instance feature template...", "mode": "mode3",
        })

        sam = SAMEngine.get_instance()
        dino = DINOEngine.get_instance()

        ref_image = Image.open(ref_image_path).convert("RGB")

        # 重新生成选中实例的 Mask（从聚类结果）
        cluster_map, instance_masks, _, _ = dino.extract_patch_features_clustering(ref_image)

        if selected_instance_id >= len(instance_masks):
            raise ValueError(f"Invalid instance ID: {selected_instance_id}")

        selected_mask = instance_masks[selected_instance_id]

        # 提取实例特征模板
        template_feature = dino.extract_instance_feature(ref_image, selected_mask)
        logger.info("[Task %s] Mode3: instance feature extracted, dim=%d", task_id, len(template_feature))

        # 遍历目标图批量标注
        mask_urls = {}
        coco_annotations = []
        coco_images = []
        annotation_id = 1

        for idx, (img_path, img_id) in enumerate(zip(target_image_paths, target_image_ids)):
            ctrl = _check_task_status(r, task_id)
            if ctrl == "canceled":
                r.hset(f"task:{task_id}", mapping={"status": "canceled", "message": "Task canceled by user."})
                r.expire(f"task:{task_id}", REDIS_TASK_EXPIRE)
                _release_all_models()
                return {"status": "canceled", "task_id": task_id}

            t0 = time.time()
            r.hset(f"task:{task_id}", mapping={
                "progress": idx, "message": f"Processing image {idx + 1}/{total}...",
            })

            target_image = Image.open(img_path).convert("RGB")
            img_w, img_h = target_image.size

            coco_images.append({
                "id": img_id, "file_name": Path(img_path).name,
                "width": img_w, "height": img_h,
            })

            # DINOv3 特征匹配
            match_mask, match_bboxes = dino.match_in_image(
                target_image, template_feature, threshold=COSINE_SIMILARITY_THRESHOLD
            )

            image_mask_urls = []
            for inst_idx, matched_bbox in enumerate(match_bboxes):
                precise_mask = sam.generate_mask(target_image, matched_bbox)
                if precise_mask.sum() < 10:
                    continue

                # 绿色透明 PNG（模式3 视觉区分）
                mask_path = get_mask_path(img_id, inst_idx)
                mask_to_transparent_png(
                    precise_mask, mask_path,
                    color=MASK_MODE3_COLOR, alpha=MASK_MODE3_ALPHA,
                )
                mask_url = get_mask_url(img_id, inst_idx)
                image_mask_urls.append(mask_url)

                coco_annotations.append({
                    "id": annotation_id, "image_id": img_id, "category_id": 1,
                    "bbox": mask_to_bbox(precise_mask),
                    "segmentation": mask_to_polygon(precise_mask),
                    "area": float(precise_mask.sum()), "iscrowd": 0,
                })
                annotation_id += 1

            mask_urls[img_id] = image_mask_urls
            elapsed = time.time() - t0
            logger.info("[Task %s] Image %d/%d done (%.2fs), %d instances",
                        task_id, idx + 1, total, elapsed, len(image_mask_urls))

        # COCO 导出
        coco_result = {
            "images": coco_images,
            "annotations": coco_annotations,
            "categories": [{"id": 1, "name": "instance_target", "supercategory": "object"}],
        }
        export_path = get_export_path(task_id)
        export_path.write_text(json.dumps(coco_result, ensure_ascii=False, indent=2))

        total_annotations = annotation_id - 1
        r.hset(f"task:{task_id}", mapping={
            "status": "success", "progress": total, "total": total,
            "message": f"Completed. {total_annotations} instances found across {total} images.",
            "mask_urls": json.dumps(mask_urls),
            "export_url": get_export_url(task_id),
        })
        r.expire(f"task:{task_id}", REDIS_TASK_EXPIRE)
        _release_all_models()
        return {"status": "success", "task_id": task_id}

    except Exception as e:
        logger.exception("[Task %s] Mode3 FAILED: %s", task_id, str(e))
        _handle_gpu_oom(r, task_id, e)
        _release_all_models()
        if self.request.retries < self.max_retries and "out of memory" not in str(e).lower():
            raise self.retry(exc=e)
        return {"status": "failed", "task_id": task_id, "error": str(e)}
