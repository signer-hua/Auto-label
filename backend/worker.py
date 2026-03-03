"""
Celery Worker 入口（增强版）
注册异步任务，实现模型预热。

启动命令（强制并发数为 1，避免 GPU OOM）：
    celery -A backend.worker worker --concurrency=1 --pool=solo -l info

核心任务：
    process_text_annotation:     模式1 文本提示标注
    process_batch_annotation:    模式2 框选批量标注（支持多类别 + 多框选参考）
    process_instance_discovery:  模式3 阶段1 粗分割实例生成
    process_instance_annotation: 模式3 阶段2 选实例跨图标注（支持多类别 + 多实例融合）

v2 增强：
    - 图像预处理流水线（高斯去噪 + CLAHE 对比度增强）
    - 动态余弦相似度阈值
    - 多类别并行标注（模式2/3）
    - 多实例特征融合（模式3）
    - 多框选参考特征融合（模式2）
"""
import json
import time
import logging
import numpy as np
import cv2
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
    MULTI_CATEGORY_COLORS,
    PREPROCESS_CLAHE, PREPROCESS_DENOISE,
    PREPROCESS_CLAHE_CLIP, PREPROCESS_CLAHE_GRID,
    COSINE_SIM_AREA_CUTOFF,
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
    import redis
    return redis.Redis.from_url(REDIS_URL, decode_responses=True)


def _check_task_status(r, task_id: str) -> str:
    status = r.hget(f"task:{task_id}", "status")
    if status == "canceled":
        return "canceled"
    if status == "paused":
        logger.info("[Task %s] Paused, waiting for resume...", task_id)
        while status == "paused":
            time.sleep(1)
            status = r.hget(f"task:{task_id}", "status")
            if status == "canceled":
                return "canceled"
        r.hset(f"task:{task_id}", "status", "processing")
    return "processing"


def _release_all_models():
    from backend.models.sam_engine import SAMEngine
    from backend.models.dino_engine import DINOEngine
    from backend.models.grounding_dino_engine import GroundingDINOEngine

    SAMEngine.get_instance().release_memory()
    DINOEngine.get_instance().release_memory()
    GroundingDINOEngine.get_instance().release_memory()


def _handle_gpu_oom(r, task_id: str, e: Exception):
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


# ==================== 图像预处理流水线 ====================
def preprocess_image(image: Image.Image) -> Image.Image:
    """
    图像预处理流水线：高斯去噪 + CLAHE 对比度增强。
    统一在加载图像时执行，提升特征提取在噪声/低对比度场景下的鲁棒性。

    Args:
        image: PIL Image (RGB)
    Returns:
        preprocessed: 预处理后的 PIL Image (RGB)
    """
    img_np = np.array(image)

    if PREPROCESS_DENOISE:
        img_np = cv2.GaussianBlur(img_np, (3, 3), 0)

    if PREPROCESS_CLAHE:
        lab = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB)
        l_channel = lab[:, :, 0]
        clahe = cv2.createCLAHE(
            clipLimit=PREPROCESS_CLAHE_CLIP,
            tileGridSize=(PREPROCESS_CLAHE_GRID, PREPROCESS_CLAHE_GRID),
        )
        lab[:, :, 0] = clahe.apply(l_channel)
        img_np = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

    return Image.fromarray(img_np)


# ==================== Worker 启动时预热模型 ====================
@worker_process_init.connect
def warmup_models(**kwargs):
    logger.info("=" * 60)
    logger.info("[Worker] Warming up all models...")
    logger.info("=" * 60)

    from backend.models.sam_engine import SAMEngine
    from backend.models.dino_engine import DINOEngine
    from backend.models.grounding_dino_engine import GroundingDINOEngine

    SAMEngine.get_instance().warmup()
    DINOEngine.get_instance().warmup()
    GroundingDINOEngine.get_instance().warmup()

    logger.info("[Worker] All models (SAM3 + DINOv3 + Grounding DINO) warmed up. Ready.")


# ==================== 异步任务：模式1 文本提示标注 ====================
@celery_app.task(name="process_text_annotation", bind=True, max_retries=2, retry_backoff=5)
def process_text_annotation(
    self,
    image_paths: list[str],
    image_ids: list[str],
    text_prompt: str,
    task_id: str,
):
    """模式1 文本提示标注：文本 → Grounding DINO 检测 bbox → SAM3 精准 Mask"""
    from backend.models.sam_engine import SAMEngine
    from backend.models.grounding_dino_engine import GroundingDINOEngine
    from backend.services.mask_utils import mask_to_transparent_png, mask_to_bbox, mask_to_polygon
    from backend.services.storage import get_mask_path, get_mask_url, get_export_path, get_export_url
    from backend.core.config import GROUNDING_DINO_SCORE_THR_LOW

    r = _get_redis()
    total = len(image_paths)

    try:
        r.hset(f"task:{task_id}", mapping={
            "status": "processing", "progress": 0, "total": total,
            "message": "Starting text-driven annotation...", "mode": "mode1",
        })

        sam = SAMEngine.get_instance()
        gd = GroundingDINOEngine.get_instance()

        categories = [c.strip() for c in
                      text_prompt.replace('，', ',').replace('、', ',').replace('；', ',').split(',')
                      if c.strip()]
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

            detections = gd.detect(target_image, categories)

            if len(detections) == 0:
                logger.info("[Task %s] No detections at default threshold, retrying with low threshold...", task_id)
                detections = gd.detect(target_image, categories, score_thr=GROUNDING_DINO_SCORE_THR_LOW)

            image_mask_urls = []
            inst_idx = 0

            for det in detections:
                bbox_det = det["box"]
                label_id = det["label_id"]
                score = det["score"]

                precise_mask = sam.generate_mask(target_image, bbox_det)
                if precise_mask.sum() < 10:
                    continue

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


# ==================== 异步任务：模式2 框选批量标注（支持多类别） ====================
@celery_app.task(name="process_batch_annotation", bind=True, max_retries=2, retry_backoff=5)
def process_batch_annotation(
    self,
    ref_image_path: str,
    bbox: list[float],
    target_image_paths: list[str],
    target_image_ids: list[str],
    task_id: str,
    categories: list[dict] | None = None,
):
    """
    模式2 框选批量标注（增强版：支持多类别 + 多框选参考 + 动态阈值 + 图像预处理）。

    单类别模式（向后兼容）：
        bbox 为单一框选坐标 → 单类别标注

    多类别模式（categories 非空时）：
        categories = [{"name": "cat", "bboxes": [[x1,y1,x2,y2], ...]}, ...]
        每个类别独立提取特征 → 独立匹配 → 独立 Mask 生成
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
            "message": "Generating reference features...", "mode": "mode2",
        })

        sam = SAMEngine.get_instance()
        dino = DINOEngine.get_instance()

        ref_image = Image.open(ref_image_path).convert("RGB")
        ref_preprocessed = preprocess_image(ref_image)

        # 构建类别-特征模板列表
        category_templates = []

        if categories and len(categories) > 0:
            # 多类别模式
            for cat_idx, cat_info in enumerate(categories):
                cat_name = cat_info.get("name", f"category_{cat_idx}")
                cat_bboxes = cat_info.get("bboxes", [])

                if not cat_bboxes:
                    continue

                r.hset(f"task:{task_id}", "message",
                       f"Extracting features for '{cat_name}'...")

                if len(cat_bboxes) == 1:
                    ref_mask = sam.generate_mask(ref_preprocessed, cat_bboxes[0])
                    if ref_mask.sum() < 10:
                        logger.warning("[Task %s] Category '%s' reference mask too small", task_id, cat_name)
                        continue
                    template = dino.extract_mask_feature(ref_preprocessed, ref_mask)
                    ref_area_ratio = float(ref_mask.sum()) / (ref_image.size[0] * ref_image.size[1])
                else:
                    template = dino.extract_multi_bbox_feature(
                        ref_preprocessed, cat_bboxes, sam_engine=sam
                    )
                    total_mask_area = 0
                    for b in cat_bboxes:
                        m = sam.generate_mask(ref_preprocessed, b)
                        total_mask_area += m.sum()
                    ref_area_ratio = float(total_mask_area / len(cat_bboxes)) / (
                        ref_image.size[0] * ref_image.size[1]
                    )

                color = MULTI_CATEGORY_COLORS[cat_idx % len(MULTI_CATEGORY_COLORS)]
                category_templates.append({
                    "name": cat_name,
                    "template": template,
                    "color": color,
                    "category_id": cat_idx + 1,
                    "area_ratio": ref_area_ratio,
                })
        else:
            # 单类别模式（向后兼容）
            ref_mask = sam.generate_mask(ref_preprocessed, bbox)
            if ref_mask.sum() < 10:
                raise ValueError("Reference mask is too small. Please adjust the bounding box.")

            r.hset(f"task:{task_id}", "message", "Extracting feature template...")
            template = dino.extract_mask_feature(ref_preprocessed, ref_mask)
            ref_area_ratio = float(ref_mask.sum()) / (ref_image.size[0] * ref_image.size[1])

            category_templates.append({
                "name": "target",
                "template": template,
                "color": (255, 80, 80),
                "category_id": 1,
                "area_ratio": ref_area_ratio,
            })

        if not category_templates:
            raise ValueError("No valid category references found")

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
            target_preprocessed = preprocess_image(target_image)
            img_w, img_h = target_image.size

            coco_images.append({
                "id": img_id, "file_name": Path(img_path).name,
                "width": img_w, "height": img_h,
            })

            image_mask_urls = []
            inst_idx = 0

            for cat in category_templates:
                _, match_bboxes = dino.match_in_image(
                    target_preprocessed, cat["template"],
                    ref_area_ratio=cat["area_ratio"],
                )

                for matched_bbox in match_bboxes:
                    precise_mask = sam.generate_mask(target_image, matched_bbox)
                    if precise_mask.sum() < 10:
                        continue

                    mask_path = get_mask_path(img_id, inst_idx)
                    mask_to_transparent_png(
                        precise_mask, mask_path,
                        color=cat["color"], alpha=140,
                    )
                    mask_url = get_mask_url(img_id, inst_idx)
                    image_mask_urls.append(mask_url)

                    coco_annotations.append({
                        "id": annotation_id, "image_id": img_id,
                        "category_id": cat["category_id"],
                        "category_name": cat["name"],
                        "bbox": mask_to_bbox(precise_mask),
                        "segmentation": mask_to_polygon(precise_mask),
                        "area": float(precise_mask.sum()), "iscrowd": 0,
                    })
                    annotation_id += 1
                    inst_idx += 1

            mask_urls[img_id] = image_mask_urls
            elapsed = time.time() - t0
            logger.info("[Task %s] Image %d/%d done (%.2fs), %d instances",
                        task_id, idx + 1, total, elapsed, len(image_mask_urls))

        # COCO 导出
        coco_cats = [
            {"id": ct["category_id"], "name": ct["name"], "supercategory": "object"}
            for ct in category_templates
        ]
        coco_result = {
            "images": coco_images,
            "annotations": coco_annotations,
            "categories": coco_cats,
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
        logger.exception("[Task %s] Mode2 FAILED: %s", task_id, str(e))
        _handle_gpu_oom(r, task_id, e)
        _release_all_models()
        if self.request.retries < self.max_retries and "out of memory" not in str(e).lower():
            raise self.retry(exc=e)
        return {"status": "failed", "task_id": task_id, "error": str(e)}


# ==================== 异步任务：模式3 粗分割实例生成 ====================
@celery_app.task(name="process_instance_discovery", bind=True, max_retries=2, retry_backoff=5)
def process_instance_discovery(
    self,
    ref_image_path: str,
    ref_image_id: str,
    task_id: str,
):
    """模式3 阶段1：参考图粗分割实例生成（增强版：肘部法则自适应聚类 + 背景过滤）。"""
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
        ref_preprocessed = preprocess_image(ref_image)

        cluster_map, instance_masks, h, w = dino.extract_patch_features_clustering(
            ref_preprocessed, use_elbow=True
        )

        if len(instance_masks) == 0:
            r.hset(f"task:{task_id}", mapping={
                "status": "failed",
                "message": "参考图未检测到可分割实例，请更换图片或调整聚类参数",
            })
            r.expire(f"task:{task_id}", REDIS_TASK_EXPIRE)
            return {"status": "failed", "task_id": task_id}

        instance_colors = [
            (255, 80, 80), (80, 200, 80), (80, 120, 255),
            (255, 200, 0), (200, 80, 255), (0, 220, 180),
            (255, 120, 0), (180, 180, 0), (100, 150, 255),
            (255, 100, 100),
        ]

        instance_data = []
        for i, inst_mask in enumerate(instance_masks):
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
            "message": f"Found {len(instance_data)} instances. Select one or more.",
            "instance_masks": json.dumps(instance_data),
            "ref_image_id": ref_image_id,
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


# ==================== 异步任务：模式3 选实例跨图标注（支持多类别 + 多实例） ====================
@celery_app.task(name="process_instance_annotation", bind=True, max_retries=2, retry_backoff=5)
def process_instance_annotation(
    self,
    ref_image_path: str,
    ref_image_id: str,
    selected_instance_id: int,
    target_image_paths: list[str],
    target_image_ids: list[str],
    task_id: str,
    categories: list[dict] | None = None,
):
    """
    模式3 阶段2：选实例跨图标注（增强版：支持多类别 + 多实例融合）。

    单实例单类别（向后兼容）：
        selected_instance_id → 单一实例 → 单类别标注

    多类别多实例（categories 非空时）：
        categories = [{"name": "cat", "instance_ids": [0, 2]}, {"name": "dog", "instance_ids": [1, 3]}]
        每个类别独立融合实例特征 → 独立匹配 → 独立 Mask
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
            "message": "Extracting instance features...", "mode": "mode3",
        })

        sam = SAMEngine.get_instance()
        dino = DINOEngine.get_instance()

        ref_image = Image.open(ref_image_path).convert("RGB")
        ref_preprocessed = preprocess_image(ref_image)

        cluster_map, instance_masks, _, _ = dino.extract_patch_features_clustering(
            ref_preprocessed, use_elbow=True
        )

        # 构建类别-特征模板列表
        category_templates = []

        if categories and len(categories) > 0:
            for cat_idx, cat_info in enumerate(categories):
                cat_name = cat_info.get("name", f"category_{cat_idx}")
                instance_ids = cat_info.get("instance_ids", [])

                if not instance_ids:
                    continue

                selected_masks = []
                for iid in instance_ids:
                    if iid < len(instance_masks):
                        selected_masks.append(instance_masks[iid])

                if not selected_masks:
                    logger.warning("[Task %s] Category '%s' has no valid instances", task_id, cat_name)
                    continue

                r.hset(f"task:{task_id}", "message",
                       f"Extracting features for '{cat_name}'...")

                if len(selected_masks) == 1:
                    template = dino.extract_instance_feature(ref_preprocessed, selected_masks[0])
                else:
                    template = dino.extract_multi_instance_feature(ref_preprocessed, selected_masks)

                total_area = sum(m.sum() for m in selected_masks)
                ref_area_ratio = float(total_area / len(selected_masks)) / (
                    ref_image.size[0] * ref_image.size[1]
                )

                color = MULTI_CATEGORY_COLORS[cat_idx % len(MULTI_CATEGORY_COLORS)]
                category_templates.append({
                    "name": cat_name,
                    "template": template,
                    "color": color,
                    "category_id": cat_idx + 1,
                    "area_ratio": ref_area_ratio,
                })
        else:
            # 单实例单类别（向后兼容）
            if selected_instance_id >= len(instance_masks):
                raise ValueError(f"Invalid instance ID: {selected_instance_id}")

            selected_mask = instance_masks[selected_instance_id]
            template = dino.extract_instance_feature(ref_preprocessed, selected_mask)
            ref_area_ratio = float(selected_mask.sum()) / (ref_image.size[0] * ref_image.size[1])

            logger.info("[Task %s] Mode3: instance feature extracted, dim=%d", task_id, len(template))

            category_templates.append({
                "name": "instance_target",
                "template": template,
                "color": MASK_MODE3_COLOR,
                "category_id": 1,
                "area_ratio": ref_area_ratio,
            })

        if not category_templates:
            raise ValueError("No valid category templates generated")

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
            target_preprocessed = preprocess_image(target_image)
            img_w, img_h = target_image.size

            coco_images.append({
                "id": img_id, "file_name": Path(img_path).name,
                "width": img_w, "height": img_h,
            })

            image_mask_urls = []
            inst_idx = 0

            for cat in category_templates:
                _, match_bboxes = dino.match_in_image(
                    target_preprocessed, cat["template"],
                    ref_area_ratio=cat["area_ratio"],
                )

                for matched_bbox in match_bboxes:
                    precise_mask = sam.generate_mask(target_image, matched_bbox)
                    if precise_mask.sum() < 10:
                        continue

                    mask_path = get_mask_path(img_id, inst_idx)
                    mask_to_transparent_png(
                        precise_mask, mask_path,
                        color=cat["color"],
                        alpha=MASK_MODE3_ALPHA,
                    )
                    mask_url = get_mask_url(img_id, inst_idx)
                    image_mask_urls.append(mask_url)

                    coco_annotations.append({
                        "id": annotation_id, "image_id": img_id,
                        "category_id": cat["category_id"],
                        "category_name": cat["name"],
                        "bbox": mask_to_bbox(precise_mask),
                        "segmentation": mask_to_polygon(precise_mask),
                        "area": float(precise_mask.sum()), "iscrowd": 0,
                    })
                    annotation_id += 1
                    inst_idx += 1

            mask_urls[img_id] = image_mask_urls
            elapsed = time.time() - t0
            logger.info("[Task %s] Image %d/%d done (%.2fs), %d instances",
                        task_id, idx + 1, total, elapsed, len(image_mask_urls))

        # COCO 导出
        coco_cats = [
            {"id": ct["category_id"], "name": ct["name"], "supercategory": "object"}
            for ct in category_templates
        ]
        coco_result = {
            "images": coco_images,
            "annotations": coco_annotations,
            "categories": coco_cats,
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
