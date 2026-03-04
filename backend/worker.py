"""
Celery Worker 入口（v3 增强版）

v3 增强：
    - 多参考图加权融合（2~5张参考图，自定义权重）
    - 置信度评分集成（标注完成后自动计算每图0~100分）
    - match_in_image 返回相似度值用于评分

启动命令：
    celery -A backend.worker worker --concurrency=1 --pool=solo -l info
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
        r.hset(f"task:{task_id}", mapping={"status": "failed", "message": msg})
    r.expire(f"task:{task_id}", REDIS_TASK_EXPIRE)


def preprocess_image(image: Image.Image) -> Image.Image:
    """图像预处理：高斯去噪 + CLAHE 对比度增强"""
    img_np = np.array(image)
    if PREPROCESS_DENOISE:
        img_np = cv2.GaussianBlur(img_np, (3, 3), 0)
    if PREPROCESS_CLAHE:
        lab = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB)
        clahe = cv2.createCLAHE(clipLimit=PREPROCESS_CLAHE_CLIP,
                                tileGridSize=(PREPROCESS_CLAHE_GRID, PREPROCESS_CLAHE_GRID))
        lab[:, :, 0] = clahe.apply(lab[:, :, 0])
        img_np = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    return Image.fromarray(img_np)


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
    logger.info("[Worker] All models warmed up. Ready.")


# ==================== 模式1 文本提示标注 ====================
@celery_app.task(name="process_text_annotation", bind=True, max_retries=2, retry_backoff=5)
def process_text_annotation(self, image_paths, image_ids, text_prompt, task_id,
                            category_name=None, category_color=None):
    from backend.models.sam_engine import SAMEngine
    from backend.models.grounding_dino_engine import GroundingDINOEngine
    from backend.services.mask_utils import mask_to_transparent_png, mask_to_bbox, mask_to_polygon
    from backend.services.storage import get_mask_path, get_mask_url, get_export_path, get_export_url
    from backend.core.config import GROUNDING_DINO_SCORE_THR_LOW
    from backend.utils.score_calculator import compute_image_score, compute_mask_coverage

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

        mask_urls = {}
        coco_annotations = []
        coco_images = []
        image_scores = {}
        annotation_id = 1
        coco_categories = [{"id": i + 1, "name": cat, "supercategory": "object"} for i, cat in enumerate(categories)]

        for idx, (img_path, img_id) in enumerate(zip(image_paths, image_ids)):
            ctrl = _check_task_status(r, task_id)
            if ctrl == "canceled":
                r.hset(f"task:{task_id}", mapping={"status": "canceled", "message": "Task canceled."})
                r.expire(f"task:{task_id}", REDIS_TASK_EXPIRE)
                _release_all_models()
                return {"status": "canceled", "task_id": task_id}

            t0 = time.time()
            r.hset(f"task:{task_id}", mapping={"progress": idx, "message": f"Processing {idx+1}/{total}..."})

            target_image = Image.open(img_path).convert("RGB")
            img_w, img_h = target_image.size
            coco_images.append({"id": img_id, "file_name": Path(img_path).name, "width": img_w, "height": img_h})

            detections = gd.detect(target_image, categories)
            if len(detections) == 0:
                detections = gd.detect(target_image, categories, score_thr=GROUNDING_DINO_SCORE_THR_LOW)

            image_mask_urls = []
            inst_idx = 0
            sim_vals, cov_vals, area_vals = [], [], []

            for det in detections:
                precise_mask = sam.generate_mask(target_image, det["box"])
                if precise_mask.sum() < 10:
                    continue
                if category_color:
                    # 全局类别颜色（hex → RGB）
                    hex_c = category_color.lstrip('#')
                    color = tuple(int(hex_c[i:i+2], 16) for i in (0, 2, 4))
                else:
                    color = MODE1_CATEGORY_COLORS[det["label_id"] % len(MODE1_CATEGORY_COLORS)]
                mask_path = get_mask_path(img_id, inst_idx)
                mask_to_transparent_png(precise_mask, mask_path, color=color, alpha=MASK_MODE1_ALPHA)
                image_mask_urls.append(get_mask_url(img_id, inst_idx))

                coco_bbox = mask_to_bbox(precise_mask)
                coco_annotations.append({
                    "id": annotation_id, "image_id": img_id,
                    "category_id": det["label_id"] + 1,
                    "bbox": coco_bbox, "segmentation": mask_to_polygon(precise_mask),
                    "area": float(precise_mask.sum()), "score": det["score"], "iscrowd": 0,
                })
                sim_vals.append(det["score"])
                cov_vals.append(compute_mask_coverage(float(precise_mask.sum()), coco_bbox))
                area_vals.append(float(precise_mask.sum()) / (img_w * img_h))
                annotation_id += 1
                inst_idx += 1

            mask_urls[img_id] = image_mask_urls
            image_scores[img_id] = compute_image_score(sim_vals, cov_vals, area_vals, len(image_mask_urls))
            logger.info("[Task %s] Image %d/%d done (%.2fs), %d masks, score=%.0f",
                        task_id, idx+1, total, time.time()-t0, len(image_mask_urls), image_scores[img_id]["total"])

        coco_result = {"images": coco_images, "annotations": coco_annotations, "categories": coco_categories}
        export_path = get_export_path(task_id)
        export_path.write_text(json.dumps(coco_result, ensure_ascii=False, indent=2))

        r.hset(f"task:{task_id}", mapping={
            "status": "success", "progress": total, "total": total,
            "message": f"Completed. {annotation_id-1} instances.",
            "mask_urls": json.dumps(mask_urls),
            "export_url": get_export_url(task_id),
            "image_scores": json.dumps(image_scores),
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


# ==================== 模式2 框选批量标注（多参考图 + 评分） ====================
@celery_app.task(name="process_batch_annotation", bind=True, max_retries=2, retry_backoff=5)
def process_batch_annotation(
    self, ref_image_path, bbox, target_image_paths, target_image_ids, task_id,
    categories=None, ref_images=None,
):
    """
    模式2 框选批量标注。

    多参考图模式（ref_images 非空时）：
        ref_images = [{"path": str, "bbox": [x1,y1,x2,y2], "weight": float}, ...]
        从多张参考图提取特征 → 加权融合 → 跨图匹配
    """
    from backend.models.sam_engine import SAMEngine
    from backend.models.dino_engine import DINOEngine
    from backend.services.mask_utils import mask_to_transparent_png, mask_to_bbox, mask_to_polygon
    from backend.services.storage import get_mask_path, get_mask_url, get_export_path, get_export_url
    from backend.utils.score_calculator import compute_image_score, compute_mask_coverage

    r = _get_redis()
    total = len(target_image_paths)
    try:
        r.hset(f"task:{task_id}", mapping={
            "status": "processing", "progress": 0, "total": total,
            "message": "Generating reference features...", "mode": "mode2",
        })
        sam = SAMEngine.get_instance()
        dino = DINOEngine.get_instance()

        category_templates = []

        if categories and len(categories) > 0:
            for cat_idx, cat_info in enumerate(categories):
                cat_name = cat_info.get("name", f"category_{cat_idx}")
                cat_bboxes = cat_info.get("bboxes", [])
                cat_ref_images = cat_info.get("ref_images", [])

                if not cat_bboxes and not cat_ref_images:
                    continue

                r.hset(f"task:{task_id}", "message", f"Extracting features for '{cat_name}'...")

                # 多参考图特征融合
                if cat_ref_images and len(cat_ref_images) > 0:
                    ref_feats, ref_weights = [], []
                    total_area = 0
                    for ri in cat_ref_images:
                        ri_image = Image.open(ri["path"]).convert("RGB")
                        ri_pp = preprocess_image(ri_image)
                        ri_mask = sam.generate_mask(ri_pp, ri["bbox"])
                        if ri_mask.sum() < 10:
                            continue
                        feat = dino.extract_mask_feature(ri_pp, ri_mask)
                        ref_feats.append(feat)
                        ref_weights.append(ri.get("weight", 1.0))
                        total_area += ri_mask.sum()
                    if ref_feats:
                        template = dino.fuse_multi_ref_features(ref_feats, ref_weights)
                        ref_area_ratio = float(total_area / len(ref_feats)) / (1000 * 1000)
                    else:
                        continue
                elif len(cat_bboxes) == 1:
                    ref_image = Image.open(ref_image_path).convert("RGB")
                    ref_pp = preprocess_image(ref_image)
                    ref_mask = sam.generate_mask(ref_pp, cat_bboxes[0])
                    if ref_mask.sum() < 10:
                        continue
                    template = dino.extract_mask_feature(ref_pp, ref_mask)
                    ref_area_ratio = float(ref_mask.sum()) / (ref_image.size[0] * ref_image.size[1])
                else:
                    ref_image = Image.open(ref_image_path).convert("RGB")
                    ref_pp = preprocess_image(ref_image)
                    template = dino.extract_multi_bbox_feature(ref_pp, cat_bboxes, sam_engine=sam)
                    ref_area_ratio = 0.05

                color = MULTI_CATEGORY_COLORS[cat_idx % len(MULTI_CATEGORY_COLORS)]
                category_templates.append({
                    "name": cat_name, "template": template, "color": color,
                    "category_id": cat_idx + 1, "area_ratio": ref_area_ratio,
                })
        else:
            # 单类别：支持多参考图或单参考图
            ref_feats, ref_weights = [], []
            ref_area_ratio = 0.05

            if ref_images and len(ref_images) > 0:
                r.hset(f"task:{task_id}", "message",
                       f"Fusing {len(ref_images)} reference images...")
                total_area = 0
                for ri in ref_images:
                    ri_image = Image.open(ri["path"]).convert("RGB")
                    ri_pp = preprocess_image(ri_image)
                    ri_bbox = ri.get("bbox", bbox)
                    # 全图参考：裁剪 bbox 到实际图片尺寸
                    ri_w, ri_h = ri_image.size
                    ri_bbox = [
                        max(0, min(ri_bbox[0], ri_w)),
                        max(0, min(ri_bbox[1], ri_h)),
                        max(0, min(ri_bbox[2], ri_w)),
                        max(0, min(ri_bbox[3], ri_h)),
                    ]
                    ri_mask = sam.generate_mask(ri_pp, ri_bbox)
                    if ri_mask.sum() < 10:
                        continue
                    feat = dino.extract_mask_feature(ri_pp, ri_mask)
                    ref_feats.append(feat)
                    ref_weights.append(ri.get("weight", 1.0))
                    total_area += ri_mask.sum()
                    ref_area_ratio = float(total_area / len(ref_feats)) / (
                        ri_image.size[0] * ri_image.size[1])

                if not ref_feats:
                    raise ValueError("All reference images produced empty masks")
                template = dino.fuse_multi_ref_features(ref_feats, ref_weights)
            else:
                ref_image = Image.open(ref_image_path).convert("RGB")
                ref_pp = preprocess_image(ref_image)
                ref_mask = sam.generate_mask(ref_pp, bbox)
                if ref_mask.sum() < 10:
                    raise ValueError("Reference mask is too small")
                template = dino.extract_mask_feature(ref_pp, ref_mask)
                ref_area_ratio = float(ref_mask.sum()) / (ref_image.size[0] * ref_image.size[1])

            category_templates.append({
                "name": "target", "template": template, "color": (255, 80, 80),
                "category_id": 1, "area_ratio": ref_area_ratio,
            })

        if not category_templates:
            raise ValueError("No valid category references found")

        mask_urls = {}
        coco_annotations = []
        coco_images = []
        image_scores = {}
        annotation_id = 1

        for idx, (img_path, img_id) in enumerate(zip(target_image_paths, target_image_ids)):
            ctrl = _check_task_status(r, task_id)
            if ctrl == "canceled":
                r.hset(f"task:{task_id}", mapping={"status": "canceled", "message": "Task canceled."})
                r.expire(f"task:{task_id}", REDIS_TASK_EXPIRE)
                _release_all_models()
                return {"status": "canceled", "task_id": task_id}

            t0 = time.time()
            r.hset(f"task:{task_id}", mapping={"progress": idx, "message": f"Processing {idx+1}/{total}..."})

            target_image = Image.open(img_path).convert("RGB")
            target_pp = preprocess_image(target_image)
            img_w, img_h = target_image.size
            coco_images.append({"id": img_id, "file_name": Path(img_path).name, "width": img_w, "height": img_h})

            image_mask_urls = []
            inst_idx = 0
            sim_vals, cov_vals, area_vals = [], [], []

            for cat in category_templates:
                result = dino.match_in_image(
                    target_pp, cat["template"],
                    ref_area_ratio=cat["area_ratio"],
                    return_similarities=True,
                )
                _, match_bboxes, match_sims = result

                for bi, matched_bbox in enumerate(match_bboxes):
                    precise_mask = sam.generate_mask(target_image, matched_bbox)
                    if precise_mask.sum() < 10:
                        continue
                    mask_path = get_mask_path(img_id, inst_idx)
                    mask_to_transparent_png(precise_mask, mask_path, color=cat["color"], alpha=140)
                    image_mask_urls.append(get_mask_url(img_id, inst_idx))

                    coco_bbox = mask_to_bbox(precise_mask)
                    coco_annotations.append({
                        "id": annotation_id, "image_id": img_id,
                        "category_id": cat["category_id"], "category_name": cat["name"],
                        "category_rgb": list(cat["color"]),
                        "bbox": coco_bbox, "segmentation": mask_to_polygon(precise_mask),
                        "area": float(precise_mask.sum()), "iscrowd": 0,
                    })
                    if bi < len(match_sims):
                        sim_vals.append(match_sims[bi])
                    cov_vals.append(compute_mask_coverage(float(precise_mask.sum()), coco_bbox))
                    area_vals.append(float(precise_mask.sum()) / (img_w * img_h))
                    annotation_id += 1
                    inst_idx += 1

            mask_urls[img_id] = image_mask_urls
            image_scores[img_id] = compute_image_score(sim_vals, cov_vals, area_vals, len(image_mask_urls))
            logger.info("[Task %s] Image %d/%d done (%.2fs), %d inst, score=%.0f",
                        task_id, idx+1, total, time.time()-t0, len(image_mask_urls), image_scores[img_id]["total"])

        coco_cats = [{"id": ct["category_id"], "name": ct["name"], "supercategory": "object"} for ct in category_templates]
        coco_result = {"images": coco_images, "annotations": coco_annotations, "categories": coco_cats}
        export_path = get_export_path(task_id)
        export_path.write_text(json.dumps(coco_result, ensure_ascii=False, indent=2))

        r.hset(f"task:{task_id}", mapping={
            "status": "success", "progress": total, "total": total,
            "message": f"Completed. {annotation_id-1} instances.",
            "mask_urls": json.dumps(mask_urls),
            "export_url": get_export_url(task_id),
            "image_scores": json.dumps(image_scores),
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


# ==================== 模式3 阶段1：粗分割实例生成 ====================
@celery_app.task(name="process_instance_discovery", bind=True, max_retries=2, retry_backoff=5)
def process_instance_discovery(self, ref_image_path, ref_image_id, task_id):
    from backend.models.sam_engine import SAMEngine
    from backend.models.dino_engine import DINOEngine
    from backend.services.mask_utils import mask_to_transparent_png
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
        ref_pp = preprocess_image(ref_image)

        cluster_map, instance_masks, h, w = dino.extract_patch_features_clustering(ref_pp, use_elbow=True)
        if len(instance_masks) == 0:
            r.hset(f"task:{task_id}", mapping={"status": "failed", "message": "参考图未检测到可分割实例"})
            r.expire(f"task:{task_id}", REDIS_TASK_EXPIRE)
            return {"status": "failed", "task_id": task_id}

        instance_colors = [
            (255,80,80),(80,200,80),(80,120,255),(255,200,0),
            (200,80,255),(0,220,180),(255,120,0),(180,180,0),(100,150,255),(255,100,100),
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
            instance_data.append({
                "id": i, "mask_url": get_mask_url(f"{ref_image_id}_inst", i),
                "bbox": inst_bbox, "color": list(color),
            })

        r.hset(f"task:{task_id}", mapping={
            "status": "instance_ready", "progress": 1, "total": 1,
            "message": f"Found {len(instance_data)} instances.",
            "instance_masks": json.dumps(instance_data), "ref_image_id": ref_image_id,
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


# ==================== 模式3 阶段2：选实例跨图标注（多参考图 + 评分） ====================
@celery_app.task(name="process_instance_annotation", bind=True, max_retries=2, retry_backoff=5)
def process_instance_annotation(
    self, ref_image_path, ref_image_id, selected_instance_id,
    target_image_paths, target_image_ids, task_id,
    categories=None, ref_images=None,
):
    from backend.models.sam_engine import SAMEngine
    from backend.models.dino_engine import DINOEngine
    from backend.services.mask_utils import mask_to_transparent_png, mask_to_bbox, mask_to_polygon
    from backend.services.storage import get_mask_path, get_mask_url, get_export_path, get_export_url
    from backend.utils.score_calculator import compute_image_score, compute_mask_coverage

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
        ref_pp = preprocess_image(ref_image)
        cluster_map, instance_masks, _, _ = dino.extract_patch_features_clustering(ref_pp, use_elbow=True)

        category_templates = []

        if categories and len(categories) > 0:
            for cat_idx, cat_info in enumerate(categories):
                cat_name = cat_info.get("name", f"category_{cat_idx}")
                instance_ids = cat_info.get("instance_ids", [])
                if not instance_ids:
                    continue
                selected_masks = [instance_masks[iid] for iid in instance_ids if iid < len(instance_masks)]
                if not selected_masks:
                    continue

                if len(selected_masks) == 1:
                    template = dino.extract_instance_feature(ref_pp, selected_masks[0])
                else:
                    template = dino.extract_multi_instance_feature(ref_pp, selected_masks)

                # 多参考图辅助特征融合
                cat_ref_images = cat_info.get("ref_images", [])
                if cat_ref_images:
                    aux_feats = [template]
                    aux_weights = [1.0]
                    for ri in cat_ref_images:
                        ri_img = Image.open(ri["path"]).convert("RGB")
                        ri_pp = preprocess_image(ri_img)
                        ri_mask = sam.generate_mask(ri_pp, ri["bbox"])
                        if ri_mask.sum() < 10:
                            continue
                        aux_feats.append(dino.extract_mask_feature(ri_pp, ri_mask))
                        aux_weights.append(ri.get("weight", 0.8))
                    template = dino.fuse_multi_ref_features(aux_feats, aux_weights)

                total_area = sum(m.sum() for m in selected_masks)
                ref_area_ratio = float(total_area / len(selected_masks)) / (ref_image.size[0] * ref_image.size[1])
                color = MULTI_CATEGORY_COLORS[cat_idx % len(MULTI_CATEGORY_COLORS)]
                category_templates.append({
                    "name": cat_name, "template": template, "color": color,
                    "category_id": cat_idx + 1, "area_ratio": ref_area_ratio,
                })
        else:
            if selected_instance_id >= len(instance_masks):
                raise ValueError(f"Invalid instance ID: {selected_instance_id}")

            selected_mask = instance_masks[selected_instance_id]
            template = dino.extract_instance_feature(ref_pp, selected_mask)

            # 多参考图辅助融合
            if ref_images and len(ref_images) > 0:
                aux_feats = [template]
                aux_weights = [1.0]
                for ri in ref_images:
                    ri_img = Image.open(ri["path"]).convert("RGB")
                    ri_pp = preprocess_image(ri_img)
                    ri_mask = sam.generate_mask(ri_pp, ri["bbox"])
                    if ri_mask.sum() < 10:
                        continue
                    aux_feats.append(dino.extract_mask_feature(ri_pp, ri_mask))
                    aux_weights.append(ri.get("weight", 0.8))
                template = dino.fuse_multi_ref_features(aux_feats, aux_weights)

            ref_area_ratio = float(selected_mask.sum()) / (ref_image.size[0] * ref_image.size[1])
            category_templates.append({
                "name": "instance_target", "template": template, "color": MASK_MODE3_COLOR,
                "category_id": 1, "area_ratio": ref_area_ratio,
            })

        if not category_templates:
            raise ValueError("No valid category templates")

        mask_urls = {}
        coco_annotations = []
        coco_images = []
        image_scores = {}
        annotation_id = 1

        for idx, (img_path, img_id) in enumerate(zip(target_image_paths, target_image_ids)):
            ctrl = _check_task_status(r, task_id)
            if ctrl == "canceled":
                r.hset(f"task:{task_id}", mapping={"status": "canceled", "message": "Task canceled."})
                r.expire(f"task:{task_id}", REDIS_TASK_EXPIRE)
                _release_all_models()
                return {"status": "canceled", "task_id": task_id}

            t0 = time.time()
            r.hset(f"task:{task_id}", mapping={"progress": idx, "message": f"Processing {idx+1}/{total}..."})

            target_image = Image.open(img_path).convert("RGB")
            target_pp = preprocess_image(target_image)
            img_w, img_h = target_image.size
            coco_images.append({"id": img_id, "file_name": Path(img_path).name, "width": img_w, "height": img_h})

            image_mask_urls = []
            inst_idx = 0
            sim_vals, cov_vals, area_vals = [], [], []

            for cat in category_templates:
                result = dino.match_in_image(
                    target_pp, cat["template"], ref_area_ratio=cat["area_ratio"], return_similarities=True,
                )
                _, match_bboxes, match_sims = result

                for bi, matched_bbox in enumerate(match_bboxes):
                    precise_mask = sam.generate_mask(target_image, matched_bbox)
                    if precise_mask.sum() < 10:
                        continue
                    mask_path = get_mask_path(img_id, inst_idx)
                    mask_to_transparent_png(precise_mask, mask_path, color=cat["color"], alpha=MASK_MODE3_ALPHA)
                    image_mask_urls.append(get_mask_url(img_id, inst_idx))

                    coco_bbox = mask_to_bbox(precise_mask)
                    coco_annotations.append({
                        "id": annotation_id, "image_id": img_id,
                        "category_id": cat["category_id"], "category_name": cat["name"],
                        "category_rgb": list(cat["color"]),
                        "bbox": coco_bbox, "segmentation": mask_to_polygon(precise_mask),
                        "area": float(precise_mask.sum()), "iscrowd": 0,
                    })
                    if bi < len(match_sims):
                        sim_vals.append(match_sims[bi])
                    cov_vals.append(compute_mask_coverage(float(precise_mask.sum()), coco_bbox))
                    area_vals.append(float(precise_mask.sum()) / (img_w * img_h))
                    annotation_id += 1
                    inst_idx += 1

            mask_urls[img_id] = image_mask_urls
            image_scores[img_id] = compute_image_score(sim_vals, cov_vals, area_vals, len(image_mask_urls))
            logger.info("[Task %s] Image %d/%d done (%.2fs), %d inst, score=%.0f",
                        task_id, idx+1, total, time.time()-t0, len(image_mask_urls), image_scores[img_id]["total"])

        coco_cats = [{"id": ct["category_id"], "name": ct["name"], "supercategory": "object"} for ct in category_templates]
        coco_result = {"images": coco_images, "annotations": coco_annotations, "categories": coco_cats}
        export_path = get_export_path(task_id)
        export_path.write_text(json.dumps(coco_result, ensure_ascii=False, indent=2))

        r.hset(f"task:{task_id}", mapping={
            "status": "success", "progress": total, "total": total,
            "message": f"Completed. {annotation_id-1} instances.",
            "mask_urls": json.dumps(mask_urls),
            "export_url": get_export_url(task_id),
            "image_scores": json.dumps(image_scores),
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


# ==================== 手动标注：单框触发 SAM3 Mask ====================
@celery_app.task(name="process_manual_sam", bind=True, max_retries=1)
def process_manual_sam(self, image_path, image_id, bbox, task_id,
                       category_color=None, category_name=None):
    """
    手动标注触发 SAM3 生成 Mask。
    支持传入类别颜色，使手动标注 Mask 与自动标注同类别颜色一致。
    """
    from backend.models.sam_engine import SAMEngine
    from backend.services.mask_utils import mask_to_transparent_png
    from backend.services.storage import get_mask_path, get_mask_url

    r = _get_redis()
    try:
        r.hset(f"task:{task_id}", mapping={"status": "processing", "mode": "manual"})
        sam = SAMEngine.get_instance()
        target_image = Image.open(image_path).convert("RGB")
        precise_mask = sam.generate_mask(target_image, bbox)

        if precise_mask.sum() < 10:
            r.hset(f"task:{task_id}", mapping={"status": "failed", "message": "Mask too small"})
            r.expire(f"task:{task_id}", 3600)
            return {"status": "failed", "task_id": task_id}

        # 使用传入的类别颜色，否则使用默认橙色
        if category_color and isinstance(category_color, str) and category_color.startswith('#'):
            hex_c = category_color.lstrip('#')
            color = (int(hex_c[0:2], 16), int(hex_c[2:4], 16), int(hex_c[4:6], 16))
        else:
            color = (255, 165, 0)

        inst_idx = int(time.time() * 1000) % 10000
        mask_path = get_mask_path(image_id, inst_idx)
        mask_to_transparent_png(precise_mask, mask_path, color=color, alpha=140)
        mask_url = get_mask_url(image_id, inst_idx)

        r.hset(f"task:{task_id}", mapping={
            "status": "success", "mask_url": mask_url,
            "message": "Manual mask generated",
        })
        r.expire(f"task:{task_id}", 3600)
        _release_all_models()
        return {"status": "success", "task_id": task_id, "mask_url": mask_url}
    except Exception as e:
        logger.exception("[Manual SAM] FAILED: %s", str(e))
        r.hset(f"task:{task_id}", mapping={"status": "failed", "message": str(e)})
        r.expire(f"task:{task_id}", 3600)
        _release_all_models()
        return {"status": "failed", "task_id": task_id, "error": str(e)}
