"""
API 路由模块
实现核心接口：
    POST /upload                — 上传图像
    GET  /images                — 获取图片列表
    DELETE /images/{id}         — 删除图片
    POST /annotate/mode1        — 模式1：文本提示标注
    POST /annotate/mode2        — 模式2：框选批量标注
    POST /annotate/mode3        — 模式3 阶段1：粗分割实例生成
    POST /annotate/mode3/select — 模式3 阶段2：选中实例跨图标注
    GET  /tasks/{task_id}       — 查询任务状态
    POST /tasks/{task_id}/pause — 暂停任务
    POST /tasks/{task_id}/resume — 恢复任务
    POST /tasks/{task_id}/cancel — 取消任务
    GET  /export/{task_id}/{fmt} — 多格式导出（coco/voc/yolo）
"""
import json
import uuid
import logging
from typing import Optional

import redis
from fastapi import APIRouter, UploadFile, File, HTTPException
from pydantic import BaseModel, Field

from backend.core.config import REDIS_URL, IMAGE_DIR, REDIS_TASK_EXPIRE
from backend.core.exceptions import TaskNotFoundError, FileUploadError
from backend.services.storage import save_upload_file, get_image_url

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api", tags=["标注"])


def _get_redis():
    """获取 Redis 连接"""
    return redis.Redis.from_url(REDIS_URL, decode_responses=True)


# ==================== 请求/响应模型 ====================

class UploadResponse(BaseModel):
    """上传响应"""
    image_id: str
    filename: str
    url: str
    path: str


class Mode1Request(BaseModel):
    """模式1 文本标注请求"""
    text_prompt: str = Field(..., description="文本提示，如 'person, car, dog'")
    image_ids: list[str] = Field(..., description="待标注图像 ID 列表")
    image_paths: list[str] = Field(..., description="待标注图像路径列表")


class Mode2Request(BaseModel):
    """模式2 标注请求"""
    ref_image_id: str = Field(..., description="首图（参考图）ID")
    ref_image_path: str = Field(..., description="首图本地路径")
    bbox: list[float] = Field(..., description="框选坐标 [x1, y1, x2, y2]")
    target_images: list[dict] = Field(..., description="待标注图像列表 [{id, path}]")


class Mode3DiscoveryRequest(BaseModel):
    """模式3 阶段1：粗分割实例生成请求"""
    ref_image_id: str = Field(..., description="参考图 ID")
    ref_image_path: str = Field(..., description="参考图本地路径")


class Mode3SelectRequest(BaseModel):
    """模式3 阶段2：选中实例跨图标注请求"""
    discovery_task_id: str = Field(..., description="阶段1 的 task_id")
    ref_image_id: str = Field(..., description="参考图 ID")
    ref_image_path: str = Field(..., description="参考图本地路径")
    selected_instance_id: int = Field(..., description="用户选中的实例 ID")
    target_images: list[dict] = Field(..., description="待标注图像列表 [{id, path}]")


class TaskStatusResponse(BaseModel):
    """任务状态响应"""
    task_id: str
    status: str
    progress: int = 0
    total: int = 0
    message: str = ""
    mode: Optional[str] = None
    mask_urls: Optional[dict] = None
    export_url: Optional[str] = None
    instance_masks: Optional[list] = None  # 模式3 阶段1 实例列表
    error_type: Optional[str] = None       # gpu_oom 等


# ==================== 接口实现 ====================

@router.post("/upload", response_model=UploadResponse)
async def upload_image(file: UploadFile = File(...)):
    """上传图像文件，存入 /data/images/，返回 id+url+path。"""
    try:
        image_id, original_name, saved_path = await save_upload_file(file)
        ext = saved_path.suffix
        url = get_image_url(image_id, ext)
        logger.info("Uploaded: %s -> %s (id=%s)", original_name, saved_path, image_id)
        return UploadResponse(image_id=image_id, filename=original_name, url=url, path=str(saved_path))
    except ValueError as e:
        raise FileUploadError(file.filename or "unknown", str(e))
    except Exception as e:
        logger.exception("Upload failed: %s", str(e))
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")


@router.get("/images")
async def list_images():
    """获取已上传的图片列表。"""
    images = []
    if IMAGE_DIR.exists():
        for f in sorted(IMAGE_DIR.iterdir()):
            if f.suffix.lower() in {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}:
                image_id = f.stem
                images.append({
                    "image_id": image_id, "filename": f.name,
                    "url": get_image_url(image_id, f.suffix), "path": str(f),
                })
    return {"images": images, "count": len(images)}


@router.delete("/images/{image_id}")
async def delete_image(image_id: str):
    """删除指定图片及其关联的 Mask 文件。"""
    from backend.core.config import MASK_DIR

    deleted = False
    for f in IMAGE_DIR.iterdir():
        if f.stem == image_id:
            f.unlink()
            deleted = True
            break

    if not deleted:
        raise HTTPException(status_code=404, detail=f"Image not found: {image_id}")

    if MASK_DIR.exists():
        for mask_file in MASK_DIR.iterdir():
            if mask_file.name.startswith(image_id):
                mask_file.unlink()

    logger.info("Deleted image: %s", image_id)
    return {"message": "ok", "image_id": image_id}


# ==================== 模式1：文本提示标注 ====================

@router.post("/annotate/mode1")
async def annotate_mode1(req: Mode1Request):
    """模式1：文本 → YOLO-World → SAM3 → 蓝色 Mask PNG"""
    text = req.text_prompt.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Text prompt cannot be empty")
    if len(req.image_ids) == 0:
        raise HTTPException(status_code=400, detail="No images provided")

    task_id = uuid.uuid4().hex
    r = _get_redis()
    r.hset(f"task:{task_id}", mapping={
        "status": "pending", "progress": 0, "total": len(req.image_ids),
        "message": "Task queued...", "mode": "mode1",
    })
    r.expire(f"task:{task_id}", REDIS_TASK_EXPIRE)

    from backend.worker import process_text_annotation
    process_text_annotation.delay(
        image_paths=req.image_paths, image_ids=req.image_ids,
        text_prompt=text, task_id=task_id,
    )
    logger.info("Mode1 task: %s (prompt='%s', images=%d)", task_id, text, len(req.image_ids))
    return {"task_id": task_id, "status": "pending", "mode": "mode1"}


# ==================== 模式2：框选批量标注 ====================

@router.post("/annotate/mode2")
async def annotate_mode2(req: Mode2Request):
    """模式2：框选 → SAM3 → DINOv3 匹配 → 批量 SAM3 → 红色 Mask PNG"""
    task_id = uuid.uuid4().hex
    r = _get_redis()
    r.hset(f"task:{task_id}", mapping={
        "status": "pending", "progress": 0, "total": len(req.target_images),
        "message": "Task queued...", "mode": "mode2",
    })
    r.expire(f"task:{task_id}", REDIS_TASK_EXPIRE)

    target_paths = [img["path"] for img in req.target_images]
    target_ids = [img["id"] for img in req.target_images]

    from backend.worker import process_batch_annotation
    process_batch_annotation.delay(
        ref_image_path=req.ref_image_path, bbox=req.bbox,
        target_image_paths=target_paths, target_image_ids=target_ids,
        task_id=task_id,
    )
    logger.info("Mode2 task: %s (ref=%s, targets=%d)", task_id, req.ref_image_id, len(req.target_images))
    return {"task_id": task_id, "status": "pending", "mode": "mode2"}


# ==================== 模式3：选实例跨图标注 ====================

@router.post("/annotate/mode3")
async def annotate_mode3_discovery(req: Mode3DiscoveryRequest):
    """
    模式3 阶段1：粗分割实例生成。
    DINOv3 聚类 → SAM3 粗分割 → 返回实例 Mask URL 列表供用户选择。
    """
    task_id = uuid.uuid4().hex
    r = _get_redis()
    r.hset(f"task:{task_id}", mapping={
        "status": "pending", "progress": 0, "total": 1,
        "message": "Generating instance proposals...", "mode": "mode3_discovery",
    })
    r.expire(f"task:{task_id}", REDIS_TASK_EXPIRE)

    from backend.worker import process_instance_discovery
    process_instance_discovery.delay(
        ref_image_path=req.ref_image_path,
        ref_image_id=req.ref_image_id,
        task_id=task_id,
    )
    logger.info("Mode3 discovery task: %s (ref=%s)", task_id, req.ref_image_id)
    return {"task_id": task_id, "status": "pending", "mode": "mode3_discovery"}


@router.post("/annotate/mode3/select")
async def annotate_mode3_select(req: Mode3SelectRequest):
    """
    模式3 阶段2：用户选中实例后，跨图批量标注。
    DINOv3 实例特征 → 遍历目标图匹配 → SAM3 精准 Mask → 绿色 PNG
    """
    if len(req.target_images) == 0:
        raise HTTPException(status_code=400, detail="No target images provided")

    task_id = uuid.uuid4().hex
    r = _get_redis()
    r.hset(f"task:{task_id}", mapping={
        "status": "pending", "progress": 0, "total": len(req.target_images),
        "message": "Task queued...", "mode": "mode3",
    })
    r.expire(f"task:{task_id}", REDIS_TASK_EXPIRE)

    target_paths = [img["path"] for img in req.target_images]
    target_ids = [img["id"] for img in req.target_images]

    from backend.worker import process_instance_annotation
    process_instance_annotation.delay(
        ref_image_path=req.ref_image_path,
        ref_image_id=req.ref_image_id,
        selected_instance_id=req.selected_instance_id,
        target_image_paths=target_paths,
        target_image_ids=target_ids,
        task_id=task_id,
    )
    logger.info("Mode3 select task: %s (instance=%d, targets=%d)",
                task_id, req.selected_instance_id, len(req.target_images))
    return {"task_id": task_id, "status": "pending", "mode": "mode3"}


# ==================== 任务控制：暂停/恢复/取消 ====================

@router.post("/tasks/{task_id}/pause")
async def pause_task(task_id: str):
    """暂停正在执行的任务。Worker 会在下一个图片处理前检查状态。"""
    r = _get_redis()
    task_data = r.hgetall(f"task:{task_id}")
    if not task_data:
        raise TaskNotFoundError(task_id)

    status = task_data.get("status")
    if status not in ("processing", "pending"):
        raise HTTPException(status_code=400, detail=f"Cannot pause task in '{status}' state")

    r.hset(f"task:{task_id}", "status", "paused")
    logger.info("Task paused: %s", task_id)
    return {"task_id": task_id, "status": "paused"}


@router.post("/tasks/{task_id}/resume")
async def resume_task(task_id: str):
    """恢复暂停的任务。"""
    r = _get_redis()
    task_data = r.hgetall(f"task:{task_id}")
    if not task_data:
        raise TaskNotFoundError(task_id)

    if task_data.get("status") != "paused":
        raise HTTPException(status_code=400, detail="Task is not paused")

    r.hset(f"task:{task_id}", "status", "processing")
    logger.info("Task resumed: %s", task_id)
    return {"task_id": task_id, "status": "processing"}


@router.post("/tasks/{task_id}/cancel")
async def cancel_task(task_id: str):
    """取消任务。Worker 会在下一个图片处理前检查状态并终止。"""
    r = _get_redis()
    task_data = r.hgetall(f"task:{task_id}")
    if not task_data:
        raise TaskNotFoundError(task_id)

    status = task_data.get("status")
    if status in ("success", "canceled"):
        raise HTTPException(status_code=400, detail=f"Cannot cancel task in '{status}' state")

    r.hset(f"task:{task_id}", mapping={"status": "canceled", "message": "Task canceled by user."})
    logger.info("Task canceled: %s", task_id)
    return {"task_id": task_id, "status": "canceled"}


# ==================== 任务状态查询 ====================

@router.get("/tasks/{task_id}", response_model=TaskStatusResponse)
async def get_task_status(task_id: str):
    """查询任务状态、进度、Mask URL、实例列表等。"""
    r = _get_redis()
    task_data = r.hgetall(f"task:{task_id}")

    if not task_data:
        raise TaskNotFoundError(task_id)

    # 解析 JSON 字段
    mask_urls = None
    if "mask_urls" in task_data:
        try:
            mask_urls = json.loads(task_data["mask_urls"])
        except json.JSONDecodeError:
            mask_urls = None

    instance_masks = None
    if "instance_masks" in task_data:
        try:
            instance_masks = json.loads(task_data["instance_masks"])
        except json.JSONDecodeError:
            instance_masks = None

    return TaskStatusResponse(
        task_id=task_id,
        status=task_data.get("status", "unknown"),
        progress=int(task_data.get("progress", 0)),
        total=int(task_data.get("total", 0)),
        message=task_data.get("message", ""),
        mode=task_data.get("mode"),
        mask_urls=mask_urls,
        export_url=task_data.get("export_url"),
        instance_masks=instance_masks,
        error_type=task_data.get("error_type"),
    )


# ==================== 多格式导出 ====================

@router.get("/export/{task_id}/{fmt}")
async def export_annotations(task_id: str, fmt: str):
    """
    多格式导出标注结果。

    Args:
        task_id: 任务 ID
        fmt: 导出格式 (coco / voc / yolo)

    Returns:
        COCO: 直接返回 JSON
        VOC: 返回 XML 字符串列表
        YOLO: 返回 txt 字符串列表
    """
    from backend.core.config import EXPORT_DIR
    from backend.services.mask_utils import generate_voc_annotation, generate_yolo_annotation

    export_path = EXPORT_DIR / f"{task_id}_coco.json"
    if not export_path.exists():
        raise HTTPException(status_code=404, detail=f"Export not found for task: {task_id}")

    coco_data = json.loads(export_path.read_text())

    if fmt == "coco":
        return coco_data

    elif fmt == "voc":
        # 将 COCO 转为 VOC XML
        # 按 image_id 分组
        img_map = {img["id"]: img for img in coco_data.get("images", [])}
        cat_map = {cat["id"]: cat["name"] for cat in coco_data.get("categories", [])}

        voc_results = {}
        for ann in coco_data.get("annotations", []):
            img_id = ann["image_id"]
            img_info = img_map.get(img_id, {})
            if img_id not in voc_results:
                voc_results[img_id] = {
                    "filename": img_info.get("file_name", ""),
                    "width": img_info.get("width", 0),
                    "height": img_info.get("height", 0),
                    "objects": [],
                }
            # COCO bbox [x,y,w,h] → VOC [x1,y1,x2,y2]
            bx, by, bw, bh = ann["bbox"]
            voc_results[img_id]["objects"].append({
                "name": cat_map.get(ann["category_id"], "object"),
                "bbox": [bx, by, bx + bw, by + bh],
            })

        voc_xmls = {}
        for img_id, data in voc_results.items():
            voc_xmls[img_id] = generate_voc_annotation(
                data["filename"], data["width"], data["height"], data["objects"]
            )
        return {"format": "voc", "annotations": voc_xmls}

    elif fmt == "yolo":
        img_map = {img["id"]: img for img in coco_data.get("images", [])}

        yolo_results = {}
        for ann in coco_data.get("annotations", []):
            img_id = ann["image_id"]
            img_info = img_map.get(img_id, {})
            if img_id not in yolo_results:
                yolo_results[img_id] = {
                    "width": img_info.get("width", 1),
                    "height": img_info.get("height", 1),
                    "objects": [],
                }
            bx, by, bw, bh = ann["bbox"]
            yolo_results[img_id]["objects"].append({
                "class_id": ann["category_id"] - 1,
                "bbox": [bx, by, bx + bw, by + bh],
            })

        yolo_txts = {}
        for img_id, data in yolo_results.items():
            yolo_txts[img_id] = generate_yolo_annotation(
                data["width"], data["height"], data["objects"]
            )
        return {"format": "yolo", "annotations": yolo_txts}

    else:
        raise HTTPException(status_code=400, detail=f"Unsupported format: {fmt}. Use coco/voc/yolo.")
