"""
API 路由模块
实现三个核心接口：
    POST /upload          — 上传图像，存本地返回路径 + ID
    POST /annotate/mode2  — 触发模式2异步批量标注
    GET  /tasks/{task_id} — 查询 Redis 中任务状态/进度/Mask URL
"""
import json
import uuid
import logging
from typing import Optional

import redis
from fastapi import APIRouter, UploadFile, File, HTTPException
from pydantic import BaseModel, Field

from backend.core.config import REDIS_URL, IMAGE_DIR
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


class Mode2Request(BaseModel):
    """模式2 标注请求"""
    ref_image_id: str = Field(..., description="首图（参考图）ID")
    ref_image_path: str = Field(..., description="首图本地路径")
    bbox: list[float] = Field(..., description="框选坐标 [x1, y1, x2, y2]")
    target_images: list[dict] = Field(
        ...,
        description="待标注图像列表，每项包含 {id, path}",
    )


class TaskStatusResponse(BaseModel):
    """任务状态响应"""
    task_id: str
    status: str          # pending / processing / success / failed
    progress: int = 0
    total: int = 0
    message: str = ""
    mask_urls: Optional[dict] = None   # {image_id: [mask_url, ...]}
    export_url: Optional[str] = None


# ==================== 接口实现 ====================

@router.post("/upload", response_model=UploadResponse)
async def upload_image(file: UploadFile = File(...)):
    """
    上传图像文件。

    接收 multipart/form-data 格式图片，写入本地 /data/images/，
    返回文件路径 + 唯一 ID + 访问 URL。

    响应时间 ≤ 100ms。
    """
    try:
        image_id, original_name, saved_path = await save_upload_file(file)
        ext = saved_path.suffix
        url = get_image_url(image_id, ext)

        logger.info("Uploaded: %s -> %s (id=%s)", original_name, saved_path, image_id)

        return UploadResponse(
            image_id=image_id,
            filename=original_name,
            url=url,
            path=str(saved_path),
        )
    except ValueError as e:
        raise FileUploadError(file.filename or "unknown", str(e))
    except Exception as e:
        logger.exception("Upload failed: %s", str(e))
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")


@router.get("/images")
async def list_images():
    """
    获取已上传的图片列表。
    扫描 /data/images/ 目录，返回所有图片信息。
    """
    images = []
    if IMAGE_DIR.exists():
        for f in sorted(IMAGE_DIR.iterdir()):
            if f.suffix.lower() in {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}:
                image_id = f.stem
                images.append({
                    "image_id": image_id,
                    "filename": f.name,
                    "url": get_image_url(image_id, f.suffix),
                    "path": str(f),
                })
    return {"images": images, "count": len(images)}


@router.delete("/images/{image_id}")
async def delete_image(image_id: str):
    """
    删除指定图片及其关联的 Mask 文件。
    """
    from backend.core.config import MASK_DIR

    # 查找并删除原图
    deleted = False
    for f in IMAGE_DIR.iterdir():
        if f.stem == image_id:
            f.unlink()
            deleted = True
            break

    if not deleted:
        raise HTTPException(status_code=404, detail=f"Image not found: {image_id}")

    # 删除关联的 Mask 文件
    if MASK_DIR.exists():
        for mask_file in MASK_DIR.iterdir():
            if mask_file.name.startswith(image_id):
                mask_file.unlink()

    logger.info("Deleted image: %s", image_id)
    return {"message": "ok", "image_id": image_id}


@router.post("/annotate/mode2")
async def annotate_mode2(req: Mode2Request):
    """
    模式2：人工预标注 → 批量自动标注。

    接收首图 ID、待标注图列表、bbox 坐标，
    调用 Celery 异步任务 process_batch_annotation.delay()，
    生成 task_id 并写入 Redis（状态 pending），立即返回 task_id。

    响应时间 ≤ 100ms（仅触发异步任务，不执行推理）。
    """
    # 生成任务 ID
    task_id = uuid.uuid4().hex

    # 写入 Redis 初始状态
    r = _get_redis()
    r.hset(f"task:{task_id}", mapping={
        "status": "pending",
        "progress": 0,
        "total": len(req.target_images),
        "message": "Task queued, waiting for worker...",
    })
    r.expire(f"task:{task_id}", 86400)  # 24 小时过期

    # 提取目标图路径和 ID
    target_paths = [img["path"] for img in req.target_images]
    target_ids = [img["id"] for img in req.target_images]

    # 触发 Celery 异步任务
    from backend.worker import process_batch_annotation
    process_batch_annotation.delay(
        ref_image_path=req.ref_image_path,
        bbox=req.bbox,
        target_image_paths=target_paths,
        target_image_ids=target_ids,
        task_id=task_id,
    )

    logger.info("Task created: %s (ref=%s, targets=%d)",
                task_id, req.ref_image_id, len(req.target_images))

    return {"task_id": task_id, "status": "pending"}


@router.get("/tasks/{task_id}", response_model=TaskStatusResponse)
async def get_task_status(task_id: str):
    """
    查询任务状态。

    从 Redis 读取任务状态、进度、Mask URL 列表。
    前端每 1 秒轮询此接口。

    响应时间 ≤ 10ms。
    """
    r = _get_redis()
    task_data = r.hgetall(f"task:{task_id}")

    if not task_data:
        raise TaskNotFoundError(task_id)

    # 解析 mask_urls JSON
    mask_urls = None
    if "mask_urls" in task_data:
        try:
            mask_urls = json.loads(task_data["mask_urls"])
        except json.JSONDecodeError:
            mask_urls = None

    return TaskStatusResponse(
        task_id=task_id,
        status=task_data.get("status", "unknown"),
        progress=int(task_data.get("progress", 0)),
        total=int(task_data.get("total", 0)),
        message=task_data.get("message", ""),
        mask_urls=mask_urls,
        export_url=task_data.get("export_url"),
    )
