"""
标注 API
三种标注模式的触发接口
"""
import json
import numpy as np
from typing import List, Optional
from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
from PIL import Image
from pathlib import Path

from backend.config import UPLOAD_DIR, OUTPUT_DIR, annotation_config
from backend.services.annotation_service import AnnotationService

router = APIRouter(prefix="/api/annotate", tags=["标注"])

# 全局标注服务实例（延迟初始化）
_service: Optional[AnnotationService] = None


def get_service() -> AnnotationService:
    """获取标注服务单例"""
    global _service
    if _service is None:
        _service = AnnotationService()
    return _service


# ==================== 请求/响应模型 ====================

class Mode1Request(BaseModel):
    """模式1请求：文本提示一键标注"""
    image_names: List[str] = Field(..., description="图像文件名列表")
    text_prompts: List[str] = Field(..., description="文本提示列表，如 ['person', 'car']")
    score_thr: float = Field(default=0.3, description="检测置信度阈值")
    export_format: str = Field(default="coco", description="导出格式: coco / voc")


class Mode2Request(BaseModel):
    """模式2请求：人工预标注 → 批量标注"""
    ref_image_name: str = Field(..., description="参考图像文件名")
    user_boxes: List[List[float]] = Field(..., description="用户框选的边界框 [[x1,y1,x2,y2], ...]")
    target_image_names: List[str] = Field(..., description="待标注图像文件名列表")
    similarity_threshold: float = Field(default=0.8, description="特征匹配阈值")
    export_format: str = Field(default="coco", description="导出格式")


class Mode3ClusterRequest(BaseModel):
    """模式3请求（第一步）：全图聚类"""
    image_name: str = Field(..., description="图像文件名")
    n_clusters: int = Field(default=10, description="聚类数")


class Mode3AnnotateRequest(BaseModel):
    """模式3请求（第二步）：跨图标注"""
    selected_feature: List[float] = Field(..., description="选中实例的特征向量")
    target_image_names: List[str] = Field(..., description="待标注图像文件名列表")
    similarity_threshold: float = Field(default=0.8, description="特征匹配阈值")
    export_format: str = Field(default="coco", description="导出格式")


class ThresholdUpdateRequest(BaseModel):
    """更新参数请求"""
    similarity_threshold: Optional[float] = None
    kmeans_clusters: Optional[int] = None
    score_thr: Optional[float] = None


# ==================== API 路由 ====================

@router.post("/mode1", summary="模式1：文本提示一键自动标注")
async def mode1_annotate(req: Mode1Request):
    """
    文本提示一键自动标注

    流程：YOLO-World 检测 → DINOv3 特征增强 → SAM3 精准分割
    """
    service = get_service()

    # 加载图像
    images = []
    for name in req.image_names:
        img_path = UPLOAD_DIR / name
        if not img_path.exists():
            raise HTTPException(status_code=404, detail=f"图像 {name} 不存在")
        images.append((name, Image.open(img_path).convert("RGB")))

    # 执行标注
    results = service.run_mode1(
        images=images,
        text_prompts=req.text_prompts,
        score_thr=req.score_thr,
        export_format=req.export_format,
    )

    return results


@router.post("/mode2", summary="模式2：人工预标注 → 批量自动标注")
async def mode2_annotate(req: Mode2Request):
    """
    人工预标注 → 批量自动标注

    流程：SAM3 生成参考 mask → DINOv3 构建特征模板 → 跨图匹配 → SAM3 精准分割
    """
    service = get_service()

    # 加载参考图像
    ref_path = UPLOAD_DIR / req.ref_image_name
    if not ref_path.exists():
        raise HTTPException(status_code=404, detail=f"参考图像 {req.ref_image_name} 不存在")
    ref_image = Image.open(ref_path).convert("RGB")

    # 加载目标图像
    target_images = []
    for name in req.target_image_names:
        img_path = UPLOAD_DIR / name
        if not img_path.exists():
            raise HTTPException(status_code=404, detail=f"图像 {name} 不存在")
        target_images.append((name, Image.open(img_path).convert("RGB")))

    results = service.run_mode2(
        ref_image=(req.ref_image_name, ref_image),
        user_boxes=req.user_boxes,
        target_images=target_images,
        threshold=req.similarity_threshold,
        export_format=req.export_format,
    )

    return results


@router.post("/mode3/cluster", summary="模式3第一步：全图聚类粗分割")
async def mode3_cluster(req: Mode3ClusterRequest):
    """
    全图聚类粗分割

    流程：DINOv3 提取 patch 特征 → K-Means 聚类 → SAM3 粗分割
    """
    service = get_service()

    img_path = UPLOAD_DIR / req.image_name
    if not img_path.exists():
        raise HTTPException(status_code=404, detail=f"图像 {req.image_name} 不存在")
    image = Image.open(img_path).convert("RGB")

    results = service.run_mode3_cluster(
        image=(req.image_name, image),
        n_clusters=req.n_clusters,
    )

    return results


@router.post("/mode3/annotate", summary="模式3第二步：跨图批量标注")
async def mode3_annotate(req: Mode3AnnotateRequest):
    """
    选中实例后跨图批量标注

    流程：DINOv3 特征匹配 → SAM3 精准分割
    """
    service = get_service()

    target_images = []
    for name in req.target_image_names:
        img_path = UPLOAD_DIR / name
        if not img_path.exists():
            raise HTTPException(status_code=404, detail=f"图像 {name} 不存在")
        target_images.append((name, Image.open(img_path).convert("RGB")))

    selected_feature = np.array(req.selected_feature, dtype=np.float32)

    results = service.run_mode3_annotate(
        selected_feature=selected_feature,
        target_images=target_images,
        threshold=req.similarity_threshold,
        export_format=req.export_format,
    )

    return results


@router.post("/update_params", summary="更新标注参数")
async def update_params(req: ThresholdUpdateRequest):
    """动态更新标注参数（无需重启服务）"""
    service = get_service()
    updated = {}

    if req.similarity_threshold is not None:
        service.pipeline.similarity_threshold = req.similarity_threshold
        updated["similarity_threshold"] = req.similarity_threshold

    if req.kmeans_clusters is not None:
        service.pipeline.kmeans_clusters = req.kmeans_clusters
        updated["kmeans_clusters"] = req.kmeans_clusters

    if req.score_thr is not None:
        service.pipeline.detector.score_thr = req.score_thr
        updated["score_thr"] = req.score_thr

    return {"message": "参数已更新", "updated": updated}
