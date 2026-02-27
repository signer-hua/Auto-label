"""
文件管理 API
处理图像上传、列表查询、删除等操作
"""
import os
import uuid
import shutil
from typing import List
from pathlib import Path
from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import FileResponse

from backend.config import UPLOAD_DIR, OUTPUT_DIR

router = APIRouter(prefix="/api/files", tags=["文件管理"])

# 支持的图像格式
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}


def _validate_image(filename: str) -> bool:
    """验证文件是否为支持的图像格式"""
    ext = Path(filename).suffix.lower()
    return ext in ALLOWED_EXTENSIONS


@router.post("/upload", summary="上传图像文件")
async def upload_images(files: List[UploadFile] = File(...)):
    """
    上传一张或多张图像文件

    Returns:
        uploaded: 上传成功的文件信息列表
    """
    uploaded = []
    for file in files:
        if not _validate_image(file.filename):
            continue

        # 生成唯一文件名，保留原始扩展名
        ext = Path(file.filename).suffix.lower()
        unique_name = f"{uuid.uuid4().hex}{ext}"
        save_path = UPLOAD_DIR / unique_name

        with open(save_path, "wb") as f:
            content = await file.read()
            f.write(content)

        uploaded.append({
            "original_name": file.filename,
            "saved_name": unique_name,
            "path": str(save_path),
            "size": len(content),
        })

    if not uploaded:
        raise HTTPException(status_code=400, detail="没有有效的图像文件")

    return {"uploaded": uploaded, "count": len(uploaded)}


@router.get("/list", summary="获取已上传图像列表")
async def list_images():
    """列出所有已上传的图像文件"""
    images = []
    for f in sorted(UPLOAD_DIR.iterdir()):
        if f.is_file() and _validate_image(f.name):
            images.append({
                "name": f.name,
                "path": str(f),
                "size": f.stat().st_size,
            })
    return {"images": images, "count": len(images)}


@router.get("/image/{filename}", summary="获取图像文件")
async def get_image(filename: str):
    """返回指定图像文件"""
    file_path = UPLOAD_DIR / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="文件不存在")
    return FileResponse(str(file_path))


@router.delete("/delete/{filename}", summary="删除图像文件")
async def delete_image(filename: str):
    """删除指定图像文件"""
    file_path = UPLOAD_DIR / filename
    if file_path.exists():
        file_path.unlink()
        return {"message": f"已删除 {filename}"}
    raise HTTPException(status_code=404, detail="文件不存在")


@router.delete("/clear", summary="清空所有上传文件")
async def clear_uploads():
    """清空上传目录"""
    count = 0
    for f in UPLOAD_DIR.iterdir():
        if f.is_file():
            f.unlink()
            count += 1
    return {"message": f"已清空 {count} 个文件"}
