"""
导出 API
标注结果导出为 COCO/VOC 格式
"""
import json
import zipfile
import io
from pathlib import Path
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse, FileResponse

from backend.config import OUTPUT_DIR

router = APIRouter(prefix="/api/export", tags=["导出"])


@router.get("/list", summary="列出所有导出结果")
async def list_exports():
    """列出输出目录中的所有标注结果文件"""
    exports = []
    for f in sorted(OUTPUT_DIR.iterdir()):
        if f.is_file():
            exports.append({
                "name": f.name,
                "path": str(f),
                "size": f.stat().st_size,
                "format": "coco" if f.suffix == ".json" else "voc",
            })
    return {"exports": exports, "count": len(exports)}


@router.get("/download/{filename}", summary="下载单个标注文件")
async def download_export(filename: str):
    """下载指定的标注结果文件"""
    file_path = OUTPUT_DIR / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="文件不存在")
    return FileResponse(
        str(file_path),
        media_type="application/octet-stream",
        filename=filename,
    )


@router.get("/download_all", summary="打包下载所有标注结果")
async def download_all():
    """将所有标注结果打包为 ZIP 下载"""
    files = list(OUTPUT_DIR.iterdir())
    if not files:
        raise HTTPException(status_code=404, detail="没有可下载的标注结果")

    # 创建内存中的 ZIP
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in files:
            if f.is_file():
                zf.write(f, f.name)

    zip_buffer.seek(0)
    return StreamingResponse(
        zip_buffer,
        media_type="application/zip",
        headers={"Content-Disposition": "attachment; filename=annotations.zip"},
    )


@router.delete("/clear", summary="清空所有导出结果")
async def clear_exports():
    """清空输出目录"""
    count = 0
    for f in OUTPUT_DIR.iterdir():
        if f.is_file():
            f.unlink()
            count += 1
    return {"message": f"已清空 {count} 个文件"}
