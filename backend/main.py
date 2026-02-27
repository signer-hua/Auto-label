"""
Auto-label 后端主入口
FastAPI 应用，提供标注、文件管理、导出等 API
"""
import sys
from pathlib import Path

# 确保项目根目录在 Python 路径中
PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from backend.config import server_config, UPLOAD_DIR, OUTPUT_DIR
from backend.api.files import router as files_router
from backend.api.annotate import router as annotate_router
from backend.api.export import router as export_router

# ==================== 创建 FastAPI 应用 ====================
app = FastAPI(
    title="Auto-label 人机协同图像自动标注工具",
    description="""
    基于 SAM3 + YOLO-World + DINOv3 的多模式人机协同标注系统

    ## 三大标注模式
    - **模式1**: 文本提示一键自动标注（纯自动）
    - **模式2**: 人工预标注 → 批量自动标注（人机协同）
    - **模式3**: 选实例 → 跨图批量标注（人机协同）

    ## 核心算法
    - YOLO-World-v2-S: 开放词汇目标检测
    - DINOv3 ViT-S/16: 自监督视觉特征提取
    - SAM3: 精准实例分割
    """,
    version="1.0.0",
)

# ==================== CORS 中间件 ====================
app.add_middleware(
    CORSMiddleware,
    allow_origins=server_config.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==================== 静态文件服务 ====================
# 上传的图像可通过 /uploads/xxx 访问
app.mount("/uploads", StaticFiles(directory=str(UPLOAD_DIR)), name="uploads")
app.mount("/outputs", StaticFiles(directory=str(OUTPUT_DIR)), name="outputs")

# ==================== 注册路由 ====================
app.include_router(files_router)
app.include_router(annotate_router)
app.include_router(export_router)


# ==================== 根路由 ====================
@app.get("/", tags=["系统"])
async def root():
    return {
        "name": "Auto-label 人机协同图像自动标注工具",
        "version": "1.0.0",
        "status": "running",
        "docs": "/docs",
        "modes": {
            "mode1": "文本提示一键自动标注",
            "mode2": "人工预标注 → 批量自动标注",
            "mode3": "选实例 → 跨图批量标注",
        },
    }


@app.get("/api/health", tags=["系统"])
async def health_check():
    """健康检查接口"""
    return {"status": "ok"}


@app.get("/api/config", tags=["系统"])
async def get_config():
    """获取当前配置"""
    from backend.config import model_config, annotation_config
    return {
        "model": model_config.model_dump(),
        "annotation": annotation_config.model_dump(),
    }


# ==================== 启动入口 ====================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "backend.main:app",
        host=server_config.host,
        port=server_config.port,
        workers=server_config.workers,
        reload=False,
    )
