"""
Auto-label 后端主入口
FastAPI 应用：定义 /upload、/annotate/mode2、/tasks/{task_id} 路由。
静态文件服务：/data/images/、/data/masks/、/data/exports/。

启动命令：
    uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
"""
import sys
import logging
from pathlib import Path

# 确保项目根目录在 Python 路径中
PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from backend.core.config import (
    API_HOST, API_PORT, CORS_ORIGINS,
    IMAGE_DIR, MASK_DIR, EXPORT_DIR, DATA_DIR,
)
from backend.core.exceptions import register_exception_handlers
from backend.api.routes import router as api_router

# ==================== 日志配置 ====================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ==================== 创建 FastAPI 应用 ====================
app = FastAPI(
    title="Auto-label 人机协同图像自动标注工具",
    description="""
基于 SAM3 + DINOv3 的人机协同标注系统。
采用 FastAPI + Celery + Redis 异步架构，GPU 推理不阻塞 API 请求。

## 核心接口
- **POST /api/upload** — 上传图像
- **POST /api/annotate/mode2** — 触发模式2异步批量标注
- **GET /api/tasks/{task_id}** — 查询任务状态/进度/Mask URL
    """,
    version="2.0.0",
)

# ==================== CORS 中间件 ====================
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==================== 全局异常处理 ====================
register_exception_handlers(app)

# ==================== 静态文件服务 ====================
# 图像、Mask PNG、导出文件均通过静态路由访问
app.mount("/data/images", StaticFiles(directory=str(IMAGE_DIR)), name="images")
app.mount("/data/masks", StaticFiles(directory=str(MASK_DIR)), name="masks")
app.mount("/data/exports", StaticFiles(directory=str(EXPORT_DIR)), name="exports")

# ==================== 注册 API 路由 ====================
app.include_router(api_router)


# ==================== 系统接口 ====================
@app.get("/", tags=["系统"])
async def root():
    """根路由，返回系统信息"""
    return {
        "name": "Auto-label 人机协同图像自动标注工具",
        "version": "2.0.0",
        "architecture": "FastAPI + Celery + Redis",
        "docs": "/docs",
    }


@app.get("/api/health", tags=["系统"])
async def health_check():
    """健康检查"""
    return {"status": "ok"}


# ==================== 启动入口 ====================
if __name__ == "__main__":
    import uvicorn
    logger.info("Starting Auto-label API server on %s:%d", API_HOST, API_PORT)
    uvicorn.run(
        "backend.main:app",
        host=API_HOST,
        port=API_PORT,
        workers=1,
        reload=False,
    )
