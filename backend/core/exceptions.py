"""
全局异常处理模块
定义业务异常类，并注册 FastAPI 全局异常处理器。
"""
from fastapi import Request
from fastapi.responses import JSONResponse


class TaskNotFoundError(Exception):
    """任务不存在"""
    def __init__(self, task_id: str):
        self.task_id = task_id
        super().__init__(f"Task not found: {task_id}")


class ModelLoadError(Exception):
    """模型加载失败"""
    def __init__(self, model_name: str, detail: str = ""):
        self.model_name = model_name
        self.detail = detail
        super().__init__(f"Failed to load model '{model_name}': {detail}")


class FileUploadError(Exception):
    """文件上传错误"""
    def __init__(self, filename: str, detail: str = ""):
        self.filename = filename
        self.detail = detail
        super().__init__(f"Upload failed for '{filename}': {detail}")


class MaskConversionError(Exception):
    """Mask 转换失败"""
    def __init__(self, detail: str = ""):
        self.detail = detail
        super().__init__(f"Mask conversion failed: {detail}")


class EmptyTextPromptError(Exception):
    """文本提示为空"""
    def __init__(self):
        super().__init__("Text prompt cannot be empty")


class YOLODetectionError(Exception):
    """YOLO-World 推理失败"""
    def __init__(self, detail: str = ""):
        self.detail = detail
        super().__init__(f"YOLO-World detection failed: {detail}")


class NoDetectionResultError(Exception):
    """检测/匹配无结果"""
    def __init__(self, mode: str = "", detail: str = ""):
        self.mode = mode
        self.detail = detail
        super().__init__(f"No detection results ({mode}): {detail}")


def register_exception_handlers(app):
    """
    注册全局异常处理器到 FastAPI 应用。
    将业务异常统一转换为结构化 JSON 响应。

    Args:
        app: FastAPI 应用实例
    """

    @app.exception_handler(TaskNotFoundError)
    async def task_not_found_handler(request: Request, exc: TaskNotFoundError):
        return JSONResponse(
            status_code=404,
            content={"error": "task_not_found", "detail": str(exc), "task_id": exc.task_id},
        )

    @app.exception_handler(ModelLoadError)
    async def model_load_handler(request: Request, exc: ModelLoadError):
        return JSONResponse(
            status_code=503,
            content={"error": "model_load_failed", "detail": str(exc), "model": exc.model_name},
        )

    @app.exception_handler(FileUploadError)
    async def file_upload_handler(request: Request, exc: FileUploadError):
        return JSONResponse(
            status_code=400,
            content={"error": "upload_failed", "detail": str(exc), "filename": exc.filename},
        )

    @app.exception_handler(MaskConversionError)
    async def mask_conversion_handler(request: Request, exc: MaskConversionError):
        return JSONResponse(
            status_code=500,
            content={"error": "mask_conversion_failed", "detail": str(exc)},
        )

    @app.exception_handler(EmptyTextPromptError)
    async def empty_text_handler(request: Request, exc: EmptyTextPromptError):
        return JSONResponse(
            status_code=400,
            content={"error": "empty_text_prompt", "detail": str(exc)},
        )

    @app.exception_handler(YOLODetectionError)
    async def yolo_detection_handler(request: Request, exc: YOLODetectionError):
        return JSONResponse(
            status_code=500,
            content={"error": "yolo_detection_failed", "detail": str(exc)},
        )

    @app.exception_handler(NoDetectionResultError)
    async def no_detection_handler(request: Request, exc: NoDetectionResultError):
        return JSONResponse(
            status_code=200,
            content={"error": "no_detection_result", "detail": str(exc), "mode": exc.mode},
        )
