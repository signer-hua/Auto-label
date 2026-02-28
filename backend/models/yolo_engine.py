"""
YOLO-World 单例引擎
负责：基于文本提示的开放词汇目标检测。
采用单例模式，Celery Worker 启动时预热加载到 GPU (float16)。

核心方法：
    - warmup(): Worker 启动时调用，加载模型到 GPU
    - detect(image, text_prompts): 文本驱动检测，返回 bbox + 类别 + 置信度

链路位置：模式1 第一步 —— 文本提示 → YOLO-World 检测 bbox → SAM3 生成 Mask
"""
import sys
import gc
import torch
import numpy as np
from pathlib import Path
from typing import Optional
from PIL import Image

# 将内置 yolo_world 库加入 Python 路径
_LIBS_ROOT = Path(__file__).parent.parent / "libs"
_YOLOWORLD_LIB = _LIBS_ROOT / "yolo_world"
if str(_YOLOWORLD_LIB) not in sys.path:
    sys.path.insert(0, str(_YOLOWORLD_LIB))

from backend.core.config import (
    YOLOWORLD_CONFIG, YOLOWORLD_WEIGHTS, YOLOWORLD_DEVICE,
    YOLOWORLD_SCORE_THR, YOLOWORLD_NMS_THR,
)

import logging
logger = logging.getLogger(__name__)


class YOLOWorldEngine:
    """
    YOLO-World-v2-S 单例引擎。

    使用方式：
        engine = YOLOWorldEngine.get_instance()
        engine.warmup()
        detections = engine.detect(image, ["person", "car", "dog"])

    输出格式：
        [{"box": [x1,y1,x2,y2], "label": "person", "score": 0.92, "label_id": 0}, ...]
    """

    _instance: Optional["YOLOWorldEngine"] = None

    def __init__(self):
        self.model = None
        self.test_pipeline = None
        self.device = YOLOWORLD_DEVICE
        self.config_path = YOLOWORLD_CONFIG
        self.weights_path = YOLOWORLD_WEIGHTS
        self.score_thr = YOLOWORLD_SCORE_THR
        self.nms_thr = YOLOWORLD_NMS_THR
        self._warmed_up = False

    @classmethod
    def get_instance(cls) -> "YOLOWorldEngine":
        """获取全局单例"""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def warmup(self) -> None:
        """
        预热模型：加载 YOLO-World 到 GPU。
        应在 Celery Worker 启动时调用一次。

        注意：YOLO-World 基于 mmdet 生态，需要先 import yolo_world
        触发所有 @MODELS.register_module() 注册。
        """
        if self._warmed_up:
            return

        logger.info("[YOLO-World] Warming up model on %s...", self.device)

        # 触发 yolo_world 包的 Registry 注册
        import yolo_world  # noqa: F401

        from mmengine.config import Config
        from mmengine.dataset import Compose
        from mmdet.apis import init_detector
        from mmdet.utils import get_test_pipeline_cfg

        cfg = Config.fromfile(self.config_path)

        self.model = init_detector(
            cfg,
            checkpoint=self.weights_path if self.weights_path else None,
            device=self.device,
        )
        # 半精度推理节省显存
        self.model = self.model.half()

        # 构建测试 pipeline
        test_pipeline_cfg = get_test_pipeline_cfg(cfg=cfg)
        # 替换为从 ndarray 加载（而非文件路径）
        test_pipeline_cfg[0].type = 'mmdet.LoadImageFromNDArray'
        self.test_pipeline = Compose(test_pipeline_cfg)

        self._warmed_up = True
        logger.info("[YOLO-World] Model warmed up successfully.")

    @torch.no_grad()
    def detect(
        self,
        image: Image.Image,
        text_prompts: list[str],
        score_thr: float | None = None,
    ) -> list[dict]:
        """
        基于文本提示检测图像中的目标。

        Args:
            image: PIL Image (RGB)
            text_prompts: 文本提示列表，如 ["person", "bus", "car"]
            score_thr: 置信度阈值，None 则使用默认配置

        Returns:
            detections: 列表，每个元素：
                {
                    "box": [x1, y1, x2, y2],  # 像素坐标
                    "label": "person",          # 文本标签
                    "score": 0.92,              # 置信度
                    "label_id": 0,              # 类别索引
                }
        """
        if not self._warmed_up:
            self.warmup()

        threshold = score_thr if score_thr is not None else self.score_thr

        # PIL → numpy (RGB)
        img_array = np.array(image)

        # 构建文本格式：[[类别1], [类别2], ..., [' ']]（mmdet 要求的格式）
        texts = [[t] for t in text_prompts] + [[' ']]

        # 构建数据
        data_info = dict(img=img_array, img_id=0, texts=texts)
        data_info = self.test_pipeline(data_info)
        data_batch = dict(
            inputs=data_info['inputs'].unsqueeze(0).to(self.device),
            data_samples=[data_info['data_samples']],
        )

        # 推理
        output = self.model.test_step(data_batch)[0]
        pred_instances = output.pred_instances

        # 过滤低置信度
        keep = pred_instances.scores.float() > threshold
        pred_instances = pred_instances[keep]

        # 转换结果
        detections = []
        if len(pred_instances) > 0:
            bboxes = pred_instances.bboxes.cpu().numpy()
            labels = pred_instances.labels.cpu().numpy()
            scores = pred_instances.scores.cpu().numpy()

            for i in range(len(bboxes)):
                label_id = int(labels[i])
                label_text = text_prompts[label_id] if label_id < len(text_prompts) else "unknown"
                detections.append({
                    "box": bboxes[i].tolist(),   # [x1, y1, x2, y2]
                    "label": label_text,
                    "score": float(scores[i]),
                    "label_id": label_id,
                })

        logger.info("[YOLO-World] Detected %d objects for prompts %s", len(detections), text_prompts)
        return detections

    def release_memory(self) -> None:
        """释放 GPU 显存（批量任务完成后调用）"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
            logger.info("[YOLO-World] GPU memory cache cleared.")
