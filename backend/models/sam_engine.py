"""
SAM3 单例引擎
负责：基于 bbox 驱动的 Mask 生成，支持半精度 (float16) 推理。
采用单例模式，Celery Worker 启动时预热加载到 GPU。

核心方法：
    - warmup(): Worker 启动时调用，加载模型到 GPU 并转为半精度
    - generate_mask(image, bbox): 输入图像 + bbox，输出布尔 Mask
"""
import sys
import gc
import torch
import numpy as np
from pathlib import Path
from typing import Optional
from PIL import Image

# 将内置 sam3 库加入 Python 路径
_LIBS_ROOT = Path(__file__).parent.parent / "libs"
_SAM3_LIB = _LIBS_ROOT / "sam3"
if str(_SAM3_LIB) not in sys.path:
    sys.path.insert(0, str(_SAM3_LIB))

from backend.core.config import SAM3_CHECKPOINT, SAM3_DEVICE

import logging
logger = logging.getLogger(__name__)


class SAMEngine:
    """
    SAM3 单例引擎。

    使用方式：
        engine = SAMEngine.get_instance()
        engine.warmup()  # Worker 启动时调用一次
        mask = engine.generate_mask(image, [x1, y1, x2, y2])
    """

    _instance: Optional["SAMEngine"] = None

    def __init__(self):
        self.model = None
        self.processor = None
        self.device = SAM3_DEVICE
        self._warmed_up = False

    @classmethod
    def get_instance(cls) -> "SAMEngine":
        """获取全局单例"""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def warmup(self) -> None:
        """
        预热模型：加载 SAM3 到 GPU，转为半精度 (float16)。
        应在 Celery Worker 启动时调用一次。
        """
        if self._warmed_up:
            return

        logger.info("[SAM3] Warming up model on %s (half precision)...", self.device)

        from sam3.model_builder import build_sam3_image_model
        from sam3.model.sam3_image_processor import Sam3Processor

        checkpoint = SAM3_CHECKPOINT if SAM3_CHECKPOINT else None
        self.model = build_sam3_image_model(
            device=self.device,
            eval_mode=True,
            checkpoint_path=checkpoint,
            load_from_HF=True if checkpoint is None else False,
            enable_segmentation=True,
            enable_inst_interactivity=False,
            compile=False,
        )
        # 转为半精度以节省显存
        self.model = self.model.half()
        self.processor = Sam3Processor(self.model)
        self._warmed_up = True
        logger.info("[SAM3] Model warmed up successfully.")

    @torch.no_grad()
    def generate_mask(
        self,
        image: Image.Image,
        bbox: list[float],
    ) -> np.ndarray:
        """
        基于边界框生成实例 Mask。

        Args:
            image: PIL Image (RGB)
            bbox: [x1, y1, x2, y2] 像素坐标

        Returns:
            mask: 布尔 numpy 数组，shape [H, W]
        """
        if not self._warmed_up:
            self.warmup()

        x1, y1, x2, y2 = bbox
        img_w, img_h = image.size

        # 设置图像（编码）
        state = self.processor.set_image(image)

        # 转换为 SAM3 归一化格式 [cx, cy, w, h]
        cx = (x1 + x2) / 2.0 / img_w
        cy = (y1 + y2) / 2.0 / img_h
        w = (x2 - x1) / img_w
        h = (y2 - y1) / img_h

        output = self.processor.add_geometric_prompt(
            box=[cx, cy, w, h],
            label=True,
            state=state,
        )

        if output and "masks" in output and len(output["masks"]) > 0:
            scores = output["scores"]
            best_idx = scores.argmax().item() if hasattr(scores, 'argmax') else 0
            mask = output["masks"][best_idx]
            if isinstance(mask, torch.Tensor):
                mask = mask.cpu().numpy()
            return mask.astype(bool)

        # fallback: 返回空 Mask
        return np.zeros((img_h, img_w), dtype=bool)

    def release_memory(self) -> None:
        """释放 GPU 显存（批量任务完成后调用）"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
            logger.info("[SAM3] GPU memory cache cleared.")
