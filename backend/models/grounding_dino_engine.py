"""
Grounding DINO 单例引擎
负责：基于文本提示的开放词汇目标检测。
采用 HuggingFace transformers 库加载模型，无需 MMDetection/MMEngine 依赖。
单例模式，Celery Worker 启动时预热加载到 GPU (float32 + autocast 自动混合精度)。

核心方法：
    - warmup(): Worker 启动时调用，加载模型到 GPU
    - detect(image, text_prompts): 文本驱动检测，返回 bbox + 类别 + 置信度
    - release_memory(): 释放 GPU 显存

链路位置：模式1 第一步 —— 文本提示 → Grounding DINO 检测 bbox → SAM3 生成 Mask

输出格式：
    [{"box": [x1,y1,x2,y2], "label": str, "score": float, "label_id": int}]
    文本提示用 " . " 分隔（Grounding DINO 规范），自动处理中英文标点
    label_id 从 0 开始编号（COCO 中 category_id = label_id + 1）
"""
import re
import gc
import torch
import numpy as np
from typing import Optional
from PIL import Image

from backend.core.config import (
    GROUNDING_DINO_MODEL_NAME, GROUNDING_DINO_DEVICE,
    GROUNDING_DINO_SCORE_THR, GROUNDING_DINO_BOX_THR,
)

import logging
logger = logging.getLogger(__name__)


class GroundingDINOEngine:
    """
    Grounding DINO 单例引擎（基于 HuggingFace transformers）。

    使用方式：
        engine = GroundingDINOEngine.get_instance()
        engine.warmup()
        detections = engine.detect(image, ["person", "car", "dog"])

    输出格式：
        [{"box": [x1,y1,x2,y2], "label": "person", "score": 0.92, "label_id": 0}, ...]
    """

    _instance: Optional["GroundingDINOEngine"] = None

    def __init__(self):
        self.model = None
        self.processor = None
        self.device = GROUNDING_DINO_DEVICE
        self.model_name = GROUNDING_DINO_MODEL_NAME
        self.score_thr = GROUNDING_DINO_SCORE_THR
        self.box_thr = GROUNDING_DINO_BOX_THR
        self._warmed_up = False

    @classmethod
    def get_instance(cls) -> "GroundingDINOEngine":
        """获取全局单例"""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def warmup(self) -> None:
        """
        预热模型：加载 Grounding DINO 到 GPU (float32)，推理时用 autocast 自动混合精度。

        Grounding DINO 的 text_enhancer_layer 内部会将位置编码强制转为 float32，
        如果模型权重是 float16 会导致 dtype 不匹配。因此用 float32 加载模型，
        推理时通过 torch.autocast 让 GPU 自动选择最优精度。

        model_name 可以是 HuggingFace Hub 名称（如 IDEA-Research/grounding-dino-base），
        也可以是本地已下载的模型目录路径。首次使用 Hub 名称时会自动下载权重。
        """
        if self._warmed_up:
            return

        logger.info("[Grounding DINO] Warming up model '%s' on %s...",
                    self.model_name, self.device)

        from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

        self.processor = AutoProcessor.from_pretrained(self.model_name)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(
            self.model_name,
            torch_dtype=torch.float32,
            low_cpu_mem_usage=False,
        )
        self.model = self.model.to(self.device).eval()

        self._warmed_up = True
        logger.info("[Grounding DINO] Model warmed up successfully (float32 + autocast).")

    @staticmethod
    def _normalize_text_prompts(text_prompts: list[str]) -> tuple[str, list[str]]:
        """
        将文本提示列表转换为 Grounding DINO 要求的格式。

        Grounding DINO 使用 " . " 分隔不同类别，每个类别末尾也需要 " ."。
        同时处理中英文逗号、顿号等分隔符。

        Args:
            text_prompts: 原始文本提示列表，如 ["person", "car"]

        Returns:
            caption: Grounding DINO 格式的文本，如 "person . car ."
            cleaned_prompts: 清洗后的提示列表，如 ["person", "car"]
        """
        cleaned = []
        for prompt in text_prompts:
            p = prompt.strip().lower()
            p = re.sub(r'[，、；;]', ',', p)
            p = p.strip(' .,;')
            if p:
                cleaned.append(p)

        caption = " . ".join(cleaned) + " ." if cleaned else ""
        return caption, cleaned

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
                    "box": [x1, y1, x2, y2],  # 像素坐标（浮点数，保留4位小数）
                    "label": "person",          # 文本标签
                    "score": 0.92,              # 置信度
                    "label_id": 0,              # 类别索引（从0开始）
                }
        """
        if not self._warmed_up:
            self.warmup()

        threshold = score_thr if score_thr is not None else self.score_thr

        caption, cleaned_prompts = self._normalize_text_prompts(text_prompts)
        if not caption:
            return []

        inputs = self.processor(
            images=image, text=caption, return_tensors="pt"
        )
        inputs = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                  for k, v in inputs.items()}

        with torch.autocast(device_type="cuda", dtype=torch.float16):
            outputs = self.model(**inputs)

        results = self.processor.post_process_grounded_object_detection(
            outputs,
            input_ids=inputs["input_ids"],
            threshold=self.box_thr,
            text_threshold=threshold,
            target_sizes=[image.size[::-1]],  # (height, width)
        )[0]

        boxes = results["boxes"].cpu().numpy()     # [N, 4] in xyxy format
        scores = results["scores"].cpu().numpy()    # [N]
        labels = results["labels"]                  # list of str

        detections = []
        for i in range(len(boxes)):
            det_label = labels[i].strip().lower()
            det_score = float(scores[i])

            if det_score < threshold:
                continue

            label_id = self._match_label_to_prompt(det_label, cleaned_prompts)

            detections.append({
                "box": [round(float(boxes[i][j]), 4) for j in range(4)],
                "label": cleaned_prompts[label_id] if label_id < len(cleaned_prompts) else det_label,
                "score": round(det_score, 4),
                "label_id": label_id,
            })

        logger.info("[Grounding DINO] Detected %d objects for prompts %s",
                    len(detections), cleaned_prompts)
        return detections

    @staticmethod
    def _match_label_to_prompt(det_label: str, prompts: list[str]) -> int:
        """将 Grounding DINO 输出的 label 匹配回输入的 text_prompts 索引。"""
        det_label_lower = det_label.lower().strip()
        for idx, prompt in enumerate(prompts):
            if det_label_lower == prompt.lower().strip():
                return idx
        for idx, prompt in enumerate(prompts):
            if prompt.lower().strip() in det_label_lower or det_label_lower in prompt.lower().strip():
                return idx
        return 0

    def release_memory(self) -> None:
        """释放 GPU 显存（批量任务完成后调用）"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
            logger.info("[Grounding DINO] GPU memory cache cleared.")
