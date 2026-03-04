"""
SAM3 单例引擎（增强版）
负责：基于 bbox 驱动的 Mask 生成，采用 float32 + autocast 自动混合精度推理。

v2 增强：
    - 多 Mask 并集融合（multimask → union）提升召回率
    - 形态学后处理（闭运算填充空洞 + 高斯平滑边缘）
    - 降低分数阈值适配小目标
"""
import sys
import gc
import torch
import numpy as np
from pathlib import Path
from typing import Optional
from PIL import Image

_LIBS_ROOT = Path(__file__).parent.parent / "libs"
_SAM3_LIB = _LIBS_ROOT / "sam3"
if str(_SAM3_LIB) not in sys.path:
    sys.path.insert(0, str(_SAM3_LIB))

from backend.core.config import (
    SAM3_CHECKPOINT, SAM3_DEVICE,
    SAM3_MULTIMASK_UNION, SAM3_SCORE_THRESHOLD, SAM3_MORPH_KERNEL,
)

import logging
logger = logging.getLogger(__name__)


class SAMEngine:
    """
    SAM3 单例引擎（增强版）。

    增强特性：
        - 多 Mask 并集融合：取多个高分 Mask 的并集，提升目标完整性
        - 形态学后处理：闭运算填充空洞 + 高斯模糊平滑边缘
        - 降低分数阈值：score_threshold=0.6 提升小目标召回率
    """

    _instance: Optional["SAMEngine"] = None

    def __init__(self):
        self.model = None
        self.processor = None
        self.device = SAM3_DEVICE
        self._warmed_up = False

    @classmethod
    def get_instance(cls) -> "SAMEngine":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def warmup(self) -> None:
        if self._warmed_up:
            return

        logger.info("[SAM3] Warming up model on %s (float32 + autocast)...", self.device)

        from sam3.model_builder import build_sam3_image_model
        from sam3.model.sam3_image_processor import Sam3Processor

        checkpoint = SAM3_CHECKPOINT if SAM3_CHECKPOINT else None
        load_from_hf = True
        if checkpoint and Path(checkpoint).exists():
            logger.info("[SAM3] Loading weights from local: %s", checkpoint)
            load_from_hf = False
        else:
            logger.info("[SAM3] Downloading weights from HuggingFace...")
            checkpoint = None

        self.model = build_sam3_image_model(
            device=self.device,
            eval_mode=True,
            checkpoint_path=checkpoint,
            load_from_HF=load_from_hf,
            enable_segmentation=True,
            enable_inst_interactivity=False,
            compile=False,
        )
        self.processor = Sam3Processor(self.model)
        self._warmed_up = True
        logger.info("[SAM3] Model warmed up successfully.")

    @staticmethod
    def _morphological_postprocess(mask: np.ndarray) -> np.ndarray:
        """
        形态学后处理：闭运算填充空洞 + 高斯模糊平滑边缘。

        Args:
            mask: 布尔 numpy 数组 [H, W]

        Returns:
            processed_mask: 优化后的布尔 Mask
        """
        import cv2

        kernel_size = SAM3_MORPH_KERNEL
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)
        )

        mask_uint8 = mask.astype(np.uint8) * 255
        closed = cv2.morphologyEx(mask_uint8, cv2.MORPH_CLOSE, kernel, iterations=2)
        blurred = cv2.GaussianBlur(closed, (kernel_size, kernel_size), 0)
        result = (blurred > 127).astype(bool)

        return result

    @torch.no_grad()
    def generate_mask(
        self,
        image: Image.Image,
        bbox: list[float],
        use_multimask_union: bool | None = None,
        use_morph: bool = True,
    ) -> np.ndarray:
        """
        基于边界框生成实例 Mask（增强版）。

        增强策略：
        1. 获取多个候选 Mask
        2. 对高分 Mask 取并集融合（提升完整性）
        3. 形态学后处理（填充空洞 + 平滑边缘）

        Args:
            image: PIL Image (RGB)
            bbox: [x1, y1, x2, y2] 像素坐标
            use_multimask_union: 是否启用多 Mask 并集（None 则使用配置）
            use_morph: 是否启用形态学后处理

        Returns:
            mask: 布尔 numpy 数组，shape [H, W]
        """
        if not self._warmed_up:
            self.warmup()

        if use_multimask_union is None:
            use_multimask_union = SAM3_MULTIMASK_UNION

        x1, y1, x2, y2 = bbox
        img_w, img_h = image.size

        cx = (x1 + x2) / 2.0 / img_w
        cy = (y1 + y2) / 2.0 / img_h
        w = (x2 - x1) / img_w
        h = (y2 - y1) / img_h

        with torch.autocast(device_type="cuda", dtype=torch.float16):
            state = self.processor.set_image(image)
            output = self.processor.add_geometric_prompt(
                box=[cx, cy, w, h],
                label=True,
                state=state,
            )

        if not output or "masks" not in output or len(output["masks"]) == 0:
            return np.zeros((img_h, img_w), dtype=bool)

        masks = output["masks"]
        scores = output["scores"]

        if hasattr(scores, 'cpu'):
            scores_np = scores.cpu().numpy().flatten()
        elif isinstance(scores, (list, tuple)):
            scores_np = np.array(scores).flatten()
        else:
            scores_np = np.array([scores]).flatten()

        def _to_2d_bool(m):
            if isinstance(m, torch.Tensor):
                m = m.cpu().numpy()
            while m.ndim > 2:
                m = m[0]
            return m.astype(bool)

        if use_multimask_union and len(masks) > 1:
            valid_indices = np.where(scores_np >= SAM3_SCORE_THRESHOLD)[0]

            if len(valid_indices) == 0:
                best_idx = scores_np.argmax()
                final_mask = _to_2d_bool(masks[best_idx])
            elif len(valid_indices) == 1:
                final_mask = _to_2d_bool(masks[valid_indices[0]])
            else:
                union_mask = np.zeros((img_h, img_w), dtype=bool)
                for idx in valid_indices:
                    m = _to_2d_bool(masks[idx])
                    if m.shape != (img_h, img_w):
                        import cv2
                        m = cv2.resize(m.astype(np.uint8), (img_w, img_h),
                                       interpolation=cv2.INTER_NEAREST).astype(bool)
                    union_mask = union_mask | m
                final_mask = union_mask
        else:
            best_idx = scores_np.argmax()
            final_mask = _to_2d_bool(masks[best_idx])

        if final_mask.shape != (img_h, img_w):
            import cv2
            final_mask = cv2.resize(
                final_mask.astype(np.uint8), (img_w, img_h),
                interpolation=cv2.INTER_NEAREST
            ).astype(bool)

        if use_morph and final_mask.sum() > 50:
            final_mask = self._morphological_postprocess(final_mask)

        return final_mask

    @torch.no_grad()
    def generate_mask_from_exemplars(
        self,
        target_image: Image.Image,
        exemplar_crops: list[Image.Image],
        exemplar_coords: list[list[float]],
        text_prompt: str | None = None,
        multimask_output: bool = True,
    ) -> np.ndarray:
        """
        基于图像示例+坐标+文本提示的混合推理（SAM3-PCS范式）。

        策略：
        1. 对每个参考图示例的坐标，在目标图上生成候选Mask
        2. 启用multimask_output获取多个候选
        3. 对所有高分候选取并集融合

        Args:
            target_image: 待分割的目标图像
            exemplar_crops: 参考图裁剪区域列表（用于辅助特征引导）
            exemplar_coords: 每个示例对应的目标图bbox坐标列表 [[x1,y1,x2,y2], ...]
            text_prompt: 可选文本提示，用于语义引导
            multimask_output: 是否启用多Mask输出

        Returns:
            fused_mask: 融合后的布尔Mask [H, W]
        """
        if not self._warmed_up:
            self.warmup()

        img_w, img_h = target_image.size
        all_masks = []

        with torch.autocast(device_type="cuda", dtype=torch.float16):
            state = self.processor.set_image(target_image)

            for coords in exemplar_coords:
                x1, y1, x2, y2 = coords
                cx = (x1 + x2) / 2.0 / img_w
                cy = (y1 + y2) / 2.0 / img_h
                w = (x2 - x1) / img_w
                h = (y2 - y1) / img_h

                output = self.processor.add_geometric_prompt(
                    box=[cx, cy, w, h],
                    label=True,
                    state=state,
                )

                if not output or "masks" not in output or len(output["masks"]) == 0:
                    continue

                masks = output["masks"]
                scores = output["scores"]

                if hasattr(scores, 'cpu'):
                    scores_np = scores.cpu().numpy().flatten()
                elif isinstance(scores, (list, tuple)):
                    scores_np = np.array(scores).flatten()
                else:
                    scores_np = np.array([scores]).flatten()

                if multimask_output and len(masks) > 1:
                    valid_idx = np.where(scores_np >= SAM3_SCORE_THRESHOLD)[0]
                    if len(valid_idx) == 0:
                        valid_idx = [scores_np.argmax()]
                    for idx in valid_idx:
                        m = masks[idx]
                        if isinstance(m, torch.Tensor):
                            m = m.cpu().numpy()
                        while m.ndim > 2:
                            m = m[0]
                        m = m.astype(bool)
                        if m.shape != (img_h, img_w):
                            import cv2
                            m = cv2.resize(m.astype(np.uint8), (img_w, img_h),
                                           interpolation=cv2.INTER_NEAREST).astype(bool)
                        all_masks.append(m)
                else:
                    best_idx = scores_np.argmax()
                    m = masks[best_idx]
                    if isinstance(m, torch.Tensor):
                        m = m.cpu().numpy()
                    while m.ndim > 2:
                        m = m[0]
                    m = m.astype(bool)
                    if m.shape != (img_h, img_w):
                        import cv2
                        m = cv2.resize(m.astype(np.uint8), (img_w, img_h),
                                       interpolation=cv2.INTER_NEAREST).astype(bool)
                    all_masks.append(m)

        if not all_masks:
            return np.zeros((img_h, img_w), dtype=bool)

        fused = np.zeros((img_h, img_w), dtype=bool)
        for m in all_masks:
            fused = fused | m

        if fused.sum() > 50:
            fused = self._morphological_postprocess(fused)

        return fused

    @torch.no_grad()
    def generate_corrected_mask(
        self,
        image: Image.Image,
        positive_boxes: list[list[float]],
        negative_boxes: list[list[float]],
    ) -> np.ndarray:
        """
        人机协同负向提示修正Mask。

        通过正向框（保留区域）和负向框（排除区域）组合，
        生成修正后的精准Mask。

        Args:
            image: PIL Image (RGB)
            positive_boxes: 正向框列表 [[x1,y1,x2,y2], ...]
            negative_boxes: 负向框列表 [[x1,y1,x2,y2], ...]（需排除的区域）

        Returns:
            corrected_mask: 修正后的布尔Mask [H, W]
        """
        if not self._warmed_up:
            self.warmup()

        img_w, img_h = image.size

        positive_mask = np.zeros((img_h, img_w), dtype=bool)
        for box in positive_boxes:
            m = self.generate_mask(image, box, use_multimask_union=True, use_morph=True)
            positive_mask = positive_mask | m

        negative_mask = np.zeros((img_h, img_w), dtype=bool)
        for box in negative_boxes:
            m = self.generate_mask(image, box, use_multimask_union=False, use_morph=False)
            negative_mask = negative_mask | m

        corrected = positive_mask & (~negative_mask)

        if corrected.sum() > 50:
            corrected = self._morphological_postprocess(corrected)

        return corrected

    def load_lora_weights(self, category_id: str) -> bool:
        """
        加载指定类别的 LoRA 微调权重。
        若该类别权重不存在则跳过，使用原生 SAM3 推理。

        Args:
            category_id: 类别 ID

        Returns:
            loaded: 是否成功加载
        """
        lora_dir = Path(__file__).parent.parent.parent / "weights" / "lora" / f"sam3_lora_{category_id}"
        if not lora_dir.exists():
            return False

        try:
            lora_pt = lora_dir / "lora_weights.pt"
            if lora_pt.exists():
                state = torch.load(str(lora_pt), map_location=self.device, weights_only=True)
                missing, unexpected = self.model.load_state_dict(state, strict=False)
                logger.info("[SAM3] LoRA weights loaded for category '%s' (%d params)",
                            category_id, len(state))
                return True

            try:
                from peft import PeftModel
                self.model = PeftModel.from_pretrained(self.model, str(lora_dir))
                logger.info("[SAM3] LoRA (peft) loaded for category '%s'", category_id)
                return True
            except Exception:
                pass

        except Exception as e:
            logger.warning("[SAM3] LoRA load failed for '%s': %s", category_id, str(e))

        return False

    def release_memory(self) -> None:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
            logger.info("[SAM3] GPU memory cache cleared.")
