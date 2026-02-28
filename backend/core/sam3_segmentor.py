"""
SAM3 分割器封装
负责：基于提示（边界框/点/文本）生成精准实例 mask
算法库已内置于 backend/libs/sam3/ 中
"""
import sys
import torch
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from PIL import Image

# 将内置的 sam3 库加入 Python 路径
_LIBS_ROOT = Path(__file__).parent.parent / "libs"
_SAM3_LIB = _LIBS_ROOT / "sam3"
if str(_SAM3_LIB) not in sys.path:
    sys.path.insert(0, str(_SAM3_LIB))


class SAM3Segmentor:
    """
    SAM3 实例分割器
    核心功能：
    1. 基于边界框提示生成实例 mask
    2. 基于文本提示生成实例 mask
    3. 获取图像编码器特征（用于与 DINOv3 融合）
    """

    def __init__(self, device: str = "cuda", checkpoint_path: str = ""):
        """
        初始化 SAM3 分割器

        Args:
            device: 推理设备
            checkpoint_path: 权重路径，留空自动从 HuggingFace 下载
        """
        self.device = device
        self.checkpoint_path = checkpoint_path if checkpoint_path else None
        self.model = None
        self.processor = None

    def load_model(self):
        """加载 SAM3 预训练模型"""
        if self.model is not None:
            return

        print("[SAM3] 正在加载模型...")
        from sam3.model_builder import build_sam3_image_model
        from sam3.model.sam3_image_processor import Sam3Processor

        self.model = build_sam3_image_model(
            device=self.device,
            eval_mode=True,
            checkpoint_path=self.checkpoint_path,
            load_from_HF=True if self.checkpoint_path is None else False,
            enable_segmentation=True,
            enable_inst_interactivity=False,
            compile=False,
        )
        self.processor = Sam3Processor(self.model)
        print(f"[SAM3] 模型加载完成，设备: {self.device}")

    def set_image(self, image: Image.Image) -> dict:
        """
        编码图像，返回状态对象（后续推理复用）

        Args:
            image: PIL Image (RGB)
        Returns:
            state: 内部状态字典，包含图像编码结果
        """
        self.load_model()
        state = self.processor.set_image(image)
        return state

    @torch.no_grad()
    def segment_with_boxes(self, image: Image.Image,
                            boxes: List[List[float]]) -> List[Dict]:
        """
        基于边界框提示生成实例 mask

        Args:
            image: PIL Image
            boxes: 边界框列表，每个为 [x1, y1, x2, y2] 像素坐标
        Returns:
            results: 列表，每个元素包含 {mask, box, score}
        """
        self.load_model()
        state = self.processor.set_image(image)
        img_w, img_h = image.size
        results = []

        for box in boxes:
            x1, y1, x2, y2 = box
            # 转换为 SAM3 的 [cx, cy, w, h] 归一化格式
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
                # 取置信度最高的 mask
                scores = output["scores"]
                best_idx = scores.argmax().item() if hasattr(scores, 'argmax') else 0
                mask = output["masks"][best_idx]
                if isinstance(mask, torch.Tensor):
                    mask = mask.cpu().numpy()
                score = float(scores[best_idx]) if hasattr(scores, '__getitem__') else float(scores)

                results.append({
                    "mask": mask.astype(bool),
                    "box": box,
                    "score": score,
                })

            # 重置几何提示，为下一个 box 准备
            state = self.processor.set_image(image)

        return results

    @torch.no_grad()
    def segment_with_text(self, image: Image.Image,
                           text_prompt: str) -> List[Dict]:
        """
        基于文本提示生成实例 mask

        Args:
            image: PIL Image
            text_prompt: 文本描述，如 "a yellow school bus"
        Returns:
            results: 列表，每个元素包含 {mask, box, score, label}
        """
        self.load_model()
        state = self.processor.set_image(image)

        output = self.processor.set_text_prompt(
            state=state,
            prompt=text_prompt,
        )

        results = []
        if output and "masks" in output:
            masks = output["masks"]
            boxes = output.get("boxes", [None] * len(masks))
            scores = output.get("scores", [1.0] * len(masks))

            for i in range(len(masks)):
                mask = masks[i]
                if isinstance(mask, torch.Tensor):
                    mask = mask.cpu().numpy()
                box = boxes[i] if boxes[i] is not None else None
                if isinstance(box, torch.Tensor):
                    box = box.cpu().numpy().tolist()
                score = float(scores[i]) if hasattr(scores, '__getitem__') else float(scores)

                results.append({
                    "mask": mask.astype(bool),
                    "box": box,
                    "score": score,
                    "label": text_prompt,
                })

        return results

    @torch.no_grad()
    def segment_regions(self, image: Image.Image,
                         region_masks: List[np.ndarray]) -> List[Dict]:
        """
        基于聚类区域 mask 生成精细分割

        对每个粗聚类区域，用其边界框作为 SAM3 提示，生成精细 mask

        Args:
            image: PIL Image
            region_masks: 粗分割区域 mask 列表，每个 shape [H, W] bool
        Returns:
            results: 精细分割结果列表
        """
        results = []
        for region_mask in region_masks:
            # 从 mask 计算边界框
            ys, xs = np.where(region_mask)
            if len(xs) == 0:
                continue
            x1, y1, x2, y2 = xs.min(), ys.min(), xs.max(), ys.max()
            box = [float(x1), float(y1), float(x2), float(y2)]

            # 用边界框作为提示进行分割
            seg_results = self.segment_with_boxes(image, [box])
            if seg_results:
                results.append(seg_results[0])

        return results

    def get_image_embedding(self, image: Image.Image) -> Optional[torch.Tensor]:
        """
        获取 SAM3 图像编码器的特征嵌入（用于与 DINOv3 特征融合）

        Args:
            image: PIL Image
        Returns:
            embedding: Tensor 或 None
        """
        self.load_model()
        state = self.processor.set_image(image)
        # 从 state 中提取 backbone 输出的特征
        if "backbone_out" in state:
            backbone_out = state["backbone_out"]
            if "vision_features" in backbone_out:
                return backbone_out["vision_features"]
        return None
