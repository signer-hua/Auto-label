"""
DINOv3 单例引擎
负责：Mask 区域特征提取、全图 patch 特征匹配（余弦相似度）。
采用单例模式，Celery Worker 启动时预热加载到 GPU (float16)。

核心方法：
    - warmup(): Worker 启动时调用，加载模型到 GPU 并转为半精度
    - extract_mask_feature(image, mask): 提取 Mask 区域的平均特征向量
    - match_in_image(image, template_feature, threshold): 在目标图中匹配模板特征
"""
import sys
import gc
import torch
import numpy as np
from pathlib import Path
from typing import Optional
from PIL import Image
from torchvision import transforms

# 将内置 dinov3 库加入 Python 路径
_LIBS_ROOT = Path(__file__).parent.parent / "libs"
_DINOV3_LIB = _LIBS_ROOT / "dinov3"
if str(_DINOV3_LIB) not in sys.path:
    sys.path.insert(0, str(_DINOV3_LIB))

from backend.core.config import DINO_DEVICE, COSINE_SIMILARITY_THRESHOLD

import logging
logger = logging.getLogger(__name__)


class DINOEngine:
    """
    DINOv3 ViT-S/16 单例引擎。

    使用方式：
        engine = DINOEngine.get_instance()
        engine.warmup()
        feature = engine.extract_mask_feature(image, mask)
        matches = engine.match_in_image(target_image, feature, threshold=0.75)
    """

    _instance: Optional["DINOEngine"] = None

    def __init__(self):
        self.model = None
        self.device = DINO_DEVICE
        self.embed_dim = 384       # ViT-S/16 特征维度
        self.patch_size = 16
        self._warmed_up = False

        # 标准 ImageNet 预处理
        self.transform = transforms.Compose([
            transforms.Resize(518, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(518),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

    @classmethod
    def get_instance(cls) -> "DINOEngine":
        """获取全局单例"""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def warmup(self) -> None:
        """
        预热模型：加载 DINOv3 ViT-S/16 到 GPU，转为半精度。
        应在 Celery Worker 启动时调用一次。
        """
        if self._warmed_up:
            return

        logger.info("[DINOv3] Warming up model on %s (half precision)...", self.device)

        from dinov3.hub.backbones import dinov3_vits16
        self.model = dinov3_vits16(pretrained=True)
        self.model = self.model.to(self.device).half().eval()
        self._warmed_up = True
        logger.info("[DINOv3] Model warmed up successfully.")

    def _preprocess(self, image: Image.Image) -> torch.Tensor:
        """图像预处理，返回 [1, 3, 518, 518] 半精度 Tensor"""
        tensor = self.transform(image).unsqueeze(0).to(self.device).half()
        return tensor

    @torch.no_grad()
    def extract_mask_feature(
        self,
        image: Image.Image,
        mask: np.ndarray,
    ) -> np.ndarray:
        """
        提取 Mask 区域的平均 patch 特征向量。

        Args:
            image: PIL Image (RGB)
            mask: 布尔 numpy 数组，shape [H, W]

        Returns:
            feature: L2 归一化的特征向量，shape [384]
        """
        if not self._warmed_up:
            self.warmup()

        tensor = self._preprocess(image)

        # 获取最后一层 patch tokens: [1, 384, h, w]
        outputs = self.model.get_intermediate_layers(
            tensor, n=1, reshape=True, return_class_token=False, norm=True
        )
        patch_feat_map = outputs[0]  # [1, 384, h, w]
        _, c, h, w = patch_feat_map.shape

        # 将 mask 缩放到 patch 网格大小
        mask_t = torch.from_numpy(mask.astype(np.float32)).unsqueeze(0).unsqueeze(0)
        mask_resized = torch.nn.functional.interpolate(
            mask_t, size=(h, w), mode='bilinear', align_corners=False
        )
        mask_bool = (mask_resized.squeeze() > 0.5).to(self.device)

        feat = patch_feat_map.squeeze(0)  # [384, h, w]
        if mask_bool.sum() > 0:
            avg_feat = feat[:, mask_bool].mean(dim=1)  # [384]
        else:
            avg_feat = feat.mean(dim=(1, 2))  # fallback

        avg_feat = avg_feat.float().cpu().numpy()
        avg_feat = avg_feat / (np.linalg.norm(avg_feat) + 1e-8)
        return avg_feat

    @torch.no_grad()
    def extract_patch_features(
        self,
        image: Image.Image,
    ) -> tuple[np.ndarray, int, int]:
        """
        提取图像所有 patch 的局部特征。

        Args:
            image: PIL Image (RGB)

        Returns:
            features: L2 归一化的特征矩阵，shape [N_patches, 384]
            h_patches: patch 网格高度
            w_patches: patch 网格宽度
        """
        if not self._warmed_up:
            self.warmup()

        tensor = self._preprocess(image)
        outputs = self.model.get_intermediate_layers(
            tensor, n=1, reshape=True, return_class_token=False, norm=True
        )
        patch_feat_map = outputs[0]  # [1, 384, h, w]
        _, c, h, w = patch_feat_map.shape

        features = patch_feat_map.squeeze(0).reshape(c, -1).permute(1, 0)  # [h*w, 384]
        features = features.float().cpu().numpy()
        norms = np.linalg.norm(features, axis=1, keepdims=True) + 1e-8
        features = features / norms
        return features, h, w

    def match_in_image(
        self,
        image: Image.Image,
        template_feature: np.ndarray,
        threshold: float = COSINE_SIMILARITY_THRESHOLD,
    ) -> tuple[np.ndarray, list[list[float]]]:
        """
        在目标图像中匹配模板特征，返回匹配区域的 Mask 和 bbox 列表。

        流程：
        1. 提取目标图 patch 特征
        2. 计算余弦相似度
        3. 阈值过滤 → 空间 Mask
        4. 连通域分析 → 分离不同实例
        5. 返回每个实例的 bbox

        Args:
            image: 目标图像 PIL Image
            template_feature: 模板特征向量，shape [384]
            threshold: 余弦相似度阈值

        Returns:
            match_mask: 布尔数组，shape [H_img, W_img]，所有匹配区域
            bboxes: 每个连通域的 bbox [[x1,y1,x2,y2], ...]
        """
        import cv2
        from scipy import ndimage

        patch_feats, h, w = self.extract_patch_features(image)
        img_w, img_h = image.size

        # 余弦相似度（特征已 L2 归一化，点积即余弦）
        similarities = patch_feats @ template_feature  # [h*w]
        match_patches = (similarities >= threshold).reshape(h, w)

        if not match_patches.any():
            return np.zeros((img_h, img_w), dtype=bool), []

        # 上采样到原图大小
        match_full = cv2.resize(
            match_patches.astype(np.uint8), (img_w, img_h),
            interpolation=cv2.INTER_NEAREST
        ).astype(bool)

        # 连通域分析
        labeled, num_features = ndimage.label(match_full)
        bboxes = []
        for label_id in range(1, num_features + 1):
            region = (labeled == label_id)
            if region.sum() < 100:  # 过滤太小的区域
                continue
            ys, xs = np.where(region)
            bboxes.append([float(xs.min()), float(ys.min()),
                           float(xs.max()), float(ys.max())])

        return match_full, bboxes

    def release_memory(self) -> None:
        """释放 GPU 显存"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
            logger.info("[DINOv3] GPU memory cache cleared.")
