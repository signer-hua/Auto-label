"""
DINOv3 特征提取器封装
负责：图像全局/局部特征提取、特征匹配、K-Means聚类
使用 DINOv3 ViT-S/16 (384维特征)
算法库已内置于 backend/libs/dinov3/ 中
"""
import sys
import torch
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional
from PIL import Image
from torchvision import transforms
from sklearn.cluster import KMeans

# 将内置的 dinov3 库加入 Python 路径
_LIBS_ROOT = Path(__file__).parent.parent / "libs"
_DINOV3_LIB = _LIBS_ROOT / "dinov3"
if str(_DINOV3_LIB) not in sys.path:
    sys.path.insert(0, str(_DINOV3_LIB))


class DINOFeatureExtractor:
    """
    DINOv3 特征提取器
    核心功能：
    1. 提取图像全局特征（CLS token）
    2. 提取图像局部 patch 特征（用于区域匹配）
    3. 余弦相似度特征匹配
    4. K-Means 聚类分割
    """

    def __init__(self, model_name: str = "dinov3_vits16", device: str = "cuda"):
        """
        初始化 DINOv3 特征提取器

        Args:
            model_name: 模型名称，默认 dinov3_vits16 (384维, 12层)
            device: 推理设备
        """
        self.device = device
        self.model_name = model_name
        self.model = None
        self.patch_size = 16  # ViT-S/16 的 patch 大小
        self.embed_dim = 384  # ViT-S/16 的特征维度

        # 图像预处理：标准 ImageNet 归一化
        self.transform = transforms.Compose([
            transforms.Resize(518, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(518),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

    def load_model(self):
        """加载 DINOv3 预训练模型"""
        if self.model is not None:
            return

        print(f"[DINOv3] 正在加载模型 {self.model_name}...")
        from dinov3.hub.backbones import dinov3_vits16
        self.model = dinov3_vits16(pretrained=True)
        self.model = self.model.to(self.device)
        self.model.eval()
        print(f"[DINOv3] 模型加载完成，设备: {self.device}")

    def preprocess(self, image: Image.Image) -> torch.Tensor:
        """
        图像预处理

        Args:
            image: PIL Image (RGB)
        Returns:
            tensor: [1, 3, 518, 518]
        """
        tensor = self.transform(image).unsqueeze(0).to(self.device)
        return tensor

    @torch.no_grad()
    def extract_global_feature(self, image: Image.Image) -> np.ndarray:
        """
        提取图像全局特征（CLS token）

        Args:
            image: PIL Image
        Returns:
            feature: numpy array, shape [384]
        """
        self.load_model()
        tensor = self.preprocess(image)
        # 前向推理，eval 模式返回 CLS token
        feature = self.model(tensor)  # [1, 384]
        feature = feature.cpu().numpy().flatten()
        # L2 归一化
        feature = feature / (np.linalg.norm(feature) + 1e-8)
        return feature

    @torch.no_grad()
    def extract_patch_features(self, image: Image.Image) -> Tuple[np.ndarray, int, int]:
        """
        提取图像所有 patch 的局部特征

        Args:
            image: PIL Image
        Returns:
            features: numpy array, shape [N_patches, 384]
            h_patches: patch 网格高度
            w_patches: patch 网格宽度
        """
        self.load_model()
        tensor = self.preprocess(image)

        # 使用 get_intermediate_layers 获取最后一层的 patch tokens
        outputs = self.model.get_intermediate_layers(
            tensor, n=1, reshape=True, return_class_token=False, norm=True
        )
        # outputs: tuple of [1, 384, H/P, W/P]
        patch_feat_map = outputs[0]  # [1, 384, h, w]
        _, c, h, w = patch_feat_map.shape
        # 转为 [N, 384]
        features = patch_feat_map.squeeze(0).reshape(c, -1).permute(1, 0)  # [h*w, 384]
        features = features.cpu().numpy()
        # L2 归一化
        norms = np.linalg.norm(features, axis=1, keepdims=True) + 1e-8
        features = features / norms
        return features, h, w

    @torch.no_grad()
    def extract_region_feature(self, image: Image.Image,
                                bbox: List[float]) -> np.ndarray:
        """
        提取边界框区域的特征（裁剪后提取全局特征）

        Args:
            image: PIL Image
            bbox: [x1, y1, x2, y2] 像素坐标
        Returns:
            feature: numpy array, shape [384]
        """
        x1, y1, x2, y2 = [int(v) for v in bbox]
        # 裁剪区域
        crop = image.crop((x1, y1, x2, y2))
        if crop.size[0] < 10 or crop.size[1] < 10:
            # 区域太小，扩展
            crop = crop.resize((64, 64), Image.BILINEAR)
        return self.extract_global_feature(crop)

    @torch.no_grad()
    def extract_mask_feature(self, image: Image.Image,
                              mask: np.ndarray) -> np.ndarray:
        """
        提取 mask 区域的平均 patch 特征

        Args:
            image: PIL Image
            mask: bool numpy array, shape [H, W]
        Returns:
            feature: numpy array, shape [384]
        """
        self.load_model()
        tensor = self.preprocess(image)

        outputs = self.model.get_intermediate_layers(
            tensor, n=1, reshape=True, return_class_token=False, norm=True
        )
        patch_feat_map = outputs[0]  # [1, 384, h, w]
        _, c, h, w = patch_feat_map.shape

        # 将 mask 缩放到 patch 网格大小
        mask_tensor = torch.from_numpy(mask.astype(np.float32)).unsqueeze(0).unsqueeze(0)
        mask_resized = torch.nn.functional.interpolate(
            mask_tensor, size=(h, w), mode='bilinear', align_corners=False
        )
        mask_resized = (mask_resized.squeeze() > 0.5).to(self.device)  # [h, w]

        # 提取 mask 区域的平均特征
        feat = patch_feat_map.squeeze(0)  # [384, h, w]
        if mask_resized.sum() > 0:
            masked_feat = feat[:, mask_resized]  # [384, N_masked]
            avg_feat = masked_feat.mean(dim=1)  # [384]
        else:
            avg_feat = feat.mean(dim=(1, 2))  # fallback: 全图平均

        avg_feat = avg_feat.cpu().numpy()
        avg_feat = avg_feat / (np.linalg.norm(avg_feat) + 1e-8)
        return avg_feat

    def cosine_similarity(self, feat_a: np.ndarray,
                           feat_b: np.ndarray) -> float:
        """计算两个特征向量的余弦相似度"""
        return float(np.dot(feat_a, feat_b))

    def batch_cosine_similarity(self, query: np.ndarray,
                                 gallery: np.ndarray) -> np.ndarray:
        """批量计算余弦相似度"""
        return gallery @ query  # 已 L2 归一化，点积即余弦相似度

    def match_features(self, template_features: List[np.ndarray],
                        patch_features: np.ndarray,
                        threshold: float = 0.8) -> np.ndarray:
        """
        特征模板匹配：找出与模板相似度超过阈值的 patch 区域

        Args:
            template_features: 模板特征列表，每个 shape [384]
            patch_features: 待匹配的 patch 特征，shape [N, 384]
            threshold: 相似度阈值
        Returns:
            match_mask: bool array, shape [N]，True 表示匹配
        """
        max_sim = np.zeros(len(patch_features))
        for template in template_features:
            sim = self.batch_cosine_similarity(template, patch_features)
            max_sim = np.maximum(max_sim, sim)
        return max_sim >= threshold

    def kmeans_cluster(self, patch_features: np.ndarray,
                        n_clusters: int = 10) -> np.ndarray:
        """
        对 patch 特征进行 K-Means 聚类

        Args:
            patch_features: shape [N, 384]
            n_clusters: 聚类数
        Returns:
            labels: shape [N]，每个 patch 的聚类标签
        """
        n_clusters = min(n_clusters, len(patch_features))
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(patch_features)
        return labels
