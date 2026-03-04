"""
DINOv3 单例引擎（增强版）
负责：Mask 区域特征提取、全图 patch 特征匹配（余弦相似度）。
采用单例模式，Celery Worker 启动时预热加载到 GPU (float16)。

v2 增强：
    - 多尺度特征融合（0.8x/1.0x/1.2x）
    - 背景抑制（裁剪目标区域 + padding）
    - 通道方差注意力加权
    - 肘部法则自动选择最优聚类数
    - 多实例特征融合（模式3 多选）
    - 动态余弦相似度阈值
"""
import sys
import gc
import torch
import numpy as np
from pathlib import Path
from typing import Optional
from PIL import Image
from torchvision import transforms

_LIBS_ROOT = Path(__file__).parent.parent / "libs"
_DINOV3_LIB = _LIBS_ROOT / "dinov3"
if str(_DINOV3_LIB) not in sys.path:
    sys.path.insert(0, str(_DINOV3_LIB))

from backend.core.config import (
    DINO_DEVICE, COSINE_SIMILARITY_THRESHOLD, DINO_CLUSTER_NUM,
    DINO_WEIGHTS_PATH, DINO_MULTI_SCALES, DINO_ELBOW_MAX_K,
    DINO_MIN_CLUSTER_RATIO, DINO_BG_PAD_RATIO,
    COSINE_SIM_LARGE_THRESH, COSINE_SIM_SMALL_THRESH, COSINE_SIM_AREA_CUTOFF,
)

import logging
logger = logging.getLogger(__name__)


class DINOEngine:
    """
    DINOv3 ViT-S/16 单例引擎（增强版）。

    增强特性：
        - 多尺度特征融合提升鲁棒性
        - 背景抑制减少噪声干扰
        - 通道方差注意力强化核心特征
        - 肘部法则自适应聚类数
        - 动态余弦相似度阈值
    """

    _instance: Optional["DINOEngine"] = None
    BASE_SIZE = 518

    def __init__(self):
        self.model = None
        self.device = DINO_DEVICE
        self.embed_dim = 384
        self.patch_size = 16
        self._warmed_up = False

        self.transform = transforms.Compose([
            transforms.Resize(self.BASE_SIZE, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(self.BASE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

    @classmethod
    def get_instance(cls) -> "DINOEngine":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def warmup(self) -> None:
        if self._warmed_up:
            return

        logger.info("[DINOv3] Warming up model on %s (half precision)...", self.device)

        from dinov3.hub.backbones import dinov3_vits16

        weights_path = Path(DINO_WEIGHTS_PATH) if DINO_WEIGHTS_PATH else None

        if weights_path and weights_path.exists():
            logger.info("[DINOv3] Loading weights from local: %s", weights_path)
            self.model = dinov3_vits16(pretrained=False)
            state_dict = torch.load(str(weights_path), map_location="cpu", weights_only=True)
            self.model.load_state_dict(state_dict, strict=False)
        else:
            logger.info("[DINOv3] Downloading weights from torch.hub...")
            self.model = dinov3_vits16(pretrained=True)

        self.model = self.model.to(self.device).half().eval()
        self._warmed_up = True
        logger.info("[DINOv3] Model warmed up successfully.")

    def _preprocess(self, image: Image.Image) -> torch.Tensor:
        tensor = self.transform(image).unsqueeze(0).to(self.device).half()
        return tensor

    def _build_transform_at_scale(self, target_size: int) -> transforms.Compose:
        """为指定输入尺寸构建预处理 pipeline"""
        return transforms.Compose([
            transforms.Resize(target_size, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(target_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

    def _preprocess_at_scale(self, image: Image.Image, scale: float) -> torch.Tensor:
        """按指定缩放比例预处理图像"""
        target_size = int(self.BASE_SIZE * scale)
        target_size = max(target_size, self.patch_size * 4)
        t = self._build_transform_at_scale(target_size)
        return t(image).unsqueeze(0).to(self.device).half()

    @torch.no_grad()
    def _channel_attention(self, feat_map: torch.Tensor) -> torch.Tensor:
        """
        通道方差注意力加权。
        方差大的通道包含更多判别性信息，给予更高权重。

        Args:
            feat_map: [1, C, H, W]
        Returns:
            加权后的 feat_map: [1, C, H, W]
        """
        channel_var = feat_map.squeeze(0).var(dim=(1, 2))  # [C]
        weights = torch.softmax(channel_var, dim=0)  # [C]
        weights = weights.unsqueeze(0).unsqueeze(2).unsqueeze(3)  # [1, C, 1, 1]
        return feat_map * weights

    @torch.no_grad()
    def _extract_features_single(self, image: Image.Image) -> tuple[torch.Tensor, int, int]:
        """单尺度特征提取，返回 [1, C, h, w] 和 patch 网格尺寸"""
        tensor = self._preprocess(image)
        outputs = self.model.get_intermediate_layers(
            tensor, n=1, reshape=True, return_class_token=False, norm=True
        )
        feat_map = outputs[0]
        _, c, h, w = feat_map.shape
        return feat_map, h, w

    @torch.no_grad()
    def extract_multilayer_features(
        self,
        image: Image.Image,
        layers: tuple[int, ...] = (9, 12),
    ) -> dict[int, np.ndarray]:
        """
        提取指定层的全局平均特征（MRF融合器使用）。

        Args:
            image: PIL Image (RGB)
            layers: 需要提取的层索引元组

        Returns:
            layer_features: {层索引: L2归一化特征向量 shape [C]}
        """
        if not self._warmed_up:
            self.warmup()

        tensor = self._preprocess(image)
        max_layer = max(layers)
        outputs = self.model.get_intermediate_layers(
            tensor, n=max_layer, reshape=True, return_class_token=False, norm=True
        )

        result = {}
        for layer_idx in layers:
            idx = min(layer_idx - 1, len(outputs) - 1)
            feat_map = outputs[idx]
            feat_map = self._channel_attention(feat_map)
            avg = feat_map.squeeze(0).mean(dim=(1, 2))
            avg_np = avg.float().cpu().numpy()
            avg_np = avg_np / (np.linalg.norm(avg_np) + 1e-8)
            result[layer_idx] = avg_np

        return result

    @torch.no_grad()
    def extract_multilayer_mask_features(
        self,
        image: Image.Image,
        mask: np.ndarray,
        layers: tuple[int, ...] = (9, 12),
    ) -> dict[int, np.ndarray]:
        """
        提取指定层的Mask区域特征（MRF融合器使用）。

        Args:
            image: PIL Image (RGB)
            mask: 布尔Mask [H, W]
            layers: 需要提取的层索引元组

        Returns:
            layer_features: {层索引: L2归一化特征向量 shape [C]}
        """
        if not self._warmed_up:
            self.warmup()

        tensor = self._preprocess(image)
        max_layer = max(layers)
        outputs = self.model.get_intermediate_layers(
            tensor, n=max_layer, reshape=True, return_class_token=False, norm=True
        )

        result = {}
        for layer_idx in layers:
            idx = min(layer_idx - 1, len(outputs) - 1)
            feat_map = outputs[idx]
            feat_map = self._channel_attention(feat_map)
            _, c, h, w = feat_map.shape

            mask_t = torch.from_numpy(mask.astype(np.float32)).unsqueeze(0).unsqueeze(0)
            mask_resized = torch.nn.functional.interpolate(
                mask_t, size=(h, w), mode='bilinear', align_corners=False
            )
            mask_bool = (mask_resized.squeeze() > 0.5).to(self.device)

            feat = feat_map.squeeze(0)
            if mask_bool.sum() > 0:
                avg = feat[:, mask_bool].mean(dim=1)
            else:
                avg = feat.mean(dim=(1, 2))

            avg_np = avg.float().cpu().numpy()
            avg_np = avg_np / (np.linalg.norm(avg_np) + 1e-8)
            result[layer_idx] = avg_np

        return result

    @torch.no_grad()
    def _extract_features_multiscale(
        self,
        image: Image.Image,
    ) -> tuple[torch.Tensor, int, int]:
        """
        多尺度特征融合：在 0.8x/1.0x/1.2x 三个尺度提取特征后
        插值到基准尺度大小并加权平均，提升特征鲁棒性。

        Returns:
            fused_feat_map: [1, C, h_base, w_base]
            h_base, w_base: 基准尺度 patch 网格大小
        """
        feat_maps = []
        base_h, base_w = None, None

        for scale in DINO_MULTI_SCALES:
            tensor = self._preprocess_at_scale(image, scale)
            outputs = self.model.get_intermediate_layers(
                tensor, n=1, reshape=True, return_class_token=False, norm=True
            )
            fm = outputs[0]  # [1, C, h, w]

            if scale == 1.0:
                base_h, base_w = fm.shape[2], fm.shape[3]

            feat_maps.append(fm)

        if base_h is None:
            base_h, base_w = feat_maps[0].shape[2], feat_maps[0].shape[3]

        aligned = []
        for fm in feat_maps:
            if fm.shape[2] != base_h or fm.shape[3] != base_w:
                fm = torch.nn.functional.interpolate(
                    fm, size=(base_h, base_w), mode='bilinear', align_corners=False
                )
            aligned.append(fm)

        fused = torch.stack(aligned, dim=0).mean(dim=0)  # [1, C, h, w]
        return fused, base_h, base_w

    @torch.no_grad()
    def extract_mask_feature(
        self,
        image: Image.Image,
        mask: np.ndarray,
        use_multiscale: bool = True,
        use_bg_suppress: bool = True,
    ) -> np.ndarray:
        """
        提取 Mask 区域的平均 patch 特征向量（增强版）。

        增强策略：
        1. 背景抑制：裁剪到 mask bbox + padding，排除背景干扰
        2. 多尺度特征融合
        3. 通道方差注意力加权
        4. L2 归一化

        Args:
            image: PIL Image (RGB)
            mask: 布尔 numpy 数组，shape [H, W]
            use_multiscale: 是否启用多尺度特征融合
            use_bg_suppress: 是否启用背景抑制

        Returns:
            feature: L2 归一化的特征向量，shape [384]
        """
        if not self._warmed_up:
            self.warmup()

        work_image = image
        work_mask = mask

        if use_bg_suppress:
            ys, xs = np.where(mask)
            if len(xs) > 0:
                img_w, img_h = image.size
                x1, y1 = int(xs.min()), int(ys.min())
                x2, y2 = int(xs.max()), int(ys.max())
                pad_x = int((x2 - x1) * DINO_BG_PAD_RATIO)
                pad_y = int((y2 - y1) * DINO_BG_PAD_RATIO)
                cx1 = max(0, x1 - pad_x)
                cy1 = max(0, y1 - pad_y)
                cx2 = min(img_w, x2 + pad_x)
                cy2 = min(img_h, y2 + pad_y)

                if (cx2 - cx1) > 32 and (cy2 - cy1) > 32:
                    work_image = image.crop((cx1, cy1, cx2, cy2))
                    work_mask = mask[cy1:cy2, cx1:cx2]

        if use_multiscale:
            patch_feat_map, h, w = self._extract_features_multiscale(work_image)
        else:
            patch_feat_map, h, w = self._extract_features_single(work_image)

        patch_feat_map = self._channel_attention(patch_feat_map)
        _, c, _, _ = patch_feat_map.shape

        mask_t = torch.from_numpy(work_mask.astype(np.float32)).unsqueeze(0).unsqueeze(0)
        mask_resized = torch.nn.functional.interpolate(
            mask_t, size=(h, w), mode='bilinear', align_corners=False
        )
        mask_bool = (mask_resized.squeeze() > 0.5).to(self.device)

        feat = patch_feat_map.squeeze(0)  # [C, h, w]
        if mask_bool.sum() > 0:
            avg_feat = feat[:, mask_bool].mean(dim=1)
        else:
            avg_feat = feat.mean(dim=(1, 2))

        avg_feat = avg_feat.float().cpu().numpy()
        avg_feat = avg_feat / (np.linalg.norm(avg_feat) + 1e-8)
        return avg_feat

    @torch.no_grad()
    def extract_multi_instance_feature(
        self,
        image: Image.Image,
        instance_masks: list[np.ndarray],
    ) -> np.ndarray:
        """
        多实例特征融合：提取多个实例的特征并取均值，生成统一目标特征模板。
        用于模式3 多实例合并标注。

        Args:
            image: PIL Image (RGB)
            instance_masks: 多个布尔 Mask 列表

        Returns:
            fused_feature: L2 归一化的融合特征向量，shape [384]
        """
        features = []
        for mask in instance_masks:
            feat = self.extract_mask_feature(image, mask)
            features.append(feat)

        if not features:
            raise ValueError("No valid instance masks for feature extraction")

        fused = np.mean(features, axis=0)
        fused = fused / (np.linalg.norm(fused) + 1e-8)
        return fused

    @torch.no_grad()
    def extract_multi_bbox_feature(
        self,
        image: Image.Image,
        bboxes: list[list[float]],
        sam_engine=None,
    ) -> np.ndarray:
        """
        多框选参考特征融合（模式2 多 bbox）。
        对每个 bbox 生成 Mask 后提取特征，取均值降低人工误差。

        Args:
            image: PIL Image (RGB)
            bboxes: 多个 bbox [[x1,y1,x2,y2], ...]
            sam_engine: SAMEngine 实例（可选，用于生成精准 Mask）

        Returns:
            fused_feature: L2 归一化的融合特征向量，shape [384]
        """
        features = []
        for bbox in bboxes:
            if sam_engine is not None:
                mask = sam_engine.generate_mask(image, bbox)
            else:
                img_w, img_h = image.size
                mask = np.zeros((img_h, img_w), dtype=bool)
                x1, y1, x2, y2 = [int(v) for v in bbox]
                mask[max(0, y1):min(img_h, y2), max(0, x1):min(img_w, x2)] = True

            if mask.sum() < 10:
                continue
            feat = self.extract_mask_feature(image, mask)
            features.append(feat)

        if not features:
            raise ValueError("No valid bboxes for feature extraction")

        fused = np.mean(features, axis=0)
        fused = fused / (np.linalg.norm(fused) + 1e-8)
        return fused

    @torch.no_grad()
    def extract_patch_features(
        self,
        image: Image.Image,
    ) -> tuple[np.ndarray, int, int]:
        """
        提取图像所有 patch 的局部特征（含通道注意力加权）。

        Returns:
            features: L2 归一化的特征矩阵，shape [N_patches, 384]
            h_patches, w_patches: patch 网格尺寸
        """
        if not self._warmed_up:
            self.warmup()

        patch_feat_map, h, w = self._extract_features_single(image)
        patch_feat_map = self._channel_attention(patch_feat_map)
        _, c, _, _ = patch_feat_map.shape

        features = patch_feat_map.squeeze(0).reshape(c, -1).permute(1, 0)
        features = features.float().cpu().numpy()
        norms = np.linalg.norm(features, axis=1, keepdims=True) + 1e-8
        features = features / norms
        return features, h, w

    def _compute_dynamic_threshold(
        self,
        area_ratio: float,
    ) -> float:
        """
        根据目标面积比自适应调整余弦相似度阈值。
        大目标特征稳定 → 阈值略高；小目标特征稀疏 → 阈值略低提升召回。

        Args:
            area_ratio: 目标面积占图片总面积的比例

        Returns:
            threshold: 自适应余弦相似度阈值
        """
        if area_ratio >= COSINE_SIM_AREA_CUTOFF:
            return COSINE_SIM_LARGE_THRESH
        t = area_ratio / COSINE_SIM_AREA_CUTOFF
        return COSINE_SIM_SMALL_THRESH + t * (COSINE_SIM_LARGE_THRESH - COSINE_SIM_SMALL_THRESH)

    def match_in_image(
        self,
        image: Image.Image,
        template_feature: np.ndarray,
        threshold: float = COSINE_SIMILARITY_THRESHOLD,
        ref_area_ratio: float | None = None,
        return_similarities: bool = False,
    ) -> tuple[np.ndarray, list[list[float]]] | tuple[np.ndarray, list[list[float]], list[float]]:
        """
        在目标图像中匹配模板特征（增强版：动态阈值 + 自适应小区域过滤）。

        Args:
            image: 目标图像
            template_feature: 模板特征向量
            threshold: 余弦相似度阈值
            ref_area_ratio: 参考目标面积比（动态阈值）
            return_similarities: 是否返回每个匹配区域的平均相似度

        Returns:
            match_mask, bboxes[, sim_values]
        """
        import cv2
        from scipy import ndimage
        from backend.utils.threshold_utils import adaptive_change_threshold

        if ref_area_ratio is not None:
            threshold = self._compute_dynamic_threshold(ref_area_ratio)

        patch_feats, h, w = self.extract_patch_features(image)
        img_w, img_h = image.size

        similarities = patch_feats @ template_feature

        act_threshold = adaptive_change_threshold(
            similarities, fallback_threshold=threshold,
        )
        threshold = max(threshold * 0.85, min(act_threshold, threshold * 1.1))

        sim_grid = similarities.reshape(h, w)
        match_patches = (sim_grid >= threshold)

        if not match_patches.any():
            if return_similarities:
                return np.zeros((img_h, img_w), dtype=bool), [], []
            return np.zeros((img_h, img_w), dtype=bool), []

        match_full = cv2.resize(
            match_patches.astype(np.uint8), (img_w, img_h),
            interpolation=cv2.INTER_NEAREST
        ).astype(bool)

        labeled, num_features = ndimage.label(match_full)
        bboxes = []
        sim_values = []

        label_grid = cv2.resize(
            ndimage.label(match_patches.astype(np.uint8))[0].astype(np.uint8),
            (img_w, img_h), interpolation=cv2.INTER_NEAREST
        ) if return_similarities else None

        min_area = max(50, int(img_w * img_h / 10000))

        for label_id in range(1, num_features + 1):
            region = (labeled == label_id)
            if region.sum() < min_area:
                continue
            ys, xs = np.where(region)
            bboxes.append([float(xs.min()), float(ys.min()),
                           float(xs.max()), float(ys.max())])

            if return_similarities:
                region_sim = cv2.resize(
                    sim_grid.astype(np.float32), (img_w, img_h),
                    interpolation=cv2.INTER_LINEAR
                )
                avg_sim = float(region_sim[region].mean())
                sim_values.append(avg_sim)

        if return_similarities:
            return match_full, bboxes, sim_values
        return match_full, bboxes

    @torch.no_grad()
    def extract_patch_features_clustering(
        self,
        image: Image.Image,
        n_clusters: int | None = None,
        use_elbow: bool = True,
    ) -> tuple[np.ndarray, list[np.ndarray], int, int]:
        """
        全图 patch 特征实例分割（ProMerge 谱聚类优先，K-Means 兜底）。

        ProMerge 策略：
        1. 计算 patch 间余弦亲和矩阵
        2. 基于亲和矩阵做谱聚类（Normalized Cut）
        3. 用 scikit-image regionprops 后处理，过滤面积 < 1% 的碎片
        4. 若 ProMerge 失败，自动降级到 K-Means 聚类

        Returns:
            cluster_map, instance_masks, h_patches, w_patches
        """
        import cv2

        if not self._warmed_up:
            self.warmup()

        img_w, img_h = image.size
        patch_feats, h, w = self.extract_patch_features(image)

        try:
            cluster_map, instance_masks = self._promerge_segmentation(
                patch_feats, h, w, img_w, img_h
            )
            logger.info("[DINOv3] ProMerge: %d valid instances", len(instance_masks))
        except Exception as e:
            logger.warning("[DINOv3] ProMerge failed (%s), falling back to K-Means", str(e))
            cluster_map, instance_masks = self._kmeans_segmentation(
                patch_feats, h, w, img_w, img_h, n_clusters, use_elbow
            )

        return cluster_map, instance_masks, h, w

    def _promerge_segmentation(
        self,
        patch_feats: np.ndarray,
        h: int, w: int,
        img_w: int, img_h: int,
    ) -> tuple[np.ndarray, list[np.ndarray]]:
        """
        ProMerge 谱聚类实例分割：基于 patch 亲和矩阵的 Normalized Cut。

        步骤：
        1. 计算 patch 余弦相似度亲和矩阵 A[i,j]
        2. 对 A 做阈值过滤（去除弱连接，保留强亲和）
        3. 基于 scipy 稀疏矩阵做谱聚类（sklearn SpectralClustering）
        4. scikit-image regionprops 过滤碎片实例
        """
        from sklearn.cluster import SpectralClustering
        from skimage.measure import label as sk_label, regionprops
        import cv2

        n_patches = patch_feats.shape[0]

        affinity = patch_feats @ patch_feats.T
        np.fill_diagonal(affinity, 0)

        threshold = float(np.percentile(affinity[affinity > 0], 50))
        affinity[affinity < threshold] = 0

        max_k = min(DINO_ELBOW_MAX_K, n_patches // 4)
        n_clusters = max(2, min(max_k, 8))

        sc = SpectralClustering(
            n_clusters=n_clusters,
            affinity='precomputed',
            assign_labels='kmeans',
            random_state=42,
            n_init=5,
        )
        labels = sc.fit_predict(affinity)
        label_grid = labels.reshape(h, w)

        cluster_map = cv2.resize(
            label_grid.astype(np.uint8), (img_w, img_h),
            interpolation=cv2.INTER_NEAREST
        )

        instance_masks = []
        total_pixels = img_w * img_h
        min_area_ratio = 0.01

        for cid in range(n_clusters):
            binary = (cluster_map == cid).astype(np.uint8)
            labeled = sk_label(binary, connectivity=2)
            for region in regionprops(labeled):
                if region.area / total_pixels < min_area_ratio:
                    continue
                mask = (labeled == region.label).astype(bool)
                instance_masks.append(mask)

        return cluster_map, instance_masks

    def _kmeans_segmentation(
        self,
        patch_feats: np.ndarray,
        h: int, w: int,
        img_w: int, img_h: int,
        n_clusters: int | None,
        use_elbow: bool,
    ) -> tuple[np.ndarray, list[np.ndarray]]:
        """K-Means 聚类兜底方案"""
        from sklearn.cluster import KMeans
        import cv2

        if n_clusters is None and use_elbow:
            n_clusters = self._elbow_optimal_k(patch_feats)
        elif n_clusters is None:
            n_clusters = DINO_CLUSTER_NUM

        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10, max_iter=100)
        labels = kmeans.fit_predict(patch_feats)
        label_grid = labels.reshape(h, w)

        cluster_map = cv2.resize(
            label_grid.astype(np.uint8), (img_w, img_h),
            interpolation=cv2.INTER_NEAREST
        )

        instance_masks = []
        total_pixels = img_w * img_h
        for cid in range(n_clusters):
            mask = (cluster_map == cid)
            if mask.sum() / total_pixels >= DINO_MIN_CLUSTER_RATIO:
                instance_masks.append(mask)

        logger.info("[DINOv3] K-Means fallback: %d clusters → %d valid instances",
                    n_clusters, len(instance_masks))
        return cluster_map, instance_masks

    def _elbow_optimal_k(self, features: np.ndarray) -> int:
        """
        肘部法则自动选择最优聚类数。
        计算 K=2~ELBOW_MAX_K 的惯性，找到惯性下降速率变化最大的拐点。

        Args:
            features: patch 特征矩阵 [N, 384]

        Returns:
            optimal_k: 最优聚类数
        """
        from sklearn.cluster import KMeans

        max_k = min(DINO_ELBOW_MAX_K, len(features) // 2)
        if max_k < 3:
            return DINO_CLUSTER_NUM

        inertias = []
        k_range = range(2, max_k + 1)

        for k in k_range:
            km = KMeans(n_clusters=k, random_state=42, n_init=5, max_iter=50)
            km.fit(features)
            inertias.append(km.inertia_)

        if len(inertias) < 3:
            return DINO_CLUSTER_NUM

        diffs = np.diff(inertias)
        diffs2 = np.diff(diffs)

        optimal_idx = np.argmax(diffs2) + 2
        optimal_k = list(k_range)[optimal_idx] if optimal_idx < len(k_range) else DINO_CLUSTER_NUM

        optimal_k = max(2, min(optimal_k, max_k))
        logger.info("[DINOv3] Elbow method: optimal K=%d (range 2~%d)", optimal_k, max_k)
        return optimal_k

    def fuse_multi_ref_features(
        self,
        ref_features: list[np.ndarray],
        weights: list[float] | None = None,
    ) -> np.ndarray:
        """
        多参考图特征加权融合。
        对来自不同参考图的特征向量按权重加权平均后 L2 归一化。

        Args:
            ref_features: 各参考图特征向量列表，每个 shape [384]
            weights: 权重列表（自动归一化），None 则均分

        Returns:
            fused: L2 归一化的融合特征，shape [384]
        """
        if not ref_features:
            raise ValueError("No reference features to fuse")

        if len(ref_features) == 1:
            return ref_features[0]

        if weights is None:
            weights = [1.0] * len(ref_features)

        total_w = sum(weights)
        norm_w = [w / total_w for w in weights]

        fused = np.zeros_like(ref_features[0])
        for feat, w in zip(ref_features, norm_w):
            fused += feat * w

        fused = fused / (np.linalg.norm(fused) + 1e-8)
        logger.info("[DINOv3] Fused %d ref features with weights %s",
                    len(ref_features), [round(w, 2) for w in norm_w])
        return fused

    @torch.no_grad()
    def extract_instance_feature(
        self,
        image: Image.Image,
        instance_mask: np.ndarray,
    ) -> np.ndarray:
        """提取指定实例 Mask 区域的全局特征模板（模式3 选中实例后调用）。"""
        return self.extract_mask_feature(image, instance_mask)

    def release_memory(self) -> None:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
            logger.info("[DINOv3] GPU memory cache cleared.")
