"""
标注流水线编排器
整合 YOLO-World、DINOv3、SAM3 三大模型，实现三种标注模式
"""
import numpy as np
from typing import List, Dict, Optional
from PIL import Image

from backend.core.yoloworld_detector import YOLOWorldDetector
from backend.core.dino_extractor import DINOFeatureExtractor
from backend.core.sam3_segmentor import SAM3Segmentor
from backend.utils.format_converter import masks_to_coco, masks_to_voc


class AnnotationPipeline:
    """
    标注流水线：三种模式的核心编排逻辑

    模式1: 文本提示一键自动标注
    模式2: 人工预标注 → 批量自动标注
    模式3: 选实例 → 跨图批量标注
    """

    def __init__(self, device: str = "cuda",
                 similarity_threshold: float = 0.8,
                 kmeans_clusters: int = 10):
        """
        初始化标注流水线

        Args:
            device: 推理设备
            similarity_threshold: 特征匹配阈值（模式2/3）
            kmeans_clusters: K-Means 聚类数（模式3）
        """
        self.device = device
        self.similarity_threshold = similarity_threshold
        self.kmeans_clusters = kmeans_clusters

        # 延迟初始化各模型
        self.detector = YOLOWorldDetector(device=device)
        self.feature_extractor = DINOFeatureExtractor(device=device)
        self.segmentor = SAM3Segmentor(device=device)

    # ==================== 模式1：文本提示一键自动标注 ====================
    def mode1_text_auto_annotate(self, image: Image.Image,
                                  text_prompts: List[str],
                                  score_thr: float = 0.3) -> Dict:
        """
        模式1：文本提示一键自动标注

        流程：
        ① YOLO-World 检测文本匹配的目标边界框
        ② DINOv3 提取边界框内局部特征（增强语义理解）
        ③ SAM3 以边界框为提示生成精准实例 mask
        ④ 整合输出 COCO 格式标注

        Args:
            image: PIL Image
            text_prompts: 文本提示列表，如 ["person", "car"]
            score_thr: 检测置信度阈值
        Returns:
            result: {
                "annotations": [{mask, box, label, score, dino_feature}, ...],
                "image_size": [W, H]
            }
        """
        self.detector.score_thr = score_thr

        # ① YOLO-World 文本驱动检测
        detections = self.detector.detect(image, text_prompts)
        if not detections:
            return {"annotations": [], "image_size": list(image.size)}

        # ② DINOv3 提取边界框内局部特征
        for det in detections:
            dino_feat = self.feature_extractor.extract_region_feature(
                image, det["box"]
            )
            det["dino_feature"] = dino_feat.tolist()

        # ③ SAM3 基于边界框生成精准 mask
        boxes = [det["box"] for det in detections]
        seg_results = self.segmentor.segment_with_boxes(image, boxes)

        # ④ 整合结果
        annotations = []
        for i, det in enumerate(detections):
            ann = {
                "box": det["box"],
                "label": det["label"],
                "label_id": det["label_id"],
                "score": det["score"],
                "dino_feature": det["dino_feature"],
            }
            if i < len(seg_results):
                ann["mask"] = seg_results[i]["mask"]
                ann["mask_score"] = seg_results[i]["score"]
            annotations.append(ann)

        return {
            "annotations": annotations,
            "image_size": list(image.size),
        }

    def mode1_batch(self, images: List[Image.Image],
                     text_prompts: List[str],
                     score_thr: float = 0.3) -> List[Dict]:
        """模式1 批量处理"""
        results = []
        for img in images:
            result = self.mode1_text_auto_annotate(img, text_prompts, score_thr)
            results.append(result)
        return results

    # ==================== 模式2：人工预标注 → 批量自动标注 ====================
    def mode2_manual_to_batch(self, ref_image: Image.Image,
                               user_boxes: List[List[float]],
                               target_images: List[Image.Image],
                               threshold: Optional[float] = None) -> Dict:
        """
        模式2：人工预标注 → 批量自动标注

        流程：
        ① SAM3 基于用户框选区域生成初始 mask
        ② DINOv3 提取 mask 区域全局特征，构建"特征模板库"
        ③ 遍历待标注图像，DINOv3 提取 patch 特征，余弦相似度匹配
        ④ SAM3 对匹配区域生成精准 mask

        Args:
            ref_image: 参考图像（用户在此图上框选）
            user_boxes: 用户框选的边界框列表 [[x1,y1,x2,y2], ...]
            target_images: 待标注的目标图像列表
            threshold: 相似度阈值，None 则使用默认值
        Returns:
            result: {
                "ref_annotations": [...],  # 参考图标注
                "target_annotations": [[...], ...],  # 每张目标图的标注
                "template_features": [...]  # 特征模板（可复用）
            }
        """
        if threshold is None:
            threshold = self.similarity_threshold

        # ① SAM3 基于用户框选生成初始 mask
        ref_seg_results = self.segmentor.segment_with_boxes(ref_image, user_boxes)

        # ② DINOv3 提取 mask 区域特征，构建特征模板库
        template_features = []
        ref_annotations = []
        for i, seg in enumerate(ref_seg_results):
            mask = seg["mask"]
            feat = self.feature_extractor.extract_mask_feature(ref_image, mask)
            template_features.append(feat)
            ref_annotations.append({
                "box": seg["box"],
                "mask": mask,
                "score": seg["score"],
                "template_id": i,
            })

        if not template_features:
            return {
                "ref_annotations": [],
                "target_annotations": [[] for _ in target_images],
                "template_features": [],
            }

        # ③④ 遍历目标图像，匹配 + 分割
        all_target_annotations = []
        for target_img in target_images:
            target_anns = self._match_and_segment(
                target_img, template_features, threshold
            )
            all_target_annotations.append(target_anns)

        return {
            "ref_annotations": ref_annotations,
            "target_annotations": all_target_annotations,
            "template_features": [f.tolist() for f in template_features],
        }

    def _match_and_segment(self, image: Image.Image,
                            template_features: List[np.ndarray],
                            threshold: float) -> List[Dict]:
        """
        在单张图像上执行特征匹配 + 分割

        Args:
            image: 目标图像
            template_features: 特征模板列表
            threshold: 相似度阈值
        Returns:
            annotations: 匹配到的标注列表
        """
        # 提取 patch 特征
        patch_feats, h, w = self.feature_extractor.extract_patch_features(image)

        # 特征匹配
        match_mask = self.feature_extractor.match_features(
            template_features, patch_feats, threshold
        )

        if not match_mask.any():
            return []

        # 将匹配的 patch 转为空间 mask
        match_map = match_mask.reshape(h, w)
        img_w, img_h = image.size

        # 上采样到原图大小
        import cv2
        match_map_full = cv2.resize(
            match_map.astype(np.uint8), (img_w, img_h),
            interpolation=cv2.INTER_NEAREST
        ).astype(bool)

        # 连通域分析，分离不同实例
        from scipy import ndimage
        labeled, num_features = ndimage.label(match_map_full)

        # 对每个连通域生成精细 mask
        region_masks = []
        for label_id in range(1, num_features + 1):
            region = (labeled == label_id)
            # 过滤太小的区域
            if region.sum() < 100:
                continue
            region_masks.append(region)

        # SAM3 精细分割
        seg_results = self.segmentor.segment_regions(image, region_masks)
        return seg_results

    # ==================== 模式3：选实例 → 跨图批量标注 ====================
    def mode3_cluster_and_select(self, image: Image.Image,
                                  n_clusters: Optional[int] = None) -> Dict:
        """
        模式3 第一步：全图聚类粗分割

        流程：
        ① DINOv3 提取全图 patch 特征
        ② K-Means 聚类
        ③ SAM3 基于聚类区域生成粗分割 mask

        Args:
            image: PIL Image
            n_clusters: 聚类数，None 则使用默认值
        Returns:
            result: {
                "cluster_masks": [mask1, mask2, ...],  # 每个聚类的 mask
                "cluster_features": [feat1, feat2, ...],  # 每个聚类的平均特征
                "image_size": [W, H]
            }
        """
        if n_clusters is None:
            n_clusters = self.kmeans_clusters

        # ① 提取 patch 特征
        patch_feats, h, w = self.feature_extractor.extract_patch_features(image)

        # ② K-Means 聚类
        labels = self.feature_extractor.kmeans_cluster(patch_feats, n_clusters)
        label_map = labels.reshape(h, w)

        img_w, img_h = image.size

        # ③ 对每个聚类生成 mask
        import cv2
        cluster_masks = []
        cluster_features = []

        for cluster_id in range(n_clusters):
            cluster_map = (label_map == cluster_id)
            # 上采样到原图大小
            cluster_map_full = cv2.resize(
                cluster_map.astype(np.uint8), (img_w, img_h),
                interpolation=cv2.INTER_NEAREST
            ).astype(bool)

            if cluster_map_full.sum() < 50:
                continue

            # 计算聚类平均特征
            cluster_patch_feats = patch_feats[labels == cluster_id]
            avg_feat = cluster_patch_feats.mean(axis=0)
            avg_feat = avg_feat / (np.linalg.norm(avg_feat) + 1e-8)

            cluster_masks.append(cluster_map_full)
            cluster_features.append(avg_feat)

        # SAM3 精细分割
        seg_results = self.segmentor.segment_regions(image, cluster_masks)

        # 合并结果
        final_masks = []
        final_features = []
        for i, seg in enumerate(seg_results):
            final_masks.append(seg["mask"])
            if i < len(cluster_features):
                final_features.append(cluster_features[i])

        return {
            "cluster_masks": final_masks,
            "cluster_features": [f.tolist() for f in final_features],
            "image_size": list(image.size),
        }

    def mode3_cross_image_annotate(self, selected_feature: np.ndarray,
                                     target_images: List[Image.Image],
                                     threshold: Optional[float] = None) -> List[List[Dict]]:
        """
        模式3 第二步：跨图批量标注

        用户选中实例后，用其特征在其他图像中匹配同类实例

        Args:
            selected_feature: 选中实例的特征向量 shape [384]
            target_images: 待标注图像列表
            threshold: 相似度阈值
        Returns:
            all_annotations: 每张图像的标注结果列表
        """
        if threshold is None:
            threshold = self.similarity_threshold

        template_features = [selected_feature]
        all_annotations = []
        for img in target_images:
            anns = self._match_and_segment(img, template_features, threshold)
            all_annotations.append(anns)

        return all_annotations
