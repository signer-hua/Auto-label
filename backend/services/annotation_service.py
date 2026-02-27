"""
标注业务服务层
封装三种标注模式的业务逻辑，处理结果序列化和文件导出
"""
import json
import numpy as np
from typing import List, Dict, Tuple, Optional
from pathlib import Path
from PIL import Image

from backend.core.pipeline import AnnotationPipeline
from backend.utils.format_converter import (
    masks_to_coco, masks_to_voc, merge_coco_results,
    mask_to_polygon, mask_to_bbox
)
from backend.config import OUTPUT_DIR, annotation_config


class AnnotationService:
    """标注业务服务"""

    def __init__(self, device: str = "cuda"):
        self.pipeline = AnnotationPipeline(
            device=device,
            similarity_threshold=annotation_config.similarity_threshold,
            kmeans_clusters=annotation_config.kmeans_clusters,
        )

    def _serialize_annotations(self, annotations: List[Dict]) -> List[Dict]:
        """
        将标注结果序列化为可 JSON 化的格式
        mask 转为多边形坐标，numpy 数组转为列表
        """
        serialized = []
        for ann in annotations:
            s_ann = {}
            for k, v in ann.items():
                if k == "mask" and isinstance(v, np.ndarray):
                    # mask → 多边形 + bbox
                    s_ann["segmentation"] = mask_to_polygon(v)
                    s_ann["bbox"] = mask_to_bbox(v)
                    s_ann["area"] = float(v.sum())
                elif isinstance(v, np.ndarray):
                    s_ann[k] = v.tolist()
                elif isinstance(v, (np.float32, np.float64)):
                    s_ann[k] = float(v)
                elif isinstance(v, (np.int32, np.int64)):
                    s_ann[k] = int(v)
                else:
                    s_ann[k] = v
            serialized.append(s_ann)
        return serialized

    def _save_result(self, result_dict: Dict, filename: str):
        """保存标注结果到输出目录"""
        output_path = OUTPUT_DIR / filename
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(result_dict, f, ensure_ascii=False, indent=2)
        return str(output_path)

    # ==================== 模式1 ====================
    def run_mode1(self, images: List[Tuple[str, Image.Image]],
                   text_prompts: List[str],
                   score_thr: float = 0.3,
                   export_format: str = "coco") -> Dict:
        """
        执行模式1：文本提示一键自动标注

        Args:
            images: [(文件名, PIL Image), ...]
            text_prompts: 文本提示列表
            score_thr: 检测阈值
            export_format: 导出格式
        Returns:
            API 响应字典
        """
        all_results = []
        coco_results = []

        for idx, (filename, image) in enumerate(images):
            result = self.pipeline.mode1_text_auto_annotate(
                image, text_prompts, score_thr
            )

            annotations = result["annotations"]
            serialized = self._serialize_annotations(annotations)

            # 构建 COCO 格式
            coco = masks_to_coco(
                annotations, filename,
                image_id=idx + 1,
                image_size=list(image.size),
            )
            coco_results.append(coco)

            all_results.append({
                "image": filename,
                "annotations": serialized,
                "count": len(serialized),
            })

        # 合并并保存
        if export_format == "coco":
            merged = merge_coco_results(coco_results)
            output_file = self._save_result(merged, "mode1_coco_result.json")
        else:
            # VOC 格式：每张图一个 XML
            for idx, (filename, image) in enumerate(images):
                xml_str = masks_to_voc(
                    all_results[idx]["annotations"] if idx < len(all_results) else [],
                    filename, list(image.size)
                )
                xml_name = Path(filename).stem + ".xml"
                (OUTPUT_DIR / xml_name).write_text(xml_str, encoding="utf-8")
            output_file = str(OUTPUT_DIR)

        return {
            "mode": "mode1_text_auto",
            "results": all_results,
            "total_annotations": sum(r["count"] for r in all_results),
            "export_path": output_file,
            "export_format": export_format,
        }

    # ==================== 模式2 ====================
    def run_mode2(self, ref_image: Tuple[str, Image.Image],
                   user_boxes: List[List[float]],
                   target_images: List[Tuple[str, Image.Image]],
                   threshold: float = 0.8,
                   export_format: str = "coco") -> Dict:
        """
        执行模式2：人工预标注 → 批量自动标注
        """
        ref_name, ref_img = ref_image
        target_imgs = [img for _, img in target_images]
        target_names = [name for name, _ in target_images]

        result = self.pipeline.mode2_manual_to_batch(
            ref_img, user_boxes, target_imgs, threshold
        )

        # 序列化参考图标注
        ref_anns = self._serialize_annotations(result["ref_annotations"])

        # 序列化目标图标注
        all_target_results = []
        coco_results = []

        for idx, (name, img) in enumerate(target_images):
            if idx < len(result["target_annotations"]):
                anns = result["target_annotations"][idx]
            else:
                anns = []
            serialized = self._serialize_annotations(anns)

            coco = masks_to_coco(
                anns, name,
                image_id=idx + 1,
                image_size=list(img.size),
            )
            coco_results.append(coco)

            all_target_results.append({
                "image": name,
                "annotations": serialized,
                "count": len(serialized),
            })

        # 保存
        if export_format == "coco":
            merged = merge_coco_results(coco_results)
            output_file = self._save_result(merged, "mode2_coco_result.json")
        else:
            output_file = str(OUTPUT_DIR)

        return {
            "mode": "mode2_manual_to_batch",
            "ref_annotations": ref_anns,
            "target_results": all_target_results,
            "total_annotations": sum(r["count"] for r in all_target_results),
            "template_features": result.get("template_features", []),
            "export_path": output_file,
            "export_format": export_format,
        }

    # ==================== 模式3 ====================
    def run_mode3_cluster(self, image: Tuple[str, Image.Image],
                           n_clusters: int = 10) -> Dict:
        """
        执行模式3第一步：全图聚类粗分割
        """
        name, img = image
        result = self.pipeline.mode3_cluster_and_select(img, n_clusters)

        # 序列化 cluster masks
        cluster_data = []
        for i, mask in enumerate(result["cluster_masks"]):
            polygons = mask_to_polygon(mask) if isinstance(mask, np.ndarray) else []
            bbox = mask_to_bbox(mask) if isinstance(mask, np.ndarray) else [0, 0, 0, 0]
            feature = result["cluster_features"][i] if i < len(result["cluster_features"]) else []

            cluster_data.append({
                "cluster_id": i,
                "segmentation": polygons,
                "bbox": bbox,
                "area": float(mask.sum()) if isinstance(mask, np.ndarray) else 0,
                "feature": feature,
            })

        return {
            "mode": "mode3_cluster",
            "image": name,
            "clusters": cluster_data,
            "cluster_count": len(cluster_data),
            "image_size": result["image_size"],
        }

    def run_mode3_annotate(self, selected_feature: np.ndarray,
                            target_images: List[Tuple[str, Image.Image]],
                            threshold: float = 0.8,
                            export_format: str = "coco") -> Dict:
        """
        执行模式3第二步：跨图批量标注
        """
        target_imgs = [img for _, img in target_images]
        target_names = [name for name, _ in target_images]

        all_annotations = self.pipeline.mode3_cross_image_annotate(
            selected_feature, target_imgs, threshold
        )

        all_results = []
        coco_results = []

        for idx, (name, img) in enumerate(target_images):
            anns = all_annotations[idx] if idx < len(all_annotations) else []
            serialized = self._serialize_annotations(anns)

            coco = masks_to_coco(
                anns, name,
                image_id=idx + 1,
                image_size=list(img.size),
            )
            coco_results.append(coco)

            all_results.append({
                "image": name,
                "annotations": serialized,
                "count": len(serialized),
            })

        if export_format == "coco":
            merged = merge_coco_results(coco_results)
            output_file = self._save_result(merged, "mode3_coco_result.json")
        else:
            output_file = str(OUTPUT_DIR)

        return {
            "mode": "mode3_cross_image",
            "results": all_results,
            "total_annotations": sum(r["count"] for r in all_results),
            "export_path": output_file,
            "export_format": export_format,
        }
