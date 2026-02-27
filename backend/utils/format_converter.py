"""
标注格式转换工具
支持 COCO 和 VOC 格式导出
"""
import json
import os
import cv2
import numpy as np
from typing import List, Dict, Optional
from datetime import datetime
from pathlib import Path


def mask_to_rle(mask: np.ndarray) -> Dict:
    """
    将 bool mask 转换为 COCO RLE 编码

    Args:
        mask: bool numpy array, shape [H, W]
    Returns:
        rle: {"counts": [...], "size": [H, W]}
    """
    h, w = mask.shape
    flat = mask.flatten(order='F')  # Fortran order (列优先)
    # 计算 run-length
    diff = np.diff(np.concatenate([[0], flat, [0]]))
    starts = np.where(diff != 0)[0]
    counts = np.diff(starts).tolist()
    return {"counts": counts, "size": [h, w]}


def mask_to_polygon(mask: np.ndarray) -> List[List[float]]:
    """
    将 bool mask 转换为多边形坐标

    Args:
        mask: bool numpy array, shape [H, W]
    Returns:
        polygons: [[x1,y1,x2,y2,...], ...]
    """
    contours, _ = cv2.findContours(
        mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    polygons = []
    for contour in contours:
        if len(contour) < 3:
            continue
        polygon = contour.flatten().tolist()
        if len(polygon) >= 6:  # 至少3个点
            polygons.append(polygon)
    return polygons


def mask_to_bbox(mask: np.ndarray) -> List[float]:
    """
    从 mask 计算边界框 [x, y, w, h]

    Args:
        mask: bool numpy array
    Returns:
        bbox: [x, y, width, height]
    """
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return [0, 0, 0, 0]
    x1, y1, x2, y2 = xs.min(), ys.min(), xs.max(), ys.max()
    return [float(x1), float(y1), float(x2 - x1), float(y2 - y1)]


def masks_to_coco(annotations: List[Dict],
                   image_filename: str,
                   image_id: int = 1,
                   image_size: Optional[List[int]] = None,
                   category_map: Optional[Dict[str, int]] = None) -> Dict:
    """
    将标注结果转换为 COCO 格式

    Args:
        annotations: 标注列表，每个包含 {mask, box, label, score}
        image_filename: 图像文件名
        image_id: 图像 ID
        image_size: [width, height]
        category_map: 类别名 → ID 映射
    Returns:
        coco_dict: COCO 格式字典
    """
    if category_map is None:
        # 自动构建类别映射
        labels = set()
        for ann in annotations:
            if "label" in ann:
                labels.add(ann["label"])
        category_map = {label: i + 1 for i, label in enumerate(sorted(labels))}

    categories = [
        {"id": cat_id, "name": cat_name}
        for cat_name, cat_id in category_map.items()
    ]

    coco_annotations = []
    for idx, ann in enumerate(annotations):
        coco_ann = {
            "id": idx + 1,
            "image_id": image_id,
            "category_id": category_map.get(ann.get("label", "unknown"), 0),
            "score": ann.get("score", 1.0),
        }

        # 添加 mask（多边形格式）
        if "mask" in ann and ann["mask"] is not None:
            mask = ann["mask"]
            coco_ann["segmentation"] = mask_to_polygon(mask)
            coco_ann["bbox"] = mask_to_bbox(mask)
            coco_ann["area"] = float(mask.sum())
        elif "box" in ann:
            box = ann["box"]
            coco_ann["bbox"] = [box[0], box[1], box[2] - box[0], box[3] - box[1]]
            coco_ann["area"] = (box[2] - box[0]) * (box[3] - box[1])
            coco_ann["segmentation"] = []

        coco_ann["iscrowd"] = 0
        coco_annotations.append(coco_ann)

    w, h = image_size if image_size else [0, 0]
    coco_dict = {
        "info": {
            "description": "Auto-label 自动标注结果",
            "date_created": datetime.now().isoformat(),
            "version": "1.0",
        },
        "images": [{
            "id": image_id,
            "file_name": image_filename,
            "width": w,
            "height": h,
        }],
        "annotations": coco_annotations,
        "categories": categories,
    }
    return coco_dict


def masks_to_voc(annotations: List[Dict],
                  image_filename: str,
                  image_size: Optional[List[int]] = None) -> str:
    """
    将标注结果转换为 VOC XML 格式

    Args:
        annotations: 标注列表
        image_filename: 图像文件名
        image_size: [width, height]
    Returns:
        xml_str: VOC XML 字符串
    """
    w, h = image_size if image_size else [0, 0]

    xml_lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        '<annotation>',
        f'  <filename>{image_filename}</filename>',
        '  <source><database>Auto-label</database></source>',
        '  <size>',
        f'    <width>{w}</width>',
        f'    <height>{h}</height>',
        '    <depth>3</depth>',
        '  </size>',
    ]

    for ann in annotations:
        label = ann.get("label", "unknown")
        if "box" in ann:
            box = ann["box"]
            x1, y1, x2, y2 = box[0], box[1], box[2], box[3]
        elif "mask" in ann and ann["mask"] is not None:
            bbox = mask_to_bbox(ann["mask"])
            x1, y1 = bbox[0], bbox[1]
            x2, y2 = bbox[0] + bbox[2], bbox[1] + bbox[3]
        else:
            continue

        xml_lines.extend([
            '  <object>',
            f'    <name>{label}</name>',
            '    <pose>Unspecified</pose>',
            '    <truncated>0</truncated>',
            '    <difficult>0</difficult>',
            '    <bndbox>',
            f'      <xmin>{int(x1)}</xmin>',
            f'      <ymin>{int(y1)}</ymin>',
            f'      <xmax>{int(x2)}</xmax>',
            f'      <ymax>{int(y2)}</ymax>',
            '    </bndbox>',
            '  </object>',
        ])

    xml_lines.append('</annotation>')
    return '\n'.join(xml_lines)


def merge_coco_results(coco_results: List[Dict]) -> Dict:
    """
    合并多张图像的 COCO 标注结果

    Args:
        coco_results: COCO 格式字典列表
    Returns:
        merged: 合并后的 COCO 字典
    """
    merged = {
        "info": {
            "description": "Auto-label 批量标注结果",
            "date_created": datetime.now().isoformat(),
            "version": "1.0",
        },
        "images": [],
        "annotations": [],
        "categories": [],
    }

    # 收集所有类别
    all_categories = {}
    ann_id_counter = 1

    for coco in coco_results:
        for cat in coco.get("categories", []):
            if cat["name"] not in all_categories:
                all_categories[cat["name"]] = len(all_categories) + 1

        merged["images"].extend(coco.get("images", []))

        for ann in coco.get("annotations", []):
            ann["id"] = ann_id_counter
            ann_id_counter += 1
            # 更新 category_id
            for cat in coco.get("categories", []):
                if cat["id"] == ann.get("category_id"):
                    ann["category_id"] = all_categories.get(cat["name"], 0)
                    break
            merged["annotations"].append(ann)

    merged["categories"] = [
        {"id": cat_id, "name": cat_name}
        for cat_name, cat_id in all_categories.items()
    ]

    return merged
