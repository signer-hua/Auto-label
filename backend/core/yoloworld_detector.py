"""
YOLO-World 检测器封装
负责：基于文本提示的开放词汇目标检测
使用 YOLO-World-v2-S (Small) 模型
"""
import sys
import os
import cv2
import torch
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple
from PIL import Image

# 将 YOLO-World 加入 Python 路径
YOLOWORLD_PATH = Path(__file__).parent.parent.parent.parent / "YOLO-World"
if str(YOLOWORLD_PATH) not in sys.path:
    sys.path.insert(0, str(YOLOWORLD_PATH))


class YOLOWorldDetector:
    """
    YOLO-World 开放词汇目标检测器
    核心功能：
    1. 基于文本提示检测图像中的目标
    2. 输出边界框 + 类别 + 置信度
    """

    def __init__(self, config_path: str = "", weights_path: str = "",
                 device: str = "cuda", score_thr: float = 0.3,
                 nms_thr: float = 0.7):
        """
        初始化 YOLO-World 检测器

        Args:
            config_path: 模型配置文件路径
            weights_path: 预训练权重路径
            device: 推理设备
            score_thr: 检测置信度阈值
            nms_thr: NMS IoU 阈值
        """
        self.device = device
        self.score_thr = score_thr
        self.nms_thr = nms_thr
        self.model = None
        self.test_pipeline = None

        # 默认配置路径
        if not config_path:
            self.config_path = str(
                YOLOWORLD_PATH / "configs" / "pretrain" /
                "yolo_world_v2_s_vlpan_bn_2e-3_100e_4x8gpus_obj365v1_goldg_train_lvis_minival.py"
            )
        else:
            self.config_path = config_path

        self.weights_path = weights_path

    def load_model(self):
        """加载 YOLO-World 预训练模型"""
        if self.model is not None:
            return

        print("[YOLO-World] 正在加载模型...")

        from mmengine.config import Config
        from mmengine.dataset import Compose
        from mmdet.apis import init_detector
        from mmdet.utils import get_test_pipeline_cfg

        cfg = Config.fromfile(self.config_path)

        # 如果指定了权重路径
        if self.weights_path:
            cfg.load_from = self.weights_path

        self.model = init_detector(
            cfg,
            checkpoint=self.weights_path if self.weights_path else None,
            device=self.device,
        )

        # 构建测试 pipeline
        test_pipeline_cfg = get_test_pipeline_cfg(cfg=cfg)
        test_pipeline_cfg[0].type = 'mmdet.LoadImageFromNDArray'
        self.test_pipeline = Compose(test_pipeline_cfg)

        print(f"[YOLO-World] 模型加载完成，设备: {self.device}")

    @torch.no_grad()
    def detect(self, image: Image.Image,
               text_prompts: List[str]) -> List[Dict]:
        """
        基于文本提示检测图像中的目标

        Args:
            image: PIL Image (RGB)
            text_prompts: 文本提示列表，如 ["person", "bus", "car"]
        Returns:
            detections: 列表，每个元素包含 {box, label, score, label_id}
                box: [x1, y1, x2, y2] 像素坐标
                label: 文本标签
                score: 置信度
                label_id: 类别索引
        """
        self.load_model()

        # 转换为 numpy (RGB)
        img_array = np.array(image)

        # 构建文本格式：[[类别1], [类别2], ..., [' ']]
        texts = [[t] for t in text_prompts] + [[' ']]

        # 构建数据
        data_info = dict(img=img_array, img_id=0, texts=texts)
        data_info = self.test_pipeline(data_info)
        data_batch = dict(
            inputs=data_info['inputs'].unsqueeze(0).to(self.device),
            data_samples=[data_info['data_samples']]
        )

        # 推理
        output = self.model.test_step(data_batch)[0]
        pred_instances = output.pred_instances

        # 过滤低置信度
        keep = pred_instances.scores.float() > self.score_thr
        pred_instances = pred_instances[keep]

        # 转换结果
        detections = []
        if len(pred_instances) > 0:
            bboxes = pred_instances.bboxes.cpu().numpy()
            labels = pred_instances.labels.cpu().numpy()
            scores = pred_instances.scores.cpu().numpy()

            for i in range(len(bboxes)):
                label_id = int(labels[i])
                label_text = text_prompts[label_id] if label_id < len(text_prompts) else "unknown"
                detections.append({
                    "box": bboxes[i].tolist(),  # [x1, y1, x2, y2]
                    "label": label_text,
                    "score": float(scores[i]),
                    "label_id": label_id,
                })

        return detections

    @torch.no_grad()
    def detect_batch(self, images: List[Image.Image],
                      text_prompts: List[str]) -> List[List[Dict]]:
        """
        批量检测多张图像

        Args:
            images: PIL Image 列表
            text_prompts: 文本提示列表
        Returns:
            all_detections: 每张图像的检测结果列表
        """
        all_detections = []
        for image in images:
            dets = self.detect(image, text_prompts)
            all_detections.append(dets)
        return all_detections
