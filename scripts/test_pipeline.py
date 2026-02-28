"""
Auto-label 验证测试脚本
分步验证三个模型和三种标注模式是否正常工作
使用方法：
    cd Auto-label
    python scripts/test_pipeline.py --image path/to/test.jpg
"""
import sys
import os
import argparse
import time
from pathlib import Path

# 确保项目根目录在 Python 路径中
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from PIL import Image
import numpy as np


def test_dino(image: Image.Image):
    """测试 DINOv3 特征提取"""
    print("\n" + "=" * 60)
    print("[测试1] DINOv3 ViT-S/16 特征提取")
    print("=" * 60)

    from backend.core.dino_extractor import DINOFeatureExtractor
    extractor = DINOFeatureExtractor(device="cuda")

    # 全局特征
    t0 = time.time()
    global_feat = extractor.extract_global_feature(image)
    t1 = time.time()
    print(f"  全局特征 (CLS token): shape={global_feat.shape}, "
          f"norm={np.linalg.norm(global_feat):.4f}, 耗时={t1 - t0:.2f}s")

    # Patch 特征
    t0 = time.time()
    patch_feats, h, w = extractor.extract_patch_features(image)
    t1 = time.time()
    print(f"  Patch 特征: shape={patch_feats.shape}, "
          f"grid={h}x{w}, 耗时={t1 - t0:.2f}s")

    # K-Means 聚类
    t0 = time.time()
    labels = extractor.kmeans_cluster(patch_feats, n_clusters=5)
    t1 = time.time()
    print(f"  K-Means 聚类: {len(set(labels))} 个簇, 耗时={t1 - t0:.2f}s")

    print("  [PASS] DINOv3 测试通过 ✓")
    return extractor


def test_sam3(image: Image.Image):
    """测试 SAM3 分割"""
    print("\n" + "=" * 60)
    print("[测试2] SAM3 实例分割")
    print("=" * 60)

    from backend.core.sam3_segmentor import SAM3Segmentor
    segmentor = SAM3Segmentor(device="cuda")

    # 文本提示分割
    t0 = time.time()
    results = segmentor.segment_with_text(image, "object")
    t1 = time.time()
    print(f"  文本提示分割: 检测到 {len(results)} 个实例, 耗时={t1 - t0:.2f}s")
    for i, r in enumerate(results[:3]):
        mask = r["mask"]
        print(f"    实例{i}: mask_shape={mask.shape}, "
              f"面积={mask.sum()}, score={r['score']:.3f}")

    # 边界框提示分割
    img_w, img_h = image.size
    test_box = [img_w * 0.1, img_h * 0.1, img_w * 0.9, img_h * 0.9]
    t0 = time.time()
    results = segmentor.segment_with_boxes(image, [test_box])
    t1 = time.time()
    print(f"  边界框提示分割: {len(results)} 个结果, 耗时={t1 - t0:.2f}s")
    if results:
        print(f"    mask_shape={results[0]['mask'].shape}, "
              f"score={results[0]['score']:.3f}")

    print("  [PASS] SAM3 测试通过 ✓")
    return segmentor


def test_yoloworld(image: Image.Image):
    """测试 YOLO-World 检测"""
    print("\n" + "=" * 60)
    print("[测试3] YOLO-World 文本驱动检测")
    print("=" * 60)

    from backend.core.yoloworld_detector import YOLOWorldDetector
    detector = YOLOWorldDetector(device="cuda")

    text_prompts = ["person", "car", "dog", "cat"]
    t0 = time.time()
    detections = detector.detect(image, text_prompts)
    t1 = time.time()
    print(f"  文本提示: {text_prompts}")
    print(f"  检测到 {len(detections)} 个目标, 耗时={t1 - t0:.2f}s")
    for det in detections[:5]:
        print(f"    {det['label']}: score={det['score']:.3f}, "
              f"box={[round(v, 1) for v in det['box']]}")

    print("  [PASS] YOLO-World 测试通过 ✓")
    return detector


def test_pipeline(image: Image.Image):
    """测试完整标注流水线"""
    print("\n" + "=" * 60)
    print("[测试4] 完整标注流水线 - 模式1")
    print("=" * 60)

    from backend.core.pipeline import AnnotationPipeline
    pipeline = AnnotationPipeline(device="cuda")

    text_prompts = ["person", "car"]
    t0 = time.time()
    result = pipeline.mode1_text_auto_annotate(image, text_prompts, score_thr=0.3)
    t1 = time.time()
    anns = result["annotations"]
    print(f"  文本提示: {text_prompts}")
    print(f"  标注结果: {len(anns)} 个实例, 耗时={t1 - t0:.2f}s")
    for ann in anns[:5]:
        has_mask = "mask" in ann and ann["mask"] is not None
        print(f"    {ann['label']}: score={ann['score']:.3f}, "
              f"has_mask={has_mask}")

    print("  [PASS] 流水线测试通过 ✓")

    # 测试 COCO 格式导出
    from backend.utils.format_converter import masks_to_coco
    coco = masks_to_coco(anns, "test.jpg", image_size=list(image.size))
    print(f"\n  COCO 导出: {len(coco['annotations'])} 个标注, "
          f"{len(coco['categories'])} 个类别")
    print("  [PASS] 格式导出测试通过 ✓")


def main():
    parser = argparse.ArgumentParser(description="Auto-label 验证测试")
    parser.add_argument("--image", type=str, required=True,
                        help="测试图像路径")
    parser.add_argument("--skip-yolo", action="store_true",
                        help="跳过 YOLO-World 测试（需要权重文件）")
    parser.add_argument("--skip-sam3", action="store_true",
                        help="跳过 SAM3 测试（首次需要下载权重）")
    parser.add_argument("--skip-dino", action="store_true",
                        help="跳过 DINOv3 测试")
    args = parser.parse_args()

    # 加载测试图像
    image_path = Path(args.image)
    if not image_path.exists():
        print(f"错误：图像文件不存在: {image_path}")
        sys.exit(1)

    image = Image.open(image_path).convert("RGB")
    print(f"测试图像: {image_path}, 尺寸: {image.size}")

    passed = 0
    total = 0

    # 逐步测试
    if not args.skip_dino:
        total += 1
        try:
            test_dino(image)
            passed += 1
        except Exception as e:
            print(f"  [FAIL] DINOv3 测试失败: {e}")

    if not args.skip_sam3:
        total += 1
        try:
            test_sam3(image)
            passed += 1
        except Exception as e:
            print(f"  [FAIL] SAM3 测试失败: {e}")

    if not args.skip_yolo:
        total += 1
        try:
            test_yoloworld(image)
            passed += 1
        except Exception as e:
            print(f"  [FAIL] YOLO-World 测试失败: {e}")

    # 完整流水线测试（需要所有模型都通过）
    if passed == total and total > 0:
        total += 1
        try:
            test_pipeline(image)
            passed += 1
        except Exception as e:
            print(f"  [FAIL] 流水线测试失败: {e}")

    # 总结
    print("\n" + "=" * 60)
    print(f"测试结果: {passed}/{total} 通过")
    if passed == total:
        print("所有测试通过！✓")
    else:
        print("部分测试失败，请检查错误信息")
    print("=" * 60)


if __name__ == "__main__":
    main()
