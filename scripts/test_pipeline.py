"""
Auto-label 验证测试脚本
分步验证三个模型和三种标注模式是否正常工作。

使用方法：
    cd Auto-label
    python scripts/test_pipeline.py --image path/to/test.jpg
"""
import sys
import argparse
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from PIL import Image
import numpy as np


def test_dino(image: Image.Image):
    """测试 DINOv3 特征提取"""
    print("\n" + "=" * 60)
    print("[测试1] DINOv3 ViT-S/16 特征提取")
    print("=" * 60)

    from backend.models.dino_engine import DINOEngine
    engine = DINOEngine.get_instance()
    engine.warmup()

    # Patch 特征
    t0 = time.time()
    patch_feats, h, w = engine.extract_patch_features(image)
    t1 = time.time()
    print(f"  Patch 特征: shape={patch_feats.shape}, "
          f"grid={h}x{w}, 耗时={t1 - t0:.2f}s")

    # K-Means 聚类
    t0 = time.time()
    cluster_map, instance_masks, h2, w2 = engine.extract_patch_features_clustering(image, n_clusters=5)
    t1 = time.time()
    print(f"  K-Means 聚类: {len(instance_masks)} 个有效实例, 耗时={t1 - t0:.2f}s")

    print("  [PASS] DINOv3 测试通过")
    return engine


def test_sam3(image: Image.Image):
    """测试 SAM3 分割"""
    print("\n" + "=" * 60)
    print("[测试2] SAM3 实例分割")
    print("=" * 60)

    from backend.models.sam_engine import SAMEngine
    engine = SAMEngine.get_instance()
    engine.warmup()

    img_w, img_h = image.size
    test_box = [img_w * 0.1, img_h * 0.1, img_w * 0.9, img_h * 0.9]

    t0 = time.time()
    mask = engine.generate_mask(image, test_box)
    t1 = time.time()
    print(f"  边界框提示分割: mask_shape={mask.shape}, "
          f"面积={mask.sum()}, 耗时={t1 - t0:.2f}s")

    print("  [PASS] SAM3 测试通过")
    return engine


def test_grounding_dino(image: Image.Image):
    """测试 Grounding DINO 检测"""
    print("\n" + "=" * 60)
    print("[测试3] Grounding DINO 文本驱动检测")
    print("=" * 60)

    from backend.models.grounding_dino_engine import GroundingDINOEngine
    engine = GroundingDINOEngine.get_instance()
    engine.warmup()

    text_prompts = ["person", "car", "dog", "cat"]
    t0 = time.time()
    detections = engine.detect(image, text_prompts)
    t1 = time.time()
    print(f"  文本提示: {text_prompts}")
    print(f"  检测到 {len(detections)} 个目标, 耗时={t1 - t0:.2f}s")
    for det in detections[:5]:
        print(f"    {det['label']}: score={det['score']:.3f}, "
              f"box={[round(v, 1) for v in det['box']]}")

    print("  [PASS] Grounding DINO 测试通过")
    return engine


def test_mode1_pipeline(image: Image.Image):
    """测试模式1完整流水线：Grounding DINO 检测 → SAM3 分割"""
    print("\n" + "=" * 60)
    print("[测试4] 完整标注流水线 - 模式1 (Grounding DINO + SAM3)")
    print("=" * 60)

    from backend.models.grounding_dino_engine import GroundingDINOEngine
    from backend.models.sam_engine import SAMEngine
    from backend.services.mask_utils import mask_to_bbox

    gd = GroundingDINOEngine.get_instance()
    sam = SAMEngine.get_instance()

    text_prompts = ["person", "car"]
    t0 = time.time()

    # Step 1: Grounding DINO 检测
    detections = gd.detect(image, text_prompts)
    print(f"  Step1 - Grounding DINO 检测: {len(detections)} 个目标")

    # Step 2: SAM3 生成 Mask
    masks = []
    for det in detections:
        mask = sam.generate_mask(image, det["box"])
        if mask.sum() >= 10:
            masks.append({
                "mask": mask,
                "label": det["label"],
                "score": det["score"],
                "bbox": mask_to_bbox(mask),
            })
    t1 = time.time()

    print(f"  Step2 - SAM3 分割: {len(masks)} 个有效 Mask")
    print(f"  总耗时: {t1 - t0:.2f}s")
    for m in masks[:5]:
        print(f"    {m['label']}: score={m['score']:.3f}, "
              f"bbox={[round(v, 1) for v in m['bbox']]}, "
              f"面积={m['mask'].sum()}")

    print("  [PASS] 模式1 流水线测试通过")


def main():
    parser = argparse.ArgumentParser(description="Auto-label 验证测试")
    parser.add_argument("--image", type=str, required=True, help="测试图像路径")
    parser.add_argument("--skip-grounding-dino", action="store_true",
                        help="跳过 Grounding DINO 测试")
    parser.add_argument("--skip-sam3", action="store_true",
                        help="跳过 SAM3 测试（首次需要下载权重）")
    parser.add_argument("--skip-dino", action="store_true",
                        help="跳过 DINOv3 测试")
    args = parser.parse_args()

    image_path = Path(args.image)
    if not image_path.exists():
        print(f"错误：图像文件不存在: {image_path}")
        sys.exit(1)

    image = Image.open(image_path).convert("RGB")
    print(f"测试图像: {image_path}, 尺寸: {image.size}")

    passed = 0
    total = 0

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

    if not args.skip_grounding_dino:
        total += 1
        try:
            test_grounding_dino(image)
            passed += 1
        except Exception as e:
            print(f"  [FAIL] Grounding DINO 测试失败: {e}")

    if passed == total and total > 0:
        total += 1
        try:
            test_mode1_pipeline(image)
            passed += 1
        except Exception as e:
            print(f"  [FAIL] 流水线测试失败: {e}")

    print("\n" + "=" * 60)
    print(f"测试结果: {passed}/{total} 通过")
    if passed == total:
        print("所有测试通过！")
    else:
        print("部分测试失败，请检查错误信息")
    print("=" * 60)


if __name__ == "__main__":
    main()
