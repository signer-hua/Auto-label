/**
 * 图片自适应渲染工具类
 * 封装图片等比缩放、居中偏移计算逻辑。
 *
 * 规则：
 *   - 图片宽高比 > 画布宽高比：按宽度缩放至画布宽度，高度自适应，垂直居中
 *   - 图片宽高比 < 画布宽高比：按高度缩放至画布高度，宽度自适应，水平居中
 *   - 图片尺寸 < 画布尺寸：原图尺寸居中，不放大
 */

export interface FitResult {
  /** 缩放比例（画布像素 / 原图像素） */
  scale: number;
  /** 图片在画布中的 X 偏移（居中） */
  offsetX: number;
  /** 图片在画布中的 Y 偏移（居中） */
  offsetY: number;
  /** 缩放后图片显示宽度 */
  displayWidth: number;
  /** 缩放后图片显示高度 */
  displayHeight: number;
}

/**
 * 计算图片在容器中的自适应渲染参数
 *
 * @param imgWidth   原图宽度（像素）
 * @param imgHeight  原图高度（像素）
 * @param containerW 画布容器宽度（像素）
 * @param containerH 画布容器高度（像素）
 * @param padding    内边距（默认 20px）
 */
export function computeFit(
  imgWidth: number,
  imgHeight: number,
  containerW: number,
  containerH: number,
  padding: number = 20,
): FitResult {
  const availW = containerW - padding * 2;
  const availH = containerH - padding * 2;

  if (imgWidth <= 0 || imgHeight <= 0 || availW <= 0 || availH <= 0) {
    return { scale: 1, offsetX: 0, offsetY: 0, displayWidth: imgWidth, displayHeight: imgHeight };
  }

  // 不放大小图
  if (imgWidth <= availW && imgHeight <= availH) {
    return {
      scale: 1,
      offsetX: (containerW - imgWidth) / 2,
      offsetY: (containerH - imgHeight) / 2,
      displayWidth: imgWidth,
      displayHeight: imgHeight,
    };
  }

  const scaleX = availW / imgWidth;
  const scaleY = availH / imgHeight;
  const scale = Math.min(scaleX, scaleY);

  const displayWidth = imgWidth * scale;
  const displayHeight = imgHeight * scale;

  return {
    scale,
    offsetX: (containerW - displayWidth) / 2,
    offsetY: (containerH - displayHeight) / 2,
    displayWidth,
    displayHeight,
  };
}

/**
 * 画布坐标 → 原图坐标
 * 将用户在画布上的操作坐标转换为原图像素坐标（用于提交后端）
 */
export function canvasToImage(
  canvasX: number,
  canvasY: number,
  fit: FitResult,
): { x: number; y: number } {
  return {
    x: (canvasX - fit.offsetX) / fit.scale,
    y: (canvasY - fit.offsetY) / fit.scale,
  };
}

/**
 * 原图坐标 → 画布坐标
 * 将后端返回的原图坐标转换为画布显示坐标
 */
export function imageToCanvas(
  imgX: number,
  imgY: number,
  fit: FitResult,
): { x: number; y: number } {
  return {
    x: imgX * fit.scale + fit.offsetX,
    y: imgY * fit.scale + fit.offsetY,
  };
}

/**
 * 批量坐标转换：画布坐标 bbox → 原图坐标 bbox
 * 用于批量提交标注框的坐标映射
 */
export function canvasBboxToImage(
  bbox: { x: number; y: number; width: number; height: number },
  fit: FitResult,
): { x: number; y: number; width: number; height: number } {
  const tl = canvasToImage(bbox.x, bbox.y, fit);
  const br = canvasToImage(bbox.x + bbox.width, bbox.y + bbox.height, fit);
  return {
    x: Math.max(0, tl.x),
    y: Math.max(0, tl.y),
    width: Math.max(0, br.x - tl.x),
    height: Math.max(0, br.y - tl.y),
  };
}

/**
 * 约束坐标到有效图像范围内
 */
export function clampToImageBounds(
  x: number,
  y: number,
  imgWidth: number,
  imgHeight: number,
): { x: number; y: number } {
  return {
    x: Math.max(0, Math.min(x, imgWidth)),
    y: Math.max(0, Math.min(y, imgHeight)),
  };
}
