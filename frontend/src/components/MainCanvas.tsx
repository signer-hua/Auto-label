/**
 * 主画布组件
 * 基于 react-konva 实现三层图层：
 *   Layer0: 原图层 — 显示当前选中的图像
 *   Layer1: 交互层 — 鼠标框选矩形（仅模式2 激活）
 *   Layer2: 结果层 — Mask 透明 PNG 叠加（模式1 蓝色 / 模式2 红色）
 *
 * 支持缩放/平移，Mask 图层与原图同步变换。
 */
import React, { useRef, useCallback, useEffect, useState } from 'react';
import { Stage, Layer, Image as KonvaImage, Rect } from 'react-konva';
import useImage from 'use-image';
import { useAppStore } from '../store/useAppStore';

/** 加载图片的 Hook 封装 */
function useLoadImage(url: string | null) {
  const [image] = useImage(url || '', 'anonymous');
  return image;
}

const MainCanvas: React.FC = () => {
  const stageRef = useRef<any>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const [containerSize, setContainerSize] = useState({ width: 800, height: 600 });

  const {
    images, selectedImageId, viewingImageId,
    currentMode, activeTool, bbox, isDrawing,
    maskUrls,
    stageScale, stagePosition,
    setBBox, setIsDrawing,
    setStageScale, setStagePosition,
  } = useAppStore();

  // 当前显示的图片（查看结果时用 viewingImageId，否则用 selectedImageId）
  const displayImageId = viewingImageId || selectedImageId;
  const displayImage = images.find((img) => img.id === displayImageId);
  const originalImage = useLoadImage(displayImage?.url || null);

  // 当前图片的 Mask URL 列表
  const currentMaskUrls = displayImageId ? (maskUrls[displayImageId] || []) : [];

  // 框选起始点
  const drawStart = useRef<{ x: number; y: number } | null>(null);

  /** 监听容器大小变化 */
  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;

    const observer = new ResizeObserver((entries) => {
      for (const entry of entries) {
        setContainerSize({
          width: entry.contentRect.width,
          height: entry.contentRect.height,
        });
      }
    });
    observer.observe(container);
    return () => observer.disconnect();
  }, []);

  // 框选仅在模式2 + select 工具下激活
  const canDraw = currentMode === 'mode2' && activeTool === 'select';

  /** 鼠标按下：开始框选 */
  const handleMouseDown = useCallback((e: any) => {
    if (!canDraw) return;

    const stage = stageRef.current;
    if (!stage) return;

    const pos = stage.getRelativePointerPosition();
    if (!pos) return;

    drawStart.current = { x: pos.x, y: pos.y };
    setBBox({ x: pos.x, y: pos.y, width: 0, height: 0 });
    setIsDrawing(true);
  }, [canDraw, setBBox, setIsDrawing]);

  /** 鼠标移动：更新框选矩形 */
  const handleMouseMove = useCallback(() => {
    if (!isDrawing || !canDraw || !drawStart.current) return;

    const stage = stageRef.current;
    if (!stage) return;

    const pos = stage.getRelativePointerPosition();
    if (!pos) return;

    const x = Math.min(drawStart.current.x, pos.x);
    const y = Math.min(drawStart.current.y, pos.y);
    const width = Math.abs(pos.x - drawStart.current.x);
    const height = Math.abs(pos.y - drawStart.current.y);

    setBBox({ x, y, width, height });
  }, [isDrawing, canDraw, setBBox]);

  /** 鼠标松开：完成框选 */
  const handleMouseUp = useCallback(() => {
    if (!isDrawing) return;
    setIsDrawing(false);
    drawStart.current = null;
  }, [isDrawing, setIsDrawing]);

  /** 滚轮缩放 */
  const handleWheel = useCallback((e: any) => {
    e.evt.preventDefault();
    const stage = stageRef.current;
    if (!stage) return;

    const oldScale = stageScale;
    const pointer = stage.getPointerPosition();
    if (!pointer) return;

    const scaleBy = 1.08;
    const newScale = e.evt.deltaY < 0 ? oldScale * scaleBy : oldScale / scaleBy;
    const clampedScale = Math.max(0.1, Math.min(5, newScale));

    const mousePointTo = {
      x: (pointer.x - stagePosition.x) / oldScale,
      y: (pointer.y - stagePosition.y) / oldScale,
    };

    setStageScale(clampedScale);
    setStagePosition({
      x: pointer.x - mousePointTo.x * clampedScale,
      y: pointer.y - mousePointTo.y * clampedScale,
    });
  }, [stageScale, stagePosition, setStageScale, setStagePosition]);

  /** 拖拽平移 */
  const handleDragEnd = useCallback((e: any) => {
    if (activeTool === 'pan' || currentMode === 'mode1') {
      setStagePosition({ x: e.target.x(), y: e.target.y() });
    }
  }, [activeTool, currentMode, setStagePosition]);

  // 光标样式
  const getCursor = () => {
    if (currentMode === 'mode1') return 'default';
    if (activeTool === 'select') return 'crosshair';
    if (activeTool === 'pan') return 'grab';
    return 'zoom-in';
  };

  // 画布是否可拖拽（模式1 始终可拖拽，模式2 仅 pan 工具）
  const isDraggable = currentMode === 'mode1' || activeTool === 'pan';

  return (
    <div
      ref={containerRef}
      style={{
        flex: 1,
        background: '#2a2a2a',
        overflow: 'hidden',
        position: 'relative',
        cursor: getCursor(),
      }}
    >
      <Stage
        ref={stageRef}
        width={containerSize.width}
        height={containerSize.height}
        scaleX={stageScale}
        scaleY={stageScale}
        x={stagePosition.x}
        y={stagePosition.y}
        draggable={isDraggable}
        onMouseDown={handleMouseDown}
        onMouseMove={handleMouseMove}
        onMouseUp={handleMouseUp}
        onWheel={handleWheel}
        onDragEnd={handleDragEnd}
      >
        {/* Layer0: 原图层 */}
        <Layer>
          {originalImage && (
            <KonvaImage image={originalImage} x={0} y={0} />
          )}
        </Layer>

        {/* Layer2: Mask 结果层 */}
        <Layer opacity={0.7}>
          {currentMaskUrls.map((url, idx) => (
            <MaskImage key={`${displayImageId}-mask-${idx}`} url={url} />
          ))}
        </Layer>

        {/* Layer1: 交互层 — 框选矩形（仅模式2） */}
        {currentMode === 'mode2' && (
          <Layer>
            {bbox && bbox.width > 0 && bbox.height > 0 && (
              <Rect
                x={bbox.x}
                y={bbox.y}
                width={bbox.width}
                height={bbox.height}
                stroke="#1890ff"
                strokeWidth={2 / stageScale}
                dash={[6 / stageScale, 3 / stageScale]}
                fill="rgba(24, 144, 255, 0.1)"
              />
            )}
          </Layer>
        )}
      </Stage>

      {/* 空状态提示 */}
      {!displayImage && (
        <div style={{
          position: 'absolute',
          top: '50%',
          left: '50%',
          transform: 'translate(-50%, -50%)',
          color: '#666',
          fontSize: 16,
          textAlign: 'center',
          pointerEvents: 'none',
        }}>
          <p>请先上传图片</p>
          <p style={{ fontSize: 12, color: '#444' }}>
            点击左侧「上传图片」按钮
          </p>
        </div>
      )}

      {/* 模式提示角标 */}
      <div style={{
        position: 'absolute',
        top: 8,
        left: 8,
        background: currentMode === 'mode1' ? 'rgba(0,120,255,0.8)' : 'rgba(255,77,79,0.8)',
        color: '#fff',
        padding: '2px 10px',
        borderRadius: 4,
        fontSize: 12,
        pointerEvents: 'none',
      }}>
        {currentMode === 'mode1' ? '模式1：文本标注' : '模式2：框选标注'}
      </div>
    </div>
  );
};

/** Mask 图层组件：加载透明 PNG 并叠加 */
const MaskImage: React.FC<{ url: string }> = ({ url }) => {
  const image = useLoadImage(url);
  if (!image) return null;
  return <KonvaImage image={image} x={0} y={0} />;
};

export default MainCanvas;
