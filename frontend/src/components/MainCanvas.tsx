/**
 * 主画布组件
 * 基于 react-konva 实现多层图层：
 *   Layer0: 原图层 — 显示当前选中的图像
 *   Layer1: 交互层 — 鼠标框选矩形（仅模式2 激活）
 *   Layer2: 结果层 — Mask 透明 PNG 叠加（模式1蓝/模式2红/模式3绿）
 *   Layer3: 实例层 — 模式3 粗分割实例 Mask（彩色区分，可点击选中）
 *
 * 支持缩放/平移，Mask 图层与原图同步变换。
 * Mask 透明度由 Zustand maskOpacity 控制。
 */
import React, { useRef, useCallback, useEffect, useState } from 'react';
import { Stage, Layer, Image as KonvaImage, Rect } from 'react-konva';
import { UploadOutlined } from '@ant-design/icons';
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
    maskUrls, maskOpacity,
    instanceMasks, selectedInstanceId, setSelectedInstanceId,
    stageScale, stagePosition,
    setBBox, setIsDrawing,
    setStageScale, setStagePosition,
  } = useAppStore();

  // 当前显示的图片
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

  /** 鼠标按下 */
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

  /** 鼠标移动 */
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

  /** 鼠标松开 */
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
    if (currentMode !== 'mode2' || activeTool === 'pan') {
      setStagePosition({ x: e.target.x(), y: e.target.y() });
    }
  }, [activeTool, currentMode, setStagePosition]);

  // 光标样式
  const getCursor = () => {
    if (currentMode === 'mode2' && activeTool === 'select') return 'crosshair';
    if (activeTool === 'pan' || currentMode !== 'mode2') return 'grab';
    return 'zoom-in';
  };

  const isDraggable = currentMode !== 'mode2' || activeTool === 'pan';

  // 模式颜色标签
  const modeColors: Record<string, string> = {
    mode1: 'rgba(0,120,255,0.8)',
    mode2: 'rgba(255,77,79,0.8)',
    mode3: 'rgba(0,200,80,0.8)',
  };
  const modeLabels: Record<string, string> = {
    mode1: '模式1：文本标注',
    mode2: '模式2：框选标注',
    mode3: '模式3：实例标注',
  };

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
          {originalImage && <KonvaImage image={originalImage} x={0} y={0} />}
        </Layer>

        {/* Layer2: Mask 结果层 */}
        <Layer opacity={maskOpacity}>
          {currentMaskUrls.map((url, idx) => (
            <MaskImage key={`${displayImageId}-mask-${idx}`} url={url} />
          ))}
        </Layer>

        {/* Layer3: 模式3 实例层（粗分割实例，彩色区分） */}
        {currentMode === 'mode3' && instanceMasks.length > 0 && (
          <Layer opacity={0.6}>
            {instanceMasks.map((inst) => (
              <InstanceMaskImage
                key={`inst-${inst.id}`}
                url={inst.mask_url}
                isSelected={inst.id === selectedInstanceId}
                onClick={() => setSelectedInstanceId(inst.id)}
              />
            ))}
          </Layer>
        )}

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
          top: '50%', left: '50%',
          transform: 'translate(-50%, -50%)',
          color: '#666', fontSize: 16, textAlign: 'center',
          pointerEvents: 'none',
        }}>
          <UploadOutlined style={{ fontSize: 48, color: '#444', display: 'block', marginBottom: 12 }} />
          <p>请先上传图片</p>
          <p style={{ fontSize: 12, color: '#444' }}>
            点击左侧「上传图片」或拖拽图片到工具栏
          </p>
        </div>
      )}

      {/* 模式提示角标 */}
      <div style={{
        position: 'absolute', top: 8, left: 8,
        background: modeColors[currentMode],
        color: '#fff', padding: '2px 10px', borderRadius: 4,
        fontSize: 12, pointerEvents: 'none',
      }}>
        {modeLabels[currentMode]}
      </div>

      {/* 缩放比例显示 */}
      <div style={{
        position: 'absolute', bottom: 8, right: 8,
        background: 'rgba(0,0,0,0.5)',
        color: '#aaa', padding: '2px 8px', borderRadius: 4,
        fontSize: 11, pointerEvents: 'none',
      }}>
        {Math.round(stageScale * 100)}%
      </div>
    </div>
  );
};

/** Mask 图层组件 */
const MaskImage: React.FC<{ url: string }> = ({ url }) => {
  const image = useLoadImage(url);
  if (!image) return null;
  return <KonvaImage image={image} x={0} y={0} />;
};

/** 模式3 实例 Mask 组件（可点击选中，选中时高亮边框） */
const InstanceMaskImage: React.FC<{
  url: string;
  isSelected: boolean;
  onClick: () => void;
}> = ({ url, isSelected, onClick }) => {
  const image = useLoadImage(url);
  if (!image) return null;
  return (
    <KonvaImage
      image={image}
      x={0}
      y={0}
      opacity={isSelected ? 1 : 0.5}
      onClick={onClick}
      onTap={onClick}
    />
  );
};

export default MainCanvas;
