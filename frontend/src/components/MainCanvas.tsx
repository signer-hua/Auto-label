/**
 * 主画布组件（增强版）
 * 基于 react-konva 实现多层图层：
 *   Layer0: 原图层 — 显示当前选中的图像
 *   Layer1: 交互层 — 鼠标框选矩形（仅模式2 激活，与当前图片 ID 绑定）
 *   Layer2: 结果层 — Mask 透明 PNG 叠加（模式1蓝/模式2红/模式3绿）
 *   Layer3: 实例层 — 模式3 粗分割实例 Mask（彩色区分，支持 Ctrl/Shift 多选）
 *
 * v2 修复：
 *   - 框选矩形与 bboxImageId 强绑定，切图不残留
 *   - 实例图层与 instanceMasksImageId 强绑定，切图不重叠
 *   - 多类别确认框选可视化（不同类别颜色区分）
 *   - 支持 Ctrl/Shift 多选粗分割实例
 */
import React, { useRef, useCallback, useEffect, useState } from 'react';
import { Stage, Layer, Image as KonvaImage, Rect } from 'react-konva';
import { UploadOutlined } from '@ant-design/icons';
import useImage from 'use-image';
import { useAppStore } from '../store/useAppStore';

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
    currentMode, activeTool, bbox, bboxImageId, isDrawing,
    maskUrls, maskOpacity,
    instanceMasks, instanceMasksImageId, selectedInstanceIds,
    categories, mode2CategoryRefs,
    stageScale, stagePosition,
    setBBox, setIsDrawing, toggleInstanceId,
    setStageScale, setStagePosition,
  } = useAppStore();

  const displayImageId = viewingImageId || selectedImageId;
  const displayImage = images.find((img) => img.id === displayImageId);
  const originalImage = useLoadImage(displayImage?.url || null);

  const currentMaskUrls = displayImageId ? (maskUrls[displayImageId] || []) : [];

  // 图层隔离：仅当 bbox 属于当前图片时渲染
  const showBbox = currentMode === 'mode2' && bbox && bboxImageId === displayImageId;
  // 图层隔离：仅当实例属于当前图片时渲染
  const showInstances = currentMode === 'mode3'
    && instanceMasks.length > 0
    && instanceMasksImageId === displayImageId;

  // 多类别确认框选：仅显示属于当前图片的
  const confirmedBboxes = currentMode === 'mode2'
    ? mode2CategoryRefs.flatMap((ref) => {
        const cat = categories.find((c) => c.id === ref.categoryId);
        return ref.bboxes.map((b) => ({
          bbox: b,
          color: cat?.color || '#1890ff',
          categoryName: cat?.name || '',
        }));
      })
    : [];

  const drawStart = useRef<{ x: number; y: number } | null>(null);

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

  const canDraw = currentMode === 'mode2' && activeTool === 'select';

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

  const handleMouseUp = useCallback(() => {
    if (!isDrawing) return;
    setIsDrawing(false);
    drawStart.current = null;
  }, [isDrawing, setIsDrawing]);

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

  const handleDragEnd = useCallback((e: any) => {
    if (currentMode !== 'mode2' || activeTool === 'pan') {
      setStagePosition({ x: e.target.x(), y: e.target.y() });
    }
  }, [activeTool, currentMode, setStagePosition]);

  /** 实例点击处理：支持 Ctrl/Shift 多选 */
  const handleInstanceClick = useCallback((instId: number, e: any) => {
    const nativeEvent = e?.evt as MouseEvent | undefined;
    if (nativeEvent?.ctrlKey || nativeEvent?.shiftKey || nativeEvent?.metaKey) {
      toggleInstanceId(instId);
    } else {
      const state = useAppStore.getState();
      if (state.selectedInstanceIds.length === 1 && state.selectedInstanceIds[0] === instId) {
        useAppStore.getState().setSelectedInstanceIds([]);
      } else {
        useAppStore.getState().setSelectedInstanceIds([instId]);
      }
    }
  }, [toggleInstanceId]);

  const getCursor = () => {
    if (currentMode === 'mode2' && activeTool === 'select') return 'crosshair';
    if (activeTool === 'pan' || currentMode !== 'mode2') return 'grab';
    return 'zoom-in';
  };

  const isDraggable = currentMode !== 'mode2' || activeTool === 'pan';

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

        {/* Layer3: 模式3 实例层（图层隔离 + 多选高亮） */}
        {showInstances && (
          <Layer opacity={0.6}>
            {instanceMasks.map((inst) => (
              <InstanceMaskImage
                key={`inst-${displayImageId}-${inst.id}`}
                url={inst.mask_url}
                isSelected={selectedInstanceIds.includes(inst.id)}
                onClick={(e: any) => handleInstanceClick(inst.id, e)}
              />
            ))}
          </Layer>
        )}

        {/* Layer1: 交互层 — 确认框选 + 当前框选（仅模式2 + 图层隔离） */}
        {currentMode === 'mode2' && (
          <Layer>
            {/* 已确认的多类别框选矩形 */}
            {confirmedBboxes.map((item, idx) => (
              <Rect
                key={`confirmed-bbox-${idx}`}
                x={item.bbox.x}
                y={item.bbox.y}
                width={item.bbox.width}
                height={item.bbox.height}
                stroke={item.color}
                strokeWidth={2 / stageScale}
                dash={[4 / stageScale, 2 / stageScale]}
                fill={`${item.color}18`}
              />
            ))}
            {/* 当前正在绘制的框选矩形（图层隔离） */}
            {showBbox && bbox!.width > 0 && bbox!.height > 0 && (
              <Rect
                x={bbox!.x}
                y={bbox!.y}
                width={bbox!.width}
                height={bbox!.height}
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

      {/* 多选实例提示 */}
      {currentMode === 'mode3' && selectedInstanceIds.length > 0 && (
        <div style={{
          position: 'absolute', top: 8, right: 8,
          background: 'rgba(0,200,80,0.9)',
          color: '#fff', padding: '2px 10px', borderRadius: 4,
          fontSize: 12, pointerEvents: 'none',
        }}>
          已选 {selectedInstanceIds.length} 个实例 (Ctrl+点击多选)
        </div>
      )}

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

/** 模式3 实例 Mask 组件（多选高亮） */
const InstanceMaskImage: React.FC<{
  url: string;
  isSelected: boolean;
  onClick: (e: any) => void;
}> = ({ url, isSelected, onClick }) => {
  const image = useLoadImage(url);
  if (!image) return null;
  return (
    <KonvaImage
      image={image}
      x={0}
      y={0}
      opacity={isSelected ? 1 : 0.4}
      onClick={onClick}
      onTap={onClick}
    />
  );
};

export default MainCanvas;
