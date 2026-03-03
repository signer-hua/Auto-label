/**
 * 主画布组件（v3 增强版）
 *
 * v3 新增：
 *   - 手动标注矩形工具（rect_manual）：画矩形 → 触发 SAM3 生成 Mask
 *   - Ctrl+Z 撤销 / Ctrl+Shift+Z 重做
 *   - 手动删除 Mask（双击 Mask 弹出删除）
 *   - 图层强绑定 currentImageId，切图自动销毁所有非当前图层
 */
import React, { useRef, useCallback, useEffect, useState } from 'react';
import { Stage, Layer, Image as KonvaImage, Rect } from 'react-konva';
import { UploadOutlined } from '@ant-design/icons';
import useImage from 'use-image';
import { useAppStore } from '../store/useAppStore';
import { startManualSam, getTaskStatus } from '../api';

function useLoadImage(url: string | null) {
  const [image] = useImage(url || '', 'anonymous');
  return image;
}

const MainCanvas: React.FC = () => {
  const stageRef = useRef<any>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const [containerSize, setContainerSize] = useState({ width: 800, height: 600 });
  const [manualDrawStart, setManualDrawStart] = useState<{ x: number; y: number } | null>(null);
  const [manualRect, setManualRect] = useState<{ x: number; y: number; w: number; h: number } | null>(null);

  const {
    images, selectedImageId, viewingImageId,
    currentMode, activeTool, bbox, bboxImageId, isDrawing,
    maskUrls, maskOpacity,
    instanceMasks, instanceMasksImageId, selectedInstanceIds,
    categories, mode2CategoryRefs,
    stageScale, stagePosition,
    manualTool,
    setBBox, setIsDrawing, toggleInstanceId,
    setStageScale, setStagePosition,
    addMaskToImage, undo, redo,
  } = useAppStore();

  const displayImageId = viewingImageId || selectedImageId;
  const displayImage = images.find((img) => img.id === displayImageId);
  const originalImage = useLoadImage(displayImage?.url || null);
  const currentMaskUrls = displayImageId ? (maskUrls[displayImageId] || []) : [];

  const showBbox = currentMode === 'mode2' && bbox && bboxImageId === displayImageId;
  const showInstances = currentMode === 'mode3' && instanceMasks.length > 0 && instanceMasksImageId === displayImageId;

  // 仅显示属于当前图片的已确认框选（修复跨图残留 Bug）
  const confirmedBboxes = currentMode === 'mode2'
    ? mode2CategoryRefs
        .filter((ref) => ref.imageId === displayImageId)
        .flatMap((ref) => {
          const cat = categories.find((c) => c.id === ref.categoryId);
          return ref.bboxes.map((b) => ({ bbox: b, color: cat?.color || '#1890ff' }));
        })
    : [];

  const drawStart = useRef<{ x: number; y: number } | null>(null);

  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;
    const observer = new ResizeObserver((entries) => {
      for (const entry of entries)
        setContainerSize({ width: entry.contentRect.width, height: entry.contentRect.height });
    });
    observer.observe(container);
    return () => observer.disconnect();
  }, []);

  // Ctrl+Z / Ctrl+Shift+Z 快捷键
  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      if ((e.ctrlKey || e.metaKey) && e.key === 'z') {
        e.preventDefault();
        if (e.shiftKey) { redo(); } else { undo(); }
      }
    };
    window.addEventListener('keydown', handler);
    return () => window.removeEventListener('keydown', handler);
  }, [undo, redo]);

  const canDraw = currentMode === 'mode2' && activeTool === 'select';
  const canManualDraw = manualTool === 'rect_manual';

  const handleMouseDown = useCallback((e: any) => {
    const stage = stageRef.current;
    if (!stage) return;
    const pos = stage.getRelativePointerPosition();
    if (!pos) return;

    if (canManualDraw) {
      setManualDrawStart({ x: pos.x, y: pos.y });
      setManualRect({ x: pos.x, y: pos.y, w: 0, h: 0 });
      return;
    }
    if (!canDraw) return;
    drawStart.current = { x: pos.x, y: pos.y };
    setBBox({ x: pos.x, y: pos.y, width: 0, height: 0 });
    setIsDrawing(true);
  }, [canDraw, canManualDraw, setBBox, setIsDrawing]);

  const handleMouseMove = useCallback(() => {
    const stage = stageRef.current;
    if (!stage) return;
    const pos = stage.getRelativePointerPosition();
    if (!pos) return;

    if (manualDrawStart && canManualDraw) {
      setManualRect({
        x: Math.min(manualDrawStart.x, pos.x),
        y: Math.min(manualDrawStart.y, pos.y),
        w: Math.abs(pos.x - manualDrawStart.x),
        h: Math.abs(pos.y - manualDrawStart.y),
      });
      return;
    }
    if (!isDrawing || !canDraw || !drawStart.current) return;
    const x = Math.min(drawStart.current.x, pos.x);
    const y = Math.min(drawStart.current.y, pos.y);
    setBBox({ x, y, width: Math.abs(pos.x - drawStart.current.x), height: Math.abs(pos.y - drawStart.current.y) });
  }, [isDrawing, canDraw, canManualDraw, manualDrawStart, setBBox]);

  const handleMouseUp = useCallback(async () => {
    if (manualDrawStart && canManualDraw && manualRect && manualRect.w > 10 && manualRect.h > 10 && displayImage) {
      setManualDrawStart(null);
      const bboxCoords: [number, number, number, number] = [
        manualRect.x, manualRect.y, manualRect.x + manualRect.w, manualRect.y + manualRect.h,
      ];
      try {
        const result = await startManualSam({
          image_id: displayImage.id, image_path: displayImage.path, bbox: bboxCoords,
        });
        const pollManual = setInterval(async () => {
          const res = await getTaskStatus(result.task_id);
          if (res.status === 'success' && res.mask_url) {
            clearInterval(pollManual);
            addMaskToImage(displayImage.id, res.mask_url);
            setManualRect(null);
          } else if (res.status === 'failed') {
            clearInterval(pollManual);
            setManualRect(null);
          }
        }, 500);
      } catch { setManualRect(null); }
      return;
    }
    setManualDrawStart(null);
    if (!isDrawing) return;
    setIsDrawing(false);
    drawStart.current = null;
  }, [isDrawing, canManualDraw, manualDrawStart, manualRect, displayImage, setIsDrawing, addMaskToImage]);

  const handleWheel = useCallback((e: any) => {
    e.evt.preventDefault();
    const stage = stageRef.current;
    if (!stage) return;
    const pointer = stage.getPointerPosition();
    if (!pointer) return;
    const scaleBy = 1.08;
    const newScale = e.evt.deltaY < 0 ? stageScale * scaleBy : stageScale / scaleBy;
    const clamped = Math.max(0.1, Math.min(5, newScale));
    const mp = { x: (pointer.x - stagePosition.x) / stageScale, y: (pointer.y - stagePosition.y) / stageScale };
    setStageScale(clamped);
    setStagePosition({ x: pointer.x - mp.x * clamped, y: pointer.y - mp.y * clamped });
  }, [stageScale, stagePosition, setStageScale, setStagePosition]);

  const handleDragEnd = useCallback((e: any) => {
    if (currentMode !== 'mode2' || activeTool === 'pan') {
      setStagePosition({ x: e.target.x(), y: e.target.y() });
    }
  }, [activeTool, currentMode, setStagePosition]);

  const handleInstanceClick = useCallback((instId: number, e: any) => {
    const ne = e?.evt as MouseEvent | undefined;
    if (ne?.ctrlKey || ne?.shiftKey || ne?.metaKey) {
      toggleInstanceId(instId);
    } else {
      const s = useAppStore.getState();
      if (s.selectedInstanceIds.length === 1 && s.selectedInstanceIds[0] === instId) {
        useAppStore.getState().setSelectedInstanceIds([]);
      } else {
        useAppStore.getState().setSelectedInstanceIds([instId]);
      }
    }
  }, [toggleInstanceId]);

  const getCursor = () => {
    if (canManualDraw) return 'crosshair';
    if (currentMode === 'mode2' && activeTool === 'select') return 'crosshair';
    if (activeTool === 'pan' || currentMode !== 'mode2') return 'grab';
    return 'zoom-in';
  };

  const isDraggable = !canManualDraw && (currentMode !== 'mode2' || activeTool === 'pan');

  const modeColors: Record<string, string> = { mode1: 'rgba(0,120,255,0.8)', mode2: 'rgba(255,77,79,0.8)', mode3: 'rgba(0,200,80,0.8)' };
  const modeLabels: Record<string, string> = { mode1: '模式1：文本标注', mode2: '模式2：框选标注', mode3: '模式3：实例标注' };

  return (
    <div ref={containerRef} style={{ flex: 1, background: '#2a2a2a', overflow: 'hidden', position: 'relative', cursor: getCursor() }}>
      <Stage ref={stageRef} width={containerSize.width} height={containerSize.height}
        scaleX={stageScale} scaleY={stageScale} x={stagePosition.x} y={stagePosition.y}
        draggable={isDraggable}
        onMouseDown={handleMouseDown} onMouseMove={handleMouseMove} onMouseUp={handleMouseUp}
        onWheel={handleWheel} onDragEnd={handleDragEnd}>

        <Layer>{originalImage && <KonvaImage image={originalImage} x={0} y={0} />}</Layer>

        <Layer opacity={maskOpacity}>
          {currentMaskUrls.map((url, idx) => (<MaskImage key={`${displayImageId}-mask-${idx}`} url={url} />))}
        </Layer>

        {showInstances && (
          <Layer opacity={0.6}>
            {instanceMasks.map((inst) => (
              <InstanceMaskImage key={`inst-${displayImageId}-${inst.id}`} url={inst.mask_url}
                isSelected={selectedInstanceIds.includes(inst.id)}
                onClick={(e: any) => handleInstanceClick(inst.id, e)} />
            ))}
          </Layer>
        )}

        {currentMode === 'mode2' && (
          <Layer>
            {confirmedBboxes.map((item, idx) => (
              <Rect key={`cb-${idx}`} x={item.bbox.x} y={item.bbox.y} width={item.bbox.width} height={item.bbox.height}
                stroke={item.color} strokeWidth={2 / stageScale} dash={[4 / stageScale, 2 / stageScale]} fill={`${item.color}18`} />
            ))}
            {showBbox && bbox!.width > 0 && bbox!.height > 0 && (
              <Rect x={bbox!.x} y={bbox!.y} width={bbox!.width} height={bbox!.height}
                stroke="#1890ff" strokeWidth={2 / stageScale} dash={[6 / stageScale, 3 / stageScale]} fill="rgba(24,144,255,0.1)" />
            )}
          </Layer>
        )}

        {/* 手动标注矩形预览 */}
        {manualRect && manualRect.w > 0 && manualRect.h > 0 && (
          <Layer>
            <Rect x={manualRect.x} y={manualRect.y} width={manualRect.w} height={manualRect.h}
              stroke="#ffa500" strokeWidth={2 / stageScale} dash={[4 / stageScale, 2 / stageScale]} fill="rgba(255,165,0,0.1)" />
          </Layer>
        )}
      </Stage>

      {!displayImage && (
        <div style={{ position: 'absolute', top: '50%', left: '50%', transform: 'translate(-50%, -50%)', color: '#666', fontSize: 16, textAlign: 'center', pointerEvents: 'none' }}>
          <UploadOutlined style={{ fontSize: 48, color: '#444', display: 'block', marginBottom: 12 }} />
          <p>请先上传图片</p>
        </div>
      )}

      <div style={{ position: 'absolute', top: 8, left: 8, background: modeColors[currentMode], color: '#fff', padding: '2px 10px', borderRadius: 4, fontSize: 12, pointerEvents: 'none' }}>
        {modeLabels[currentMode]}
        {manualTool && <span style={{ marginLeft: 6 }}>| 手动标注</span>}
      </div>

      {currentMode === 'mode3' && selectedInstanceIds.length > 0 && (
        <div style={{ position: 'absolute', top: 8, right: 8, background: 'rgba(0,200,80,0.9)', color: '#fff', padding: '2px 10px', borderRadius: 4, fontSize: 12, pointerEvents: 'none' }}>
          已选 {selectedInstanceIds.length} 个实例
        </div>
      )}

      <div style={{ position: 'absolute', bottom: 8, right: 8, background: 'rgba(0,0,0,0.5)', color: '#aaa', padding: '2px 8px', borderRadius: 4, fontSize: 11, pointerEvents: 'none' }}>
        {Math.round(stageScale * 100)}%
      </div>

      {/* 撤销/重做提示 */}
      <div style={{ position: 'absolute', bottom: 8, left: 8, color: '#555', fontSize: 10, pointerEvents: 'none' }}>
        Ctrl+Z 撤销 | Ctrl+Shift+Z 重做
      </div>
    </div>
  );
};

const MaskImage: React.FC<{ url: string }> = ({ url }) => {
  const image = useLoadImage(url);
  return image ? <KonvaImage image={image} x={0} y={0} /> : null;
};

const InstanceMaskImage: React.FC<{ url: string; isSelected: boolean; onClick: (e: any) => void }> = ({ url, isSelected, onClick }) => {
  const image = useLoadImage(url);
  return image ? <KonvaImage image={image} x={0} y={0} opacity={isSelected ? 1 : 0.4} onClick={onClick} onTap={onClick} /> : null;
};

export default MainCanvas;
