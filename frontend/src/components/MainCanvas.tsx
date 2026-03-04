/**
 * 主画布组件（v5）
 *
 * v5 增强：
 *   - 图片自适应渲染：不同尺寸图片等比缩放居中，切图自动重算
 *   - 坐标映射：标注坐标基于缩放后画布，提交时转为原图坐标
 *   - 图层强绑定 displayImageId，切图销毁上一张所有图层
 *   - 手动标注矩形工具 + Ctrl+Z/Shift+Z 撤销重做
 */
import React, { useRef, useCallback, useEffect, useState, useMemo } from 'react';
import { Stage, Layer, Image as KonvaImage, Rect } from 'react-konva';
import { UploadOutlined } from '@ant-design/icons';
import useImage from 'use-image';
import { useAppStore } from '../store/useAppStore';
import { startManualSam, getTaskStatus } from '../api';
import { computeFit, canvasToImage, type FitResult } from '../utils/imageAdapter';

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
    categories, activeCategoryId, mode2CategoryRefs,
    stageScale, stagePosition,
    manualTool,
    setBBox, setIsDrawing, toggleInstanceId,
    setStageScale, setStagePosition, setImageFit,
    addMaskToImage, undo, redo,
  } = useAppStore();

  const displayImageId = viewingImageId || selectedImageId;
  const displayImage = images.find((img) => img.id === displayImageId);
  const originalImage = useLoadImage(displayImage?.url || null);
  const currentMaskUrls = displayImageId ? (maskUrls[displayImageId] || []) : [];

  const showBbox = currentMode === 'mode2' && bbox && bboxImageId === displayImageId;
  const showInstances = currentMode === 'mode3' && instanceMasks.length > 0 && instanceMasksImageId === displayImageId;

  const confirmedBboxes = currentMode === 'mode2'
    ? mode2CategoryRefs
        .filter((ref) => ref.imageId === displayImageId)
        .flatMap((ref) => {
          const cat = categories.find((c) => c.id === ref.categoryId);
          return ref.bboxes.map((b) => ({ bbox: b, color: cat?.color || '#1890ff' }));
        })
    : [];

  const drawStart = useRef<{ x: number; y: number } | null>(null);

  // ===== 容器大小监听 =====
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

  // ===== 图片自适应计算：切图或容器变化时重算 =====
  const imageFit: FitResult = useMemo(() => {
    if (!originalImage) {
      return { scale: 1, offsetX: 0, offsetY: 0, displayWidth: 0, displayHeight: 0 };
    }
    return computeFit(
      originalImage.naturalWidth || originalImage.width,
      originalImage.naturalHeight || originalImage.height,
      containerSize.width,
      containerSize.height,
    );
  }, [originalImage, containerSize.width, containerSize.height]);

  // 同步到 Store（供 Toolbar 显示和坐标转换使用）
  useEffect(() => {
    setImageFit(imageFit.scale, imageFit.offsetX, imageFit.offsetY);
  }, [imageFit.scale, imageFit.offsetX, imageFit.offsetY, setImageFit]);

  // 切图时重置 Stage 缩放和位置
  useEffect(() => {
    setStageScale(1);
    setStagePosition({ x: 0, y: 0 });
  }, [displayImageId, setStageScale, setStagePosition]);

  // Ctrl+Z / Ctrl+Shift+Z 快捷键
  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      if ((e.ctrlKey || e.metaKey) && e.key === 'z') {
        e.preventDefault();
        if (e.shiftKey) redo(); else undo();
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
        x: Math.min(manualDrawStart.x, pos.x), y: Math.min(manualDrawStart.y, pos.y),
        w: Math.abs(pos.x - manualDrawStart.x), h: Math.abs(pos.y - manualDrawStart.y),
      });
      return;
    }
    if (!isDrawing || !canDraw || !drawStart.current) return;
    setBBox({
      x: Math.min(drawStart.current.x, pos.x), y: Math.min(drawStart.current.y, pos.y),
      width: Math.abs(pos.x - drawStart.current.x), height: Math.abs(pos.y - drawStart.current.y),
    });
  }, [isDrawing, canDraw, canManualDraw, manualDrawStart, setBBox]);

  const handleMouseUp = useCallback(async () => {
    // 手动标注矩形：画布坐标 → 原图坐标 → 发送后端
    if (manualDrawStart && canManualDraw && manualRect && manualRect.w > 10 && manualRect.h > 10 && displayImage) {
      setManualDrawStart(null);
      // 转为原图坐标
      const tl = canvasToImage(manualRect.x, manualRect.y, imageFit);
      const br = canvasToImage(manualRect.x + manualRect.w, manualRect.y + manualRect.h, imageFit);
      const bboxCoords: [number, number, number, number] = [tl.x, tl.y, br.x, br.y];
      // 传入当前选中类别的颜色，使手动标注 Mask 与自动标注同类别颜色一致
      const activeCat = activeCategoryId ? categories.find(c => c.id === activeCategoryId) : null;
      try {
        const result = await startManualSam({
          image_id: displayImage.id, image_path: displayImage.path, bbox: bboxCoords,
          category_color: activeCat?.color || null,
          category_name: activeCat?.name || null,
        });
        const pollManual = setInterval(async () => {
          const res = await getTaskStatus(result.task_id);
          if (res.status === 'success' && res.mask_url) {
            clearInterval(pollManual); addMaskToImage(displayImage.id, res.mask_url); setManualRect(null);
          } else if (res.status === 'failed') {
            clearInterval(pollManual); setManualRect(null);
          }
        }, 500);
      } catch { setManualRect(null); }
      return;
    }
    setManualDrawStart(null);
    if (!isDrawing) return;
    setIsDrawing(false);
    drawStart.current = null;
  }, [isDrawing, canManualDraw, manualDrawStart, manualRect, displayImage, imageFit, setIsDrawing, addMaskToImage]);

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
      useAppStore.getState().setSelectedInstanceIds(
        s.selectedInstanceIds.length === 1 && s.selectedInstanceIds[0] === instId ? [] : [instId]
      );
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

        {/* Layer0: 原图（自适应缩放居中） */}
        <Layer>
          {originalImage && (
            <KonvaImage
              image={originalImage}
              x={imageFit.offsetX}
              y={imageFit.offsetY}
              width={imageFit.displayWidth}
              height={imageFit.displayHeight}
            />
          )}
        </Layer>

        {/* Layer2: Mask 结果层（同步缩放偏移） */}
        <Layer opacity={maskOpacity} x={imageFit.offsetX} y={imageFit.offsetY}
               scaleX={imageFit.scale} scaleY={imageFit.scale}>
          {currentMaskUrls.map((url, idx) => (
            <MaskImage key={`${displayImageId}-mask-${idx}`} url={url} />
          ))}
        </Layer>

        {/* Layer3: 模式3 实例层 */}
        {showInstances && (
          <Layer opacity={0.6} x={imageFit.offsetX} y={imageFit.offsetY}
                 scaleX={imageFit.scale} scaleY={imageFit.scale}>
            {instanceMasks.map((inst) => (
              <InstanceMaskImage key={`inst-${displayImageId}-${inst.id}`} url={inst.mask_url}
                isSelected={selectedInstanceIds.includes(inst.id)}
                onClick={(e: any) => handleInstanceClick(inst.id, e)} />
            ))}
          </Layer>
        )}

        {/* Layer1: 交互层（框选矩形、确认框选） */}
        {currentMode === 'mode2' && (
          <Layer>
            {confirmedBboxes.map((item, idx) => (
              <Rect key={`cb-${idx}`}
                x={item.bbox.x * imageFit.scale + imageFit.offsetX}
                y={item.bbox.y * imageFit.scale + imageFit.offsetY}
                width={item.bbox.width * imageFit.scale}
                height={item.bbox.height * imageFit.scale}
                stroke={item.color} strokeWidth={2 / stageScale}
                dash={[4 / stageScale, 2 / stageScale]} fill={`${item.color}18`} />
            ))}
            {showBbox && bbox!.width > 0 && bbox!.height > 0 && (
              <Rect x={bbox!.x} y={bbox!.y} width={bbox!.width} height={bbox!.height}
                stroke="#1890ff" strokeWidth={2 / stageScale}
                dash={[6 / stageScale, 3 / stageScale]} fill="rgba(24,144,255,0.1)" />
            )}
          </Layer>
        )}

        {/* 手动标注矩形预览 */}
        {manualRect && manualRect.w > 0 && manualRect.h > 0 && (
          <Layer>
            <Rect x={manualRect.x} y={manualRect.y} width={manualRect.w} height={manualRect.h}
              stroke="#ffa500" strokeWidth={2 / stageScale}
              dash={[4 / stageScale, 2 / stageScale]} fill="rgba(255,165,0,0.1)" />
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

      {/* 图片信息 + 缩放比例 */}
      <div style={{ position: 'absolute', bottom: 8, right: 8, background: 'rgba(0,0,0,0.5)', color: '#aaa', padding: '2px 8px', borderRadius: 4, fontSize: 11, pointerEvents: 'none' }}>
        {originalImage && `${originalImage.naturalWidth || originalImage.width}×${originalImage.naturalHeight || originalImage.height} | `}
        适配 {Math.round(imageFit.scale * 100)}% | 视图 {Math.round(stageScale * 100)}%
      </div>

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
