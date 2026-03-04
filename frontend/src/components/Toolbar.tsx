/**
 * 左侧工具栏（v4）
 *
 * v4 修复与增强：
 *   - 模式1 新增类别下拉框 + 多粒度提示词示例
 *   - 模式2 标注提交后自动清空 bbox（修复残留 Bug）
 *   - 全局 CategoryPanel 集成到底部
 *   - 手动标注工具集
 */
import React, { useCallback, useRef } from 'react';
import { Button, Divider, Tooltip, Input, message, Radio, Space, Slider, Select } from 'antd';
import {
  SelectOutlined, DragOutlined, ZoomInOutlined, PlayCircleOutlined,
  ClearOutlined, UploadOutlined, FontSizeOutlined, AimOutlined,
  ThunderboltOutlined, AppstoreOutlined, SearchOutlined,
  CheckOutlined, BorderOutlined, UndoOutlined, RedoOutlined, ScissorOutlined,
  ZoomInOutlined as ZoomInBtn, ZoomOutOutlined, FullscreenOutlined,
} from '@ant-design/icons';
import { useAppStore, ToolType, AnnotationMode } from '../store/useAppStore';
import {
  uploadImages, startMode1Annotation, startMode2Annotation,
  startMode3Discovery, startMode3Select,
} from '../api';
import CategoryPanel from './CategoryPanel';

const { TextArea } = Input;

const Toolbar: React.FC = () => {
  const {
    currentMode, setCurrentMode, textPrompt, setTextPrompt,
    activeTool, setActiveTool,
    images, selectedImageId, bbox, taskStatus,
    maskOpacity, setMaskOpacity,
    discoveryTaskId, instanceMasks, selectedInstanceIds,
    categories, activeCategoryId, mode1CategoryId,
    mode2CategoryRefs, mode3CategoryRefs,
    manualTool, refImages,
    setDiscoveryTaskId, addImages, selectImage,
    setTask, resetTask, setMode1CategoryId,
    setActiveCategoryId, clearCategories,
    confirmMode2Bbox, clearBboxAfterAnnotate,
    setMode3CategoryInstances, setManualTool,
    clearImageMasks, clearRefImages, undo, redo,
  } = useAppStore();

  const fileInputRef = useRef<HTMLInputElement>(null);
  const isAnnotating = taskStatus === 'pending' || taskStatus === 'processing';
  const displayImageId = useAppStore.getState().viewingImageId || selectedImageId;

  const tools: { key: ToolType; icon: React.ReactNode; label: string }[] = [
    { key: 'select', icon: <SelectOutlined />, label: '框选工具' },
    { key: 'pan', icon: <DragOutlined />, label: '平移工具' },
    { key: 'zoom', icon: <ZoomInOutlined />, label: '缩放工具' },
  ];

  const handleFiles = useCallback(async (files: File[]) => {
    if (files.length === 0) return;
    try {
      message.loading({ content: `上传 ${files.length} 张...`, key: 'upload' });
      const results = await uploadImages(files);
      const newImages = results.map((r) => ({ id: r.image_id, filename: r.filename, url: r.url, path: r.path }));
      addImages(newImages);
      if (!selectedImageId && newImages.length > 0) selectImage(newImages[0].id);
      message.success({ content: `上传成功：${results.length} 张`, key: 'upload' });
    } catch (err: any) { message.error({ content: `上传失败：${err.message}`, key: 'upload' }); }
  }, [addImages, selectImage, selectedImageId]);

  const handleUploadClick = useCallback(() => fileInputRef.current?.click(), []);
  const handleFileChange = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    handleFiles(Array.from(e.target.files || []));
    if (fileInputRef.current) fileInputRef.current.value = '';
  }, [handleFiles]);
  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault(); e.stopPropagation();
    handleFiles(Array.from(e.dataTransfer.files).filter(f => f.type.startsWith('image/')));
  }, [handleFiles]);
  const handleDragOver = useCallback((e: React.DragEvent) => { e.preventDefault(); e.stopPropagation(); }, []);

  // ===== 模式1 =====
  const handleMode1Annotate = useCallback(async () => {
    const text = textPrompt.trim();
    if (!text) { message.warning('请输入文本提示'); return; }
    if (images.length === 0) { message.warning('请先上传图片'); return; }

    const cat = mode1CategoryId ? categories.find(c => c.id === mode1CategoryId) : null;
    const result = await startMode1Annotation({
      text_prompt: text,
      image_ids: images.map(i => i.id),
      image_paths: images.map(i => i.path),
      category_name: cat?.name,
      category_color: cat?.color,
    });
    setTask(result.task_id, 'pending');
    message.info(`模式1 已提交（${images.length} 张）`);
  }, [textPrompt, images, mode1CategoryId, categories, setTask]);

  // ===== 模式2（标注后清空 bbox 修复残留） =====
  const handleConfirmBbox = useCallback(() => {
    if (!bbox || bbox.width < 5 || bbox.height < 5) { message.warning('框选区域过小'); return; }
    if (!activeCategoryId) { message.warning('请先在底部添加类别'); return; }
    confirmMode2Bbox();
  }, [bbox, activeCategoryId, confirmMode2Bbox]);

  const handleMode2Annotate = useCallback(async () => {
    if (!selectedImageId) { message.warning('请先选择参考图'); return; }
    const refImage = images.find(i => i.id === selectedImageId);
    if (!refImage) return;
    // 参考图（主参考+图库中的）不作为标注目标；其余图片均为待标注目标
    const refIds = new Set([selectedImageId, ...refImages.map(r => r.imageId)]);
    const targetImages = images.filter(i => !refIds.has(i.id)).map(i => ({ id: i.id, path: i.path }));
    if (targetImages.length === 0) { message.warning('请上传更多图片（参考图不包含在标注目标中）'); return; }

    const hasCategories = categories.length > 0 && mode2CategoryRefs.length > 0;
    const hasSingleBbox = bbox && bbox.width > 0;
    if (!hasCategories && !hasSingleBbox) { message.warning('请先框选目标区域'); return; }

    let reqCategories = undefined;
    let reqBbox: [number, number, number, number] = [0, 0, 0, 0];
    let reqRefImages = undefined;
    let reqCategoryColor: string | undefined = undefined;

    if (hasCategories) {
      reqCategories = mode2CategoryRefs.map((ref) => {
        const cat = categories.find(c => c.id === ref.categoryId);
        return {
          name: cat?.name || 'unknown',
          color: cat?.color,
          bboxes: ref.bboxes.map(b => [b.x, b.y, b.x + b.width, b.y + b.height] as [number, number, number, number]),
        };
      });
    } else if (bbox) {
      const st = useAppStore.getState();
      const sc = st.imageFitScale || 1;
      const ox = st.imageFitOffsetX || 0;
      const oy = st.imageFitOffsetY || 0;
      const x1 = (bbox.x - ox) / sc;
      const y1 = (bbox.y - oy) / sc;
      const x2 = (bbox.x + bbox.width - ox) / sc;
      const y2 = (bbox.y + bbox.height - oy) / sc;
      reqBbox = [Math.max(0, x1), Math.max(0, y1), x2, y2];
      const activeCat = activeCategoryId ? categories.find(c => c.id === activeCategoryId) : null;
      reqCategoryColor = activeCat?.color;
    }

    // 参考图库：已有 bbox 的直接使用，无 bbox 但已有 Mask 的用全图作为参考
    if (refImages.length > 0) {
      reqRefImages = refImages.map(ri => {
        const img = images.find(i => i.id === ri.imageId);
        if (!img) return null;
        const hasMask = (useAppStore.getState().maskUrls[ri.imageId] || []).length > 0;
        if (ri.bbox) {
          return { path: img.path, bbox: [ri.bbox.x, ri.bbox.y, ri.bbox.x + ri.bbox.width, ri.bbox.y + ri.bbox.height] as [number, number, number, number], weight: ri.weight };
        } else if (hasMask) {
          // 已标注图片无需手动框选，使用全图区域作为参考
          return { path: img.path, bbox: [0, 0, 9999, 9999] as [number, number, number, number], weight: ri.weight };
        }
        return null;
      }).filter(Boolean) as any[];
    }

    const result = await startMode2Annotation({
      ref_image_id: refImage.id, ref_image_path: refImage.path,
      bbox: reqBbox, target_images: targetImages,
      categories: reqCategories, ref_images: reqRefImages,
      category_color: reqCategoryColor,
    });
    setTask(result.task_id, 'pending');
    clearBboxAfterAnnotate();

    const refCount = 1 + (reqRefImages?.length || 0);
    message.info(`模式2 已提交（${refCount} 张参考图，${targetImages.length} 张待标注）`);
  }, [selectedImageId, bbox, images, categories, mode2CategoryRefs, refImages, setTask, clearBboxAfterAnnotate]);

  // ===== 模式3 =====
  const handleMode3Discovery = useCallback(async () => {
    if (!selectedImageId) { message.warning('请先选择参考图'); return; }
    const refImage = images.find(i => i.id === selectedImageId);
    if (!refImage) return;
    const result = await startMode3Discovery({ ref_image_id: refImage.id, ref_image_path: refImage.path });
    setTask(result.task_id, 'pending');
    setDiscoveryTaskId(result.task_id);
  }, [selectedImageId, images, setTask, setDiscoveryTaskId]);

  const handleAssignInstances = useCallback(() => {
    if (selectedInstanceIds.length === 0 || !activeCategoryId) return;
    setMode3CategoryInstances(activeCategoryId, [...selectedInstanceIds]);
    const catName = categories.find(c => c.id === activeCategoryId)?.name || '';
    message.success(`已将 ${selectedInstanceIds.length} 个实例分配到「${catName}」`);
    useAppStore.getState().setSelectedInstanceIds([]);
  }, [selectedInstanceIds, activeCategoryId, categories, setMode3CategoryInstances]);

  const handleMode3Select = useCallback(async () => {
    if (!selectedImageId || !discoveryTaskId) return;
    const refImage = images.find(i => i.id === selectedImageId);
    if (!refImage) return;
    const refIds = new Set([selectedImageId, ...refImages.map(r => r.imageId)]);
    const targetImages = images.filter(i => !refIds.has(i.id)).map(i => ({ id: i.id, path: i.path }));
    if (targetImages.length === 0) { message.warning('请上传更多图片'); return; }

    const hasCategories = categories.length > 0 && mode3CategoryRefs.length > 0;
    if (!hasCategories && selectedInstanceIds.length === 0) { message.warning('请先选择实例'); return; }

    let reqCategories = undefined;
    let reqRefImages = undefined;

    if (hasCategories) {
      reqCategories = mode3CategoryRefs.map(ref => {
        const cat = categories.find(c => c.id === ref.categoryId);
        return { name: cat?.name || 'unknown', instance_ids: ref.instanceIds, color: cat?.color };
      });
    }
    if (refImages.length > 0) {
      reqRefImages = refImages.map(ri => {
        const img = images.find(i => i.id === ri.imageId);
        if (!img) return null;
        const hasMask = (useAppStore.getState().maskUrls[ri.imageId] || []).length > 0;
        if (ri.bbox) {
          return { path: img.path, bbox: [ri.bbox.x, ri.bbox.y, ri.bbox.x + ri.bbox.width, ri.bbox.y + ri.bbox.height] as [number, number, number, number], weight: ri.weight };
        } else if (hasMask) {
          return { path: img.path, bbox: [0, 0, 9999, 9999] as [number, number, number, number], weight: ri.weight };
        }
        return null;
      }).filter(Boolean) as any[];
    }

    const activeCat = activeCategoryId ? categories.find(c => c.id === activeCategoryId) : null;
    const result = await startMode3Select({
      discovery_task_id: discoveryTaskId,
      ref_image_id: refImage.id, ref_image_path: refImage.path,
      selected_instance_id: selectedInstanceIds[0] ?? 0,
      target_images: targetImages, categories: reqCategories, ref_images: reqRefImages,
      category_color: activeCat?.color,
    });
    setTask(result.task_id, 'pending');
  }, [selectedInstanceIds, selectedImageId, discoveryTaskId, images, categories, mode3CategoryRefs, refImages, setTask]);

  // 当前类别的确认框选数（当前图片）
  const currentRefCount = mode2CategoryRefs
    .filter(r => r.imageId === displayImageId)
    .reduce((sum, r) => sum + r.bboxes.length, 0);

  return (
    <div style={{ width: 230, background: '#1f1f1f', display: 'flex', flexDirection: 'column', padding: '12px', gap: 8, borderRight: '1px solid #333', overflow: 'auto' }}
      onDrop={handleDrop} onDragOver={handleDragOver}>
      <input ref={fileInputRef} type="file" multiple accept="image/*" style={{ display: 'none' }} onChange={handleFileChange} />
      <Button icon={<UploadOutlined />} onClick={handleUploadClick} block>上传图片</Button>
      <div style={{ color: '#555', fontSize: 11, textAlign: 'center' }}>或拖拽图片到此处</div>

      <Divider style={{ margin: '4px 0', borderColor: '#444' }} />
      <div style={{ color: '#999', fontSize: 12, marginBottom: 4 }}>标注模式</div>
      <Radio.Group value={currentMode} onChange={(e) => setCurrentMode(e.target.value as AnnotationMode)} buttonStyle="solid" size="small" style={{ width: '100%' }}>
        <Radio.Button value="mode1" style={{ width: '33.3%', textAlign: 'center', fontSize: 12 }}><FontSizeOutlined /> 文本</Radio.Button>
        <Radio.Button value="mode2" style={{ width: '33.3%', textAlign: 'center', fontSize: 12 }}><AimOutlined /> 框选</Radio.Button>
        <Radio.Button value="mode3" style={{ width: '33.4%', textAlign: 'center', fontSize: 12 }}><AppstoreOutlined /> 实例</Radio.Button>
      </Radio.Group>

      <Divider style={{ margin: '4px 0', borderColor: '#444' }} />

      {/* ===== 模式1 ===== */}
      {currentMode === 'mode1' && (
        <>
          <div style={{ color: '#999', fontSize: 12 }}>绑定类别（可选）</div>
          <Select
            value={mode1CategoryId || undefined}
            onChange={(v) => setMode1CategoryId(v || null)}
            allowClear
            placeholder="选择类别"
            size="small"
            style={{ width: '100%' }}
            options={categories.map(c => ({
              value: c.id,
              label: (
                <span style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                  <span style={{ width: 10, height: 10, borderRadius: 2, background: c.color, display: 'inline-block' }} />
                  {c.name}
                </span>
              ),
            }))}
          />
          <div style={{ color: '#999', fontSize: 12, marginTop: 4 }}>文本提示词</div>
          <TextArea
            value={textPrompt}
            onChange={(e) => setTextPrompt(e.target.value)}
            placeholder={'示例：\n  cat, dog\n  标注所有红色小轿车\n  识别画面左侧的白色杯子'}
            autoSize={{ minRows: 3, maxRows: 5 }}
            style={{ background: '#2a2a2a', borderColor: '#444', color: '#ddd' }}
          />
          <div style={{ color: '#666', fontSize: 10 }}>
            支持逗号分隔多类别，或完整描述句子
          </div>
          <Button type="primary" icon={<ThunderboltOutlined />} onClick={handleMode1Annotate}
            disabled={isAnnotating} loading={isAnnotating} block>
            {isAnnotating ? '标注中...' : '一键标注'}
          </Button>
        </>
      )}

      {/* ===== 模式2 ===== */}
      {currentMode === 'mode2' && (
        <>
          <div style={{ color: '#999', fontSize: 12 }}>标注类别（必选）</div>
          <Select
            value={activeCategoryId || undefined}
            onChange={(v) => setActiveCategoryId(v)}
            placeholder="请先选择/新建类别"
            size="small"
            style={{ width: '100%', marginBottom: 4 }}
            status={!activeCategoryId ? 'warning' : undefined}
            options={categories.map(c => ({
              value: c.id,
              label: <span style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                <span style={{ width: 10, height: 10, borderRadius: 2, background: c.color, display: 'inline-block' }} />{c.name}</span>,
            }))}
          />
          <div style={{ color: '#999', fontSize: 12 }}>文本提示补充（可选）</div>
          <Input
            value={useAppStore.getState().mode2TextHint}
            onChange={(e) => useAppStore.getState().setMode2TextHint(e.target.value)}
            placeholder="如：红色汽车、左侧物体"
            size="small"
            style={{ background: '#2a2a2a', borderColor: '#444', color: '#ddd', marginBottom: 4 }}
          />
          <div style={{ color: '#999', fontSize: 12, marginBottom: 4 }}>画布工具</div>
          <Space wrap>
            {tools.map(t => (
              <Tooltip key={t.key} title={t.label}>
                <Button type={activeTool === t.key && !manualTool ? 'primary' : 'default'} icon={t.icon}
                  onClick={() => setActiveTool(t.key)} size="small" />
              </Tooltip>
            ))}
          </Space>
          {bbox && bbox.width > 0 && (
            <div style={{ color: '#999', fontSize: 11, marginTop: 4 }}>
              框选: [{Math.round(bbox.x)},{Math.round(bbox.y)}]→[{Math.round(bbox.x+bbox.width)},{Math.round(bbox.y+bbox.height)}]
            </div>
          )}
          {categories.length > 0 && bbox && bbox.width > 5 && (
            <Button size="small" icon={<CheckOutlined />} onClick={handleConfirmBbox} block style={{ marginTop: 2 }}>
              确认到「{categories.find(c => c.id === activeCategoryId)?.name || ''}」
            </Button>
          )}
          {currentRefCount > 0 && (
            <div style={{ color: '#52c41a', fontSize: 11 }}>当前图已确认 {currentRefCount} 个框选</div>
          )}
          <Button type="primary" icon={<PlayCircleOutlined />} onClick={handleMode2Annotate}
            disabled={isAnnotating} loading={isAnnotating} block style={{ marginTop: 4 }}>
            {isAnnotating ? '标注中...' : '批量标注'}
          </Button>
          {refImages.length > 0 && (
            <div style={{ color: '#52c41a', fontSize: 11 }}>已选 {refImages.length} 张参考图</div>
          )}
        </>
      )}

      {/* ===== 模式3 ===== */}
      {currentMode === 'mode3' && (
        <>
          <div style={{ color: '#999', fontSize: 12 }}>标注类别（必选）</div>
          <Select
            value={activeCategoryId || undefined}
            onChange={(v) => setActiveCategoryId(v)}
            placeholder="请先选择/新建类别"
            size="small"
            style={{ width: '100%', marginBottom: 4 }}
            status={!activeCategoryId ? 'warning' : undefined}
            options={categories.map(c => ({
              value: c.id,
              label: <span style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                <span style={{ width: 10, height: 10, borderRadius: 2, background: c.color, display: 'inline-block' }} />{c.name}</span>,
            }))}
          />
          <Button icon={<SearchOutlined />} onClick={handleMode3Discovery}
            disabled={isAnnotating || !selectedImageId} block>生成实例</Button>
          {instanceMasks.length > 0 && (
            <>
              <div style={{ color: '#999', fontSize: 12, marginTop: 8 }}>选择实例（Ctrl+多选）</div>
              <div style={{ display: 'flex', flexWrap: 'wrap', gap: 4 }}>
                {instanceMasks.map(inst => {
                  const assignedRef = mode3CategoryRefs.find(r => r.instanceIds.includes(inst.id));
                  const assignedCat = assignedRef ? categories.find(c => c.id === assignedRef.categoryId) : null;
                  const isSelected = selectedInstanceIds.includes(inst.id);
                  const borderColor = assignedCat ? assignedCat.color : `rgb(${inst.color.join(',')})`;
                  return (
                    <Tooltip key={inst.id} title={assignedCat ? `已分配→${assignedCat.name}` : '未分配'}>
                      <Button size="small"
                        type={isSelected ? 'primary' : 'default'}
                        onClick={() => {
                          const ids = useAppStore.getState().selectedInstanceIds;
                          useAppStore.getState().setSelectedInstanceIds(
                            ids.includes(inst.id) ? ids.filter(i => i !== inst.id) : [...ids, inst.id]
                          );
                        }}
                        style={{
                          borderColor, minWidth: 40,
                          background: assignedCat && !isSelected ? `${assignedCat.color}22` : undefined,
                        }}>
                        #{inst.id}{assignedCat ? '✓' : ''}
                      </Button>
                    </Tooltip>
                  );
                })}
              </div>
              {categories.length > 0 && selectedInstanceIds.length > 0 && activeCategoryId && (
                <Button size="small" icon={<CheckOutlined />} onClick={handleAssignInstances} block style={{ marginTop: 2 }}>
                  分配到「{categories.find(c => c.id === activeCategoryId)?.name || ''}」
                </Button>
              )}
              {categories.length > 0 && selectedInstanceIds.length === 0 && mode3CategoryRefs.length > 0 && (
                <div style={{ fontSize: 11, color: '#52c41a', marginTop: 2 }}>
                  已分配 {mode3CategoryRefs.reduce((s, r) => s + r.instanceIds.length, 0)} 个实例，
                  可继续选择实例分配或手动标注补充
                </div>
              )}
              <Button type="primary" icon={<PlayCircleOutlined />} onClick={handleMode3Select}
                disabled={isAnnotating || (selectedInstanceIds.length === 0 && mode3CategoryRefs.length === 0)}
                loading={isAnnotating} block style={{ background: '#52c41a', borderColor: '#52c41a', marginTop: 4 }}>
                {isAnnotating ? '标注中...' : '跨图标注'}
              </Button>
            </>
          )}
        </>
      )}

      {/* ===== 手动标注工具 ===== */}
      <Divider style={{ margin: '4px 0', borderColor: '#444' }} />
      <div style={{ color: '#999', fontSize: 12 }}>手动标注（兜底修正）</div>
      {/* 手动标注类别选择器：确保手动 Mask 与自动标注同类别颜色一致 */}
      <Select
        value={activeCategoryId || undefined}
        onChange={(v) => setActiveCategoryId(v)}
        placeholder="选择类别（手动标注颜色）"
        size="small"
        style={{ width: '100%', marginBottom: 4 }}
        options={categories.map(c => ({
          value: c.id,
          label: <span style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
            <span style={{ width: 10, height: 10, borderRadius: 2, background: c.color, display: 'inline-block' }} />{c.name}</span>,
        }))}
      />
      {activeCategoryId && (() => {
        const cat = categories.find(c => c.id === activeCategoryId);
        return cat ? (
          <div style={{ fontSize: 11, display: 'flex', alignItems: 'center', gap: 4, marginBottom: 2 }}>
            <span style={{ width: 10, height: 10, borderRadius: 2, background: cat.color, display: 'inline-block', flexShrink: 0 }} />
            <span style={{ color: cat.color }}>手动标注 →「{cat.name}」颜色一致</span>
          </div>
        ) : null;
      })()}
      {!activeCategoryId && categories.length > 0 && (
        <div style={{ fontSize: 11, color: '#faad14', marginBottom: 2 }}>请选择类别后再手动标注</div>
      )}
      <Space wrap>
        <Tooltip title="矩形→SAM3 Mask">
          <Button size="small" icon={<BorderOutlined />}
            type={manualTool === 'rect_manual' ? 'primary' : 'default'}
            onClick={() => setManualTool(manualTool === 'rect_manual' ? null : 'rect_manual')}
            style={manualTool === 'rect_manual' ? { background: '#fa8c16', borderColor: '#fa8c16' } : {}} />
        </Tooltip>
        <Tooltip title="负向框选（排除区域）">
          <Button size="small" icon={<ClearOutlined />}
            type={manualTool === 'negative_box' ? 'primary' : 'default'}
            onClick={() => setManualTool(manualTool === 'negative_box' ? null : 'negative_box')}
            style={manualTool === 'negative_box' ? { background: '#ff4d4f', borderColor: '#ff4d4f' } : {}} />
        </Tooltip>
        <Tooltip title="撤销 Ctrl+Z"><Button size="small" icon={<UndoOutlined />} onClick={undo} /></Tooltip>
        <Tooltip title="重做 Ctrl+Shift+Z"><Button size="small" icon={<RedoOutlined />} onClick={redo} /></Tooltip>
        <Tooltip title="清空当前图标注">
          <Button size="small" danger icon={<ScissorOutlined />}
            onClick={() => {
              if (displayImageId) clearImageMasks(displayImageId);
              useAppStore.getState().clearNegativeBoxes();
            }} />
        </Tooltip>
      </Space>
      {manualTool === 'rect_manual' && (
        <div style={{ color: '#fa8c16', fontSize: 11, marginTop: 2 }}>画布框选→自动 SAM3 生成 Mask</div>
      )}
      {manualTool === 'negative_box' && (
        <div style={{ color: '#ff4d4f', fontSize: 11, marginTop: 2 }}>红色框选→排除该区域（负向提示）</div>
      )}

      {/* ===== 缩放控制 ===== */}
      <Divider style={{ margin: '4px 0', borderColor: '#444' }} />
      <div style={{ color: '#999', fontSize: 12 }}>视图缩放</div>
      <Space>
        <Tooltip title="放大"><Button size="small" icon={<ZoomInBtn />}
          onClick={() => useAppStore.getState().setStageScale(Math.min(5, useAppStore.getState().stageScale * 1.2))} /></Tooltip>
        <Tooltip title="缩小"><Button size="small" icon={<ZoomOutOutlined />}
          onClick={() => useAppStore.getState().setStageScale(Math.max(0.1, useAppStore.getState().stageScale / 1.2))} /></Tooltip>
        <Tooltip title="还原"><Button size="small" icon={<FullscreenOutlined />}
          onClick={() => { useAppStore.getState().setStageScale(1); useAppStore.getState().setStagePosition({ x: 0, y: 0 }); }} /></Tooltip>
      </Space>

      <div style={{ flex: 1 }} />

      <Divider style={{ margin: '4px 0', borderColor: '#444' }} />
      <div style={{ color: '#999', fontSize: 12 }}>Mask 透明度</div>
      <Slider min={0} max={1} step={0.05} value={maskOpacity} onChange={(v) => setMaskOpacity(v)}
        tooltip={{ formatter: (v) => `${Math.round((v || 0) * 100)}%` }} />

      {/* 全局类别面板 */}
      <CategoryPanel />

      <Button icon={<ClearOutlined />} onClick={() => { resetTask(); clearCategories(); clearRefImages(); }}
        block size="small" danger style={{ marginTop: 4 }}>重置全部</Button>
    </div>
  );
};

export default Toolbar;
