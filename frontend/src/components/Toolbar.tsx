/**
 * 左侧工具栏（v3 增强版）
 *
 * v3 新增：
 *   - 手动标注工具集（矩形框选触发 SAM3、删除/清空 Mask）
 *   - 参考图库信息展示
 *   - 多参考图 bbox 在不同参考图上绘制后提交
 */
import React, { useCallback, useRef } from 'react';
import { Button, Divider, Tooltip, Input, message, Radio, Space, Slider, Tag } from 'antd';
import {
  SelectOutlined, DragOutlined, ZoomInOutlined, PlayCircleOutlined,
  ClearOutlined, UploadOutlined, FontSizeOutlined, AimOutlined,
  ThunderboltOutlined, AppstoreOutlined, SearchOutlined,
  PlusOutlined, DeleteOutlined, CheckOutlined,
  BorderOutlined, UndoOutlined, RedoOutlined, ScissorOutlined,
} from '@ant-design/icons';
import { useAppStore, ToolType, AnnotationMode, ManualTool } from '../store/useAppStore';
import {
  uploadImages, startMode1Annotation, startMode2Annotation,
  startMode3Discovery, startMode3Select,
} from '../api';

const { TextArea } = Input;

const Toolbar: React.FC = () => {
  const {
    currentMode, setCurrentMode, textPrompt, setTextPrompt,
    activeTool, setActiveTool,
    images, selectedImageId, bbox, taskStatus,
    maskOpacity, setMaskOpacity,
    discoveryTaskId, instanceMasks, selectedInstanceIds,
    categories, activeCategoryId, mode2CategoryRefs, mode3CategoryRefs,
    manualTool, brushSize,
    refImages,
    setDiscoveryTaskId, addImages, selectImage,
    setTask, resetTask,
    addCategory, removeCategory, setActiveCategoryId, clearCategories,
    confirmMode2Bbox, setMode3CategoryInstances,
    setManualTool, setBrushSize,
    clearImageMasks, undo, redo,
    clearRefImages,
  } = useAppStore();

  const fileInputRef = useRef<HTMLInputElement>(null);
  const categoryInputRef = useRef<HTMLInputElement>(null);
  const isAnnotating = taskStatus === 'pending' || taskStatus === 'processing';

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

  const handleUploadClick = useCallback(() => { fileInputRef.current?.click(); }, []);
  const handleFileChange = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    handleFiles(Array.from(e.target.files || []));
    if (fileInputRef.current) fileInputRef.current.value = '';
  }, [handleFiles]);
  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault(); e.stopPropagation();
    handleFiles(Array.from(e.dataTransfer.files).filter(f => f.type.startsWith('image/')));
  }, [handleFiles]);
  const handleDragOver = useCallback((e: React.DragEvent) => { e.preventDefault(); e.stopPropagation(); }, []);

  const handleAddCategory = useCallback(() => {
    const name = categoryInputRef.current?.value?.trim();
    if (!name) { message.warning('请输入类别名称'); return; }
    addCategory(name);
    if (categoryInputRef.current) categoryInputRef.current.value = '';
  }, [addCategory]);

  const handleConfirmBbox = useCallback(() => {
    if (!bbox || bbox.width < 5 || bbox.height < 5) { message.warning('框选区域过小'); return; }
    if (!activeCategoryId) { message.warning('请先添加类别'); return; }
    confirmMode2Bbox();
  }, [bbox, activeCategoryId, confirmMode2Bbox]);

  const handleAssignInstances = useCallback(() => {
    if (selectedInstanceIds.length === 0) { message.warning('请先选择实例'); return; }
    if (!activeCategoryId) { message.warning('请先添加类别'); return; }
    setMode3CategoryInstances(activeCategoryId, [...selectedInstanceIds]);
    const catName = categories.find(c => c.id === activeCategoryId)?.name || '';
    message.success(`已分配 ${selectedInstanceIds.length} 个实例到「${catName}」`);
  }, [selectedInstanceIds, activeCategoryId, categories, setMode3CategoryInstances]);

  const handleMode1Annotate = useCallback(async () => {
    const text = textPrompt.trim();
    if (!text) { message.warning('请输入文本提示'); return; }
    if (images.length === 0) { message.warning('请先上传图片'); return; }
    const result = await startMode1Annotation({
      text_prompt: text, image_ids: images.map(i => i.id), image_paths: images.map(i => i.path),
    });
    setTask(result.task_id, 'pending');
    message.info(`模式1 已提交（${images.length} 张）`);
  }, [textPrompt, images, setTask]);

  const handleMode2Annotate = useCallback(async () => {
    if (!selectedImageId) { message.warning('请先选择参考图'); return; }
    const refImage = images.find(i => i.id === selectedImageId);
    if (!refImage) return;
    const targetImages = images.filter(i => i.id !== selectedImageId).map(i => ({ id: i.id, path: i.path }));
    if (targetImages.length === 0) { message.warning('请上传至少 2 张图片'); return; }
    const hasCategories = categories.length > 0 && mode2CategoryRefs.length > 0;
    const hasSingleBbox = bbox && bbox.width > 0;
    if (!hasCategories && !hasSingleBbox) { message.warning('请先框选目标区域'); return; }

    let reqCategories = undefined;
    let reqBbox: [number, number, number, number] = [0, 0, 0, 0];
    let reqRefImages = undefined;

    if (hasCategories) {
      reqCategories = mode2CategoryRefs.map((ref) => {
        const cat = categories.find(c => c.id === ref.categoryId);
        return { name: cat?.name || 'unknown', bboxes: ref.bboxes.map(b => [b.x, b.y, b.x + b.width, b.y + b.height] as [number, number, number, number]) };
      });
    } else if (bbox) {
      reqBbox = [bbox.x, bbox.y, bbox.x + bbox.width, bbox.y + bbox.height];
    }

    // 多参考图
    if (refImages.length > 0) {
      reqRefImages = refImages
        .filter(ri => ri.bbox)
        .map(ri => {
          const img = images.find(i => i.id === ri.imageId);
          return { path: img?.path || '', bbox: [ri.bbox!.x, ri.bbox!.y, ri.bbox!.x + ri.bbox!.width, ri.bbox!.y + ri.bbox!.height] as [number, number, number, number], weight: ri.weight };
        });
    }

    const result = await startMode2Annotation({
      ref_image_id: refImage.id, ref_image_path: refImage.path,
      bbox: reqBbox, target_images: targetImages,
      categories: reqCategories, ref_images: reqRefImages,
    });
    setTask(result.task_id, 'pending');
    message.info(`模式2 已提交（${targetImages.length} 张待标注）`);
  }, [selectedImageId, bbox, images, categories, mode2CategoryRefs, refImages, setTask]);

  const handleMode3Discovery = useCallback(async () => {
    if (!selectedImageId) { message.warning('请先选择参考图'); return; }
    const refImage = images.find(i => i.id === selectedImageId);
    if (!refImage) return;
    const result = await startMode3Discovery({ ref_image_id: refImage.id, ref_image_path: refImage.path });
    setTask(result.task_id, 'pending');
    setDiscoveryTaskId(result.task_id);
    message.info('正在生成粗分割实例...');
  }, [selectedImageId, images, setTask, setDiscoveryTaskId]);

  const handleMode3Select = useCallback(async () => {
    if (!selectedImageId || !discoveryTaskId) return;
    const refImage = images.find(i => i.id === selectedImageId);
    if (!refImage) return;
    const targetImages = images.filter(i => i.id !== selectedImageId).map(i => ({ id: i.id, path: i.path }));
    if (targetImages.length === 0) { message.warning('请上传至少 2 张图片'); return; }
    const hasCategories = categories.length > 0 && mode3CategoryRefs.length > 0;
    const hasSingleSelect = selectedInstanceIds.length > 0;
    if (!hasCategories && !hasSingleSelect) { message.warning('请先选择实例'); return; }

    let reqCategories = undefined;
    let reqRefImages = undefined;

    if (hasCategories) {
      reqCategories = mode3CategoryRefs.map(ref => {
        const cat = categories.find(c => c.id === ref.categoryId);
        return { name: cat?.name || 'unknown', instance_ids: ref.instanceIds };
      });
    }
    if (refImages.length > 0) {
      reqRefImages = refImages.filter(ri => ri.bbox).map(ri => {
        const img = images.find(i => i.id === ri.imageId);
        return { path: img?.path || '', bbox: [ri.bbox!.x, ri.bbox!.y, ri.bbox!.x + ri.bbox!.width, ri.bbox!.y + ri.bbox!.height] as [number, number, number, number], weight: ri.weight };
      });
    }

    const result = await startMode3Select({
      discovery_task_id: discoveryTaskId,
      ref_image_id: refImage.id, ref_image_path: refImage.path,
      selected_instance_id: selectedInstanceIds[0] ?? 0,
      target_images: targetImages, categories: reqCategories, ref_images: reqRefImages,
    });
    setTask(result.task_id, 'pending');
    message.info(`模式3 跨图标注已提交`);
  }, [selectedInstanceIds, selectedImageId, discoveryTaskId, images, categories, mode3CategoryRefs, refImages, setTask]);

  const displayImageId = useAppStore.getState().viewingImageId || selectedImageId;

  const renderCategoryManager = () => (
    <>
      <Divider style={{ margin: '4px 0', borderColor: '#444' }} />
      <div style={{ color: '#999', fontSize: 12 }}>多类别管理</div>
      <div style={{ display: 'flex', gap: 4, marginBottom: 4 }}>
        <input ref={categoryInputRef} type="text" placeholder="类别名"
          style={{ flex: 1, background: '#2a2a2a', border: '1px solid #444', color: '#ddd', padding: '2px 6px', borderRadius: 4, fontSize: 12 }}
          onKeyDown={(e) => e.key === 'Enter' && handleAddCategory()} />
        <Button size="small" icon={<PlusOutlined />} onClick={handleAddCategory} />
      </div>
      {categories.map(cat => {
        const isActive = cat.id === activeCategoryId;
        const refCount = currentMode === 'mode2'
          ? (mode2CategoryRefs.find(r => r.categoryId === cat.id)?.bboxes.length || 0)
          : (mode3CategoryRefs.find(r => r.categoryId === cat.id)?.instanceIds.length || 0);
        return (
          <div key={cat.id} onClick={() => setActiveCategoryId(cat.id)}
            style={{ display: 'flex', alignItems: 'center', gap: 4, padding: '2px 6px', borderRadius: 4, cursor: 'pointer',
              background: isActive ? '#333' : 'transparent', border: isActive ? `1px solid ${cat.color}` : '1px solid transparent' }}>
            <div style={{ width: 10, height: 10, borderRadius: 2, background: cat.color, flexShrink: 0 }} />
            <span style={{ color: '#ddd', fontSize: 12, flex: 1 }}>{cat.name}</span>
            {refCount > 0 && <Tag color="blue" style={{ fontSize: 10, padding: '0 3px', lineHeight: '16px' }}>{refCount}</Tag>}
            <Button type="text" size="small" danger icon={<DeleteOutlined style={{ fontSize: 10 }} />}
              onClick={(e) => { e.stopPropagation(); removeCategory(cat.id); }}
              style={{ width: 18, height: 18, padding: 0, minWidth: 18 }} />
          </div>
        );
      })}
    </>
  );

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

      {currentMode === 'mode1' && (
        <>
          <div style={{ color: '#999', fontSize: 12 }}>文本提示（逗号分隔多类别）</div>
          <TextArea value={textPrompt} onChange={(e) => setTextPrompt(e.target.value)} placeholder="person, car, dog"
            autoSize={{ minRows: 2, maxRows: 4 }} style={{ background: '#2a2a2a', borderColor: '#444', color: '#ddd' }} />
          <Button type="primary" icon={<ThunderboltOutlined />} onClick={handleMode1Annotate}
            disabled={isAnnotating} loading={isAnnotating} block>
            {isAnnotating ? '标注中...' : '一键标注'}
          </Button>
        </>
      )}

      {currentMode === 'mode2' && (
        <>
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
          <Button type="primary" icon={<PlayCircleOutlined />} onClick={handleMode2Annotate}
            disabled={isAnnotating} loading={isAnnotating} block style={{ marginTop: 4 }}>
            {isAnnotating ? '标注中...' : '批量标注'}
          </Button>
          {refImages.length > 0 && (
            <div style={{ color: '#52c41a', fontSize: 11 }}>已选 {refImages.length} 张参考图（右侧图钉管理）</div>
          )}
          {renderCategoryManager()}
        </>
      )}

      {currentMode === 'mode3' && (
        <>
          <Button icon={<SearchOutlined />} onClick={handleMode3Discovery}
            disabled={isAnnotating || !selectedImageId} block>生成实例</Button>
          {instanceMasks.length > 0 && (
            <>
              <div style={{ color: '#999', fontSize: 12, marginTop: 8 }}>选择实例（Ctrl+多选）</div>
              <div style={{ display: 'flex', flexWrap: 'wrap', gap: 4 }}>
                {instanceMasks.map(inst => (
                  <Button key={inst.id} size="small"
                    type={selectedInstanceIds.includes(inst.id) ? 'primary' : 'default'}
                    onClick={() => {
                      const ids = useAppStore.getState().selectedInstanceIds;
                      useAppStore.getState().setSelectedInstanceIds(
                        ids.includes(inst.id) ? ids.filter(i => i !== inst.id) : [...ids, inst.id]
                      );
                    }}
                    style={{ borderColor: `rgb(${inst.color.join(',')})`, minWidth: 40 }}>
                    #{inst.id}
                  </Button>
                ))}
              </div>
              {categories.length > 0 && selectedInstanceIds.length > 0 && (
                <Button size="small" icon={<CheckOutlined />} onClick={handleAssignInstances} block style={{ marginTop: 2 }}>
                  分配到「{categories.find(c => c.id === activeCategoryId)?.name || ''}」
                </Button>
              )}
              <Button type="primary" icon={<PlayCircleOutlined />} onClick={handleMode3Select}
                disabled={isAnnotating || (selectedInstanceIds.length === 0 && mode3CategoryRefs.length === 0)}
                loading={isAnnotating} block style={{ background: '#52c41a', borderColor: '#52c41a', marginTop: 4 }}>
                {isAnnotating ? '标注中...' : '跨图标注'}
              </Button>
            </>
          )}
          {renderCategoryManager()}
        </>
      )}

      {/* ===== 手动标注工具集 ===== */}
      <Divider style={{ margin: '4px 0', borderColor: '#444' }} />
      <div style={{ color: '#999', fontSize: 12 }}>手动标注（兜底修正）</div>
      <Space wrap>
        <Tooltip title="矩形框选 → SAM3 生成 Mask">
          <Button size="small" icon={<BorderOutlined />}
            type={manualTool === 'rect_manual' ? 'primary' : 'default'}
            onClick={() => setManualTool(manualTool === 'rect_manual' ? null : 'rect_manual')}
            style={manualTool === 'rect_manual' ? { background: '#fa8c16', borderColor: '#fa8c16' } : {}} />
        </Tooltip>
        <Tooltip title="撤销 (Ctrl+Z)">
          <Button size="small" icon={<UndoOutlined />} onClick={undo} />
        </Tooltip>
        <Tooltip title="重做 (Ctrl+Shift+Z)">
          <Button size="small" icon={<RedoOutlined />} onClick={redo} />
        </Tooltip>
        <Tooltip title="清空当前图 Mask">
          <Button size="small" danger icon={<ScissorOutlined />}
            onClick={() => { if (displayImageId) { clearImageMasks(displayImageId); message.info('已清空当前图标注'); } }} />
        </Tooltip>
      </Space>
      {manualTool === 'rect_manual' && (
        <div style={{ color: '#fa8c16', fontSize: 11, marginTop: 2 }}>
          在画布上框选区域 → 自动触发 SAM3 生成 Mask
        </div>
      )}

      <div style={{ flex: 1 }} />
      <Divider style={{ margin: '4px 0', borderColor: '#444' }} />
      <div style={{ color: '#999', fontSize: 12 }}>Mask 透明度</div>
      <Slider min={0} max={1} step={0.05} value={maskOpacity} onChange={(v) => setMaskOpacity(v)}
        tooltip={{ formatter: (v) => `${Math.round((v || 0) * 100)}%` }} />
      <Button icon={<ClearOutlined />} onClick={() => { resetTask(); clearCategories(); clearRefImages(); }}
        block size="small" danger>重置任务</Button>
    </div>
  );
};

export default Toolbar;
