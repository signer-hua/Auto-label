/**
 * 左侧工具栏（增强版）
 *
 * v2 增强：
 *   - 模式2/3 多类别管理（添加/删除类别，每个类别独立绑定参考）
 *   - 模式2 框选确认到指定类别 + 多框选参考特征融合
 *   - 模式3 多实例选择分配到类别
 *   - 框选质量校验提示（面积过小/过大）
 */
import React, { useCallback, useRef } from 'react';
import { Button, Divider, Tooltip, Input, message, Radio, Space, Slider, Tag } from 'antd';
import {
  SelectOutlined,
  DragOutlined,
  ZoomInOutlined,
  PlayCircleOutlined,
  ClearOutlined,
  UploadOutlined,
  FontSizeOutlined,
  AimOutlined,
  ThunderboltOutlined,
  AppstoreOutlined,
  SearchOutlined,
  PlusOutlined,
  DeleteOutlined,
  CheckOutlined,
} from '@ant-design/icons';
import { useAppStore, ToolType, AnnotationMode } from '../store/useAppStore';
import {
  uploadImages,
  startMode1Annotation,
  startMode2Annotation,
  startMode3Discovery,
  startMode3Select,
} from '../api';

const { TextArea } = Input;

const Toolbar: React.FC = () => {
  const {
    currentMode, setCurrentMode,
    textPrompt, setTextPrompt,
    activeTool, setActiveTool,
    images, selectedImageId, bbox,
    taskStatus, taskId,
    maskOpacity, setMaskOpacity,
    discoveryTaskId, instanceMasks, selectedInstanceIds,
    categories, activeCategoryId,
    mode2CategoryRefs, mode3CategoryRefs,
    setDiscoveryTaskId,
    addImages, selectImage,
    setTask, resetTask,
    addCategory, removeCategory, setActiveCategoryId, clearCategories,
    confirmMode2Bbox,
    setMode3CategoryInstances,
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
      message.loading({ content: `正在上传 ${files.length} 张图片...`, key: 'upload' });
      const results = await uploadImages(files);
      const newImages = results.map((r) => ({
        id: r.image_id, filename: r.filename, url: r.url, path: r.path,
      }));
      addImages(newImages);
      if (!selectedImageId && newImages.length > 0) {
        selectImage(newImages[0].id);
      }
      message.success({ content: `上传成功：${results.length} 张`, key: 'upload' });
    } catch (err: any) {
      message.error({ content: `上传失败：${err.message}`, key: 'upload' });
    }
  }, [addImages, selectImage, selectedImageId]);

  const handleUploadClick = useCallback(() => {
    fileInputRef.current?.click();
  }, []);

  const handleFileChange = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const files = Array.from(e.target.files || []);
    handleFiles(files);
    if (fileInputRef.current) fileInputRef.current.value = '';
  }, [handleFiles]);

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    const files = Array.from(e.dataTransfer.files).filter(f => f.type.startsWith('image/'));
    handleFiles(files);
  }, [handleFiles]);

  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
  }, []);

  // ===== 类别管理 =====
  const handleAddCategory = useCallback(() => {
    const input = categoryInputRef.current;
    const name = input?.value?.trim();
    if (!name) {
      message.warning('请输入类别名称');
      return;
    }
    addCategory(name);
    if (input) input.value = '';
    message.success(`已添加类别「${name}」`);
  }, [addCategory]);

  // ===== 模式2：确认框选到当前类别 =====
  const handleConfirmBbox = useCallback(() => {
    if (!bbox || bbox.width < 5 || bbox.height < 5) {
      message.warning('框选区域过小，请重新框选');
      return;
    }
    if (!activeCategoryId) {
      message.warning('请先添加并选择一个类别');
      return;
    }
    confirmMode2Bbox();
    message.success('框选已确认到当前类别');
  }, [bbox, activeCategoryId, confirmMode2Bbox]);

  // ===== 模式3：将当前选中实例分配到活跃类别 =====
  const handleAssignInstances = useCallback(() => {
    if (selectedInstanceIds.length === 0) {
      message.warning('请先选择实例（Ctrl+点击多选）');
      return;
    }
    if (!activeCategoryId) {
      message.warning('请先添加并选择一个类别');
      return;
    }
    setMode3CategoryInstances(activeCategoryId, [...selectedInstanceIds]);
    const catName = categories.find(c => c.id === activeCategoryId)?.name || '';
    message.success(`已将 ${selectedInstanceIds.length} 个实例分配到「${catName}」`);
  }, [selectedInstanceIds, activeCategoryId, categories, setMode3CategoryInstances]);

  // ===== 模式1：文本提示一键标注 =====
  const handleMode1Annotate = useCallback(async () => {
    const text = textPrompt.trim();
    if (!text) { message.warning('请输入文本提示（如：person, car, dog）'); return; }
    if (images.length === 0) { message.warning('请先上传图片'); return; }
    try {
      const result = await startMode1Annotation({
        text_prompt: text,
        image_ids: images.map((img) => img.id),
        image_paths: images.map((img) => img.path),
      });
      setTask(result.task_id, 'pending');
      message.info(`模式1 任务已提交（${images.length} 张图片）`);
    } catch (err: any) {
      message.error(`标注失败：${err.message}`);
    }
  }, [textPrompt, images, setTask]);

  // ===== 模式2：框选批量标注（支持多类别） =====
  const handleMode2Annotate = useCallback(async () => {
    if (!selectedImageId) {
      message.warning('请先选择参考图');
      return;
    }
    const refImage = images.find((img) => img.id === selectedImageId);
    if (!refImage) return;
    const targetImages = images.filter((img) => img.id !== selectedImageId)
      .map((img) => ({ id: img.id, path: img.path }));
    if (targetImages.length === 0) {
      message.warning('请上传至少 2 张图片');
      return;
    }

    const hasCategories = categories.length > 0 && mode2CategoryRefs.length > 0;
    const hasSingleBbox = bbox && bbox.width > 0;

    if (!hasCategories && !hasSingleBbox) {
      message.warning('请先框选目标区域（或添加类别并确认框选）');
      return;
    }

    try {
      let reqCategories = undefined;
      let reqBbox: [number, number, number, number] = [0, 0, 0, 0];

      if (hasCategories) {
        reqCategories = mode2CategoryRefs.map((ref) => {
          const cat = categories.find((c) => c.id === ref.categoryId);
          return {
            name: cat?.name || 'unknown',
            bboxes: ref.bboxes.map((b) => [b.x, b.y, b.x + b.width, b.y + b.height] as [number, number, number, number]),
          };
        });
      } else if (bbox) {
        reqBbox = [bbox.x, bbox.y, bbox.x + bbox.width, bbox.y + bbox.height];
      }

      const result = await startMode2Annotation({
        ref_image_id: refImage.id, ref_image_path: refImage.path,
        bbox: reqBbox, target_images: targetImages,
        categories: reqCategories,
      });
      setTask(result.task_id, 'pending');
      const catCount = hasCategories ? mode2CategoryRefs.length : 1;
      message.info(`模式2 任务已提交（${catCount} 个类别，${targetImages.length} 张待标注）`);
    } catch (err: any) {
      message.error(`标注失败：${err.message}`);
    }
  }, [selectedImageId, bbox, images, categories, mode2CategoryRefs, setTask]);

  // ===== 模式3 阶段1：生成粗分割实例 =====
  const handleMode3Discovery = useCallback(async () => {
    if (!selectedImageId) { message.warning('请先选择参考图'); return; }
    const refImage = images.find((img) => img.id === selectedImageId);
    if (!refImage) return;
    try {
      const result = await startMode3Discovery({
        ref_image_id: refImage.id, ref_image_path: refImage.path,
      });
      setTask(result.task_id, 'pending');
      setDiscoveryTaskId(result.task_id);
      message.info('正在生成粗分割实例...');
    } catch (err: any) {
      message.error(`实例生成失败：${err.message}`);
    }
  }, [selectedImageId, images, setTask, setDiscoveryTaskId]);

  // ===== 模式3 阶段2：选中实例跨图标注（支持多类别） =====
  const handleMode3Select = useCallback(async () => {
    if (!selectedImageId || !discoveryTaskId) return;
    const refImage = images.find((img) => img.id === selectedImageId);
    if (!refImage) return;
    const targetImages = images.filter((img) => img.id !== selectedImageId)
      .map((img) => ({ id: img.id, path: img.path }));
    if (targetImages.length === 0) {
      message.warning('请上传至少 2 张图片');
      return;
    }

    const hasCategories = categories.length > 0 && mode3CategoryRefs.length > 0;
    const hasSingleSelect = selectedInstanceIds.length > 0;

    if (!hasCategories && !hasSingleSelect) {
      message.warning('请先选择实例（或添加类别并分配实例）');
      return;
    }

    try {
      let reqCategories = undefined;
      let singleInstanceId = selectedInstanceIds[0] ?? 0;

      if (hasCategories) {
        reqCategories = mode3CategoryRefs.map((ref) => {
          const cat = categories.find((c) => c.id === ref.categoryId);
          return {
            name: cat?.name || 'unknown',
            instance_ids: ref.instanceIds,
          };
        });
      }

      const result = await startMode3Select({
        discovery_task_id: discoveryTaskId,
        ref_image_id: refImage.id, ref_image_path: refImage.path,
        selected_instance_id: singleInstanceId,
        target_images: targetImages,
        categories: reqCategories,
      });
      setTask(result.task_id, 'pending');
      message.info(`模式3 跨图标注已提交（${targetImages.length} 张待标注）`);
    } catch (err: any) {
      message.error(`标注失败：${err.message}`);
    }
  }, [selectedInstanceIds, selectedImageId, discoveryTaskId, images, categories, mode3CategoryRefs, setTask]);

  // ===== 类别列表组件（模式2/3 共用） =====
  const renderCategoryManager = () => (
    <>
      <Divider style={{ margin: '4px 0', borderColor: '#444' }} />
      <div style={{ color: '#999', fontSize: 12 }}>多类别管理（可选）</div>
      <div style={{ display: 'flex', gap: 4, marginBottom: 4 }}>
        <input
          ref={categoryInputRef}
          type="text"
          placeholder="输入类别名"
          style={{
            flex: 1, background: '#2a2a2a', border: '1px solid #444',
            color: '#ddd', padding: '2px 6px', borderRadius: 4, fontSize: 12,
          }}
          onKeyDown={(e) => e.key === 'Enter' && handleAddCategory()}
        />
        <Button size="small" icon={<PlusOutlined />} onClick={handleAddCategory} />
      </div>
      {categories.length > 0 && (
        <div style={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
          {categories.map((cat) => {
            const isActive = cat.id === activeCategoryId;
            const m2ref = mode2CategoryRefs.find((r) => r.categoryId === cat.id);
            const m3ref = mode3CategoryRefs.find((r) => r.categoryId === cat.id);
            const refCount = currentMode === 'mode2'
              ? (m2ref?.bboxes.length || 0)
              : (m3ref?.instanceIds.length || 0);

            return (
              <div
                key={cat.id}
                onClick={() => setActiveCategoryId(cat.id)}
                style={{
                  display: 'flex', alignItems: 'center', gap: 4,
                  padding: '2px 6px', borderRadius: 4, cursor: 'pointer',
                  background: isActive ? '#333' : 'transparent',
                  border: isActive ? `1px solid ${cat.color}` : '1px solid transparent',
                }}
              >
                <div style={{
                  width: 10, height: 10, borderRadius: 2,
                  background: cat.color, flexShrink: 0,
                }} />
                <span style={{ color: '#ddd', fontSize: 12, flex: 1 }}>{cat.name}</span>
                {refCount > 0 && (
                  <Tag color="blue" style={{ fontSize: 10, padding: '0 3px', lineHeight: '16px' }}>
                    {refCount}
                  </Tag>
                )}
                <Button
                  type="text" size="small" danger
                  icon={<DeleteOutlined style={{ fontSize: 10 }} />}
                  onClick={(e) => { e.stopPropagation(); removeCategory(cat.id); }}
                  style={{ width: 18, height: 18, padding: 0, minWidth: 18 }}
                />
              </div>
            );
          })}
        </div>
      )}
    </>
  );

  return (
    <div
      style={{
        width: 230,
        background: '#1f1f1f',
        display: 'flex',
        flexDirection: 'column',
        padding: '12px',
        gap: 8,
        borderRight: '1px solid #333',
        overflow: 'auto',
      }}
      onDrop={handleDrop}
      onDragOver={handleDragOver}
    >
      <input
        ref={fileInputRef}
        type="file"
        multiple
        accept="image/*"
        style={{ display: 'none' }}
        onChange={handleFileChange}
      />

      <Button icon={<UploadOutlined />} onClick={handleUploadClick} block>
        上传图片
      </Button>
      <div style={{ color: '#555', fontSize: 11, textAlign: 'center' }}>
        或拖拽图片到此处
      </div>

      <Divider style={{ margin: '4px 0', borderColor: '#444' }} />

      <div style={{ color: '#999', fontSize: 12, marginBottom: 4 }}>标注模式</div>
      <Radio.Group
        value={currentMode}
        onChange={(e) => setCurrentMode(e.target.value as AnnotationMode)}
        buttonStyle="solid"
        size="small"
        style={{ width: '100%' }}
      >
        <Radio.Button value="mode1" style={{ width: '33.3%', textAlign: 'center', fontSize: 12 }}>
          <FontSizeOutlined /> 文本
        </Radio.Button>
        <Radio.Button value="mode2" style={{ width: '33.3%', textAlign: 'center', fontSize: 12 }}>
          <AimOutlined /> 框选
        </Radio.Button>
        <Radio.Button value="mode3" style={{ width: '33.4%', textAlign: 'center', fontSize: 12 }}>
          <AppstoreOutlined /> 实例
        </Radio.Button>
      </Radio.Group>

      <Divider style={{ margin: '4px 0', borderColor: '#444' }} />

      {/* ===== 模式1：文本标注 ===== */}
      {currentMode === 'mode1' && (
        <>
          <div style={{ color: '#999', fontSize: 12 }}>文本提示（逗号分隔多个类别）</div>
          <TextArea
            value={textPrompt}
            onChange={(e) => setTextPrompt(e.target.value)}
            placeholder="例如：person, car, dog"
            autoSize={{ minRows: 2, maxRows: 4 }}
            style={{ background: '#2a2a2a', borderColor: '#444', color: '#ddd' }}
          />
          <Button
            type="primary"
            icon={<ThunderboltOutlined />}
            onClick={handleMode1Annotate}
            disabled={isAnnotating}
            loading={isAnnotating}
            block
          >
            {isAnnotating ? '标注中...' : '一键标注'}
          </Button>
          <div style={{ color: '#666', fontSize: 11 }}>
            Grounding DINO 检测 → SAM3 精准分割
          </div>
        </>
      )}

      {/* ===== 模式2：框选批量标注 ===== */}
      {currentMode === 'mode2' && (
        <>
          <div style={{ color: '#999', fontSize: 12, marginBottom: 4 }}>画布工具</div>
          <Space wrap>
            {tools.map((tool) => (
              <Tooltip key={tool.key} title={tool.label}>
                <Button
                  type={activeTool === tool.key ? 'primary' : 'default'}
                  icon={tool.icon}
                  onClick={() => setActiveTool(tool.key)}
                  size="small"
                />
              </Tooltip>
            ))}
          </Space>

          {bbox && bbox.width > 0 && (
            <div style={{ color: '#999', fontSize: 11, marginTop: 4 }}>
              框选: [{Math.round(bbox.x)}, {Math.round(bbox.y)}] →
              [{Math.round(bbox.x + bbox.width)}, {Math.round(bbox.y + bbox.height)}]
              {bbox.width < 20 || bbox.height < 20 ? (
                <span style={{ color: '#ff4d4f' }}> ⚠ 区域过小</span>
              ) : null}
            </div>
          )}

          {/* 多类别：确认框选到类别 */}
          {categories.length > 0 && bbox && bbox.width > 5 && (
            <Button
              size="small"
              icon={<CheckOutlined />}
              onClick={handleConfirmBbox}
              block
              style={{ marginTop: 2 }}
            >
              确认框选到「{categories.find(c => c.id === activeCategoryId)?.name || ''}」
            </Button>
          )}

          <Button
            type="primary"
            icon={<PlayCircleOutlined />}
            onClick={handleMode2Annotate}
            disabled={isAnnotating}
            loading={isAnnotating}
            block
            style={{ marginTop: 4 }}
          >
            {isAnnotating ? '标注中...' : '批量标注'}
          </Button>
          <div style={{ color: '#666', fontSize: 11 }}>
            SAM3 分割 → DINOv3 匹配 → 批量 SAM3
          </div>

          {renderCategoryManager()}
        </>
      )}

      {/* ===== 模式3：选实例跨图标注 ===== */}
      {currentMode === 'mode3' && (
        <>
          <div style={{ color: '#999', fontSize: 12 }}>
            步骤1：在参考图上生成粗分割实例
          </div>
          <Button
            icon={<SearchOutlined />}
            onClick={handleMode3Discovery}
            disabled={isAnnotating || !selectedImageId}
            loading={taskStatus === 'pending' && !instanceMasks.length}
            block
          >
            生成实例
          </Button>

          {instanceMasks.length > 0 && (
            <>
              <div style={{ color: '#999', fontSize: 12, marginTop: 8 }}>
                步骤2：选择目标实例（Ctrl+点击多选）
              </div>
              <div style={{ display: 'flex', flexWrap: 'wrap', gap: 4 }}>
                {instanceMasks.map((inst) => (
                  <Button
                    key={inst.id}
                    size="small"
                    type={selectedInstanceIds.includes(inst.id) ? 'primary' : 'default'}
                    onClick={() => {
                      const ids = useAppStore.getState().selectedInstanceIds;
                      if (ids.includes(inst.id)) {
                        useAppStore.getState().setSelectedInstanceIds(ids.filter(i => i !== inst.id));
                      } else {
                        useAppStore.getState().setSelectedInstanceIds([...ids, inst.id]);
                      }
                    }}
                    style={{
                      borderColor: `rgb(${inst.color.join(',')})`,
                      minWidth: 40,
                    }}
                  >
                    #{inst.id}
                  </Button>
                ))}
              </div>

              {selectedInstanceIds.length > 0 && (
                <div style={{ color: '#52c41a', fontSize: 11, marginTop: 2 }}>
                  已选 {selectedInstanceIds.length} 个实例
                </div>
              )}

              {/* 多类别：分配实例到类别 */}
              {categories.length > 0 && selectedInstanceIds.length > 0 && (
                <Button
                  size="small"
                  icon={<CheckOutlined />}
                  onClick={handleAssignInstances}
                  block
                  style={{ marginTop: 2 }}
                >
                  分配到「{categories.find(c => c.id === activeCategoryId)?.name || ''}」
                </Button>
              )}

              <div style={{ color: '#999', fontSize: 12, marginTop: 8 }}>
                步骤3：跨图批量标注
              </div>
              <Button
                type="primary"
                icon={<PlayCircleOutlined />}
                onClick={handleMode3Select}
                disabled={isAnnotating || (selectedInstanceIds.length === 0 && mode3CategoryRefs.length === 0)}
                loading={isAnnotating}
                block
                style={{ background: '#52c41a', borderColor: '#52c41a' }}
              >
                {isAnnotating ? '标注中...' : '跨图标注'}
              </Button>
            </>
          )}

          <div style={{ color: '#666', fontSize: 11, marginTop: 4 }}>
            DINOv3 聚类 → SAM3 粗分割 → 选中 → 跨图匹配
          </div>

          {renderCategoryManager()}
        </>
      )}

      <div style={{ flex: 1 }} />

      <Divider style={{ margin: '4px 0', borderColor: '#444' }} />

      <div style={{ color: '#999', fontSize: 12 }}>Mask 透明度</div>
      <Slider
        min={0}
        max={1}
        step={0.05}
        value={maskOpacity}
        onChange={(v) => setMaskOpacity(v)}
        tooltip={{ formatter: (v) => `${Math.round((v || 0) * 100)}%` }}
      />

      <Button
        icon={<ClearOutlined />}
        onClick={() => { resetTask(); clearCategories(); }}
        block
        size="small"
        danger
      >
        重置任务
      </Button>
    </div>
  );
};

export default Toolbar;
