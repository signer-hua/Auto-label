/**
 * 左侧工具栏
 * 功能：
 *   - 三模式切换（模式1 文本 / 模式2 框选 / 模式3 选实例）
 *   - 模式1：文本输入框 + 一键标注
 *   - 模式2：框选/平移/缩放工具 + 批量标注
 *   - 模式3：生成实例 + 选中实例 + 跨图标注
 *   - 通用：拖拽上传、Mask 透明度调节、重置
 */
import React, { useCallback, useRef } from 'react';
import { Button, Divider, Tooltip, Input, message, Radio, Space, Slider } from 'antd';
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
    discoveryTaskId, instanceMasks, selectedInstanceId,
    setDiscoveryTaskId, setSelectedInstanceId,
    addImages, selectImage,
    setTask, resetTask,
  } = useAppStore();

  const fileInputRef = useRef<HTMLInputElement>(null);
  const isAnnotating = taskStatus === 'pending' || taskStatus === 'processing';

  /** 工具按钮配置（模式2/3 使用） */
  const tools: { key: ToolType; icon: React.ReactNode; label: string }[] = [
    { key: 'select', icon: <SelectOutlined />, label: '框选工具' },
    { key: 'pan', icon: <DragOutlined />, label: '平移工具' },
    { key: 'zoom', icon: <ZoomInOutlined />, label: '缩放工具' },
  ];

  /** 上传图片（按钮或拖拽） */
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

  /** 拖拽上传 */
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

  /** 模式1：文本提示一键标注 */
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

  /** 模式2：框选批量标注 */
  const handleMode2Annotate = useCallback(async () => {
    if (!selectedImageId || !bbox) {
      message.warning('请先选择参考图并框选目标区域');
      return;
    }
    const refImage = images.find((img) => img.id === selectedImageId);
    if (!refImage) return;
    const targetImages = images.filter((img) => img.id !== selectedImageId).map((img) => ({ id: img.id, path: img.path }));
    if (targetImages.length === 0) {
      message.warning('请上传至少 2 张图片');
      return;
    }
    try {
      const bboxCoords: [number, number, number, number] = [
        bbox.x, bbox.y, bbox.x + bbox.width, bbox.y + bbox.height,
      ];
      const result = await startMode2Annotation({
        ref_image_id: refImage.id, ref_image_path: refImage.path,
        bbox: bboxCoords, target_images: targetImages,
      });
      setTask(result.task_id, 'pending');
      message.info(`模式2 任务已提交（${targetImages.length} 张待标注）`);
    } catch (err: any) {
      message.error(`标注失败：${err.message}`);
    }
  }, [selectedImageId, bbox, images, setTask]);

  /** 模式3 阶段1：生成粗分割实例 */
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

  /** 模式3 阶段2：选中实例跨图标注 */
  const handleMode3Select = useCallback(async () => {
    if (selectedInstanceId === null) { message.warning('请先选择一个实例'); return; }
    if (!selectedImageId || !discoveryTaskId) return;
    const refImage = images.find((img) => img.id === selectedImageId);
    if (!refImage) return;
    const targetImages = images.filter((img) => img.id !== selectedImageId).map((img) => ({ id: img.id, path: img.path }));
    if (targetImages.length === 0) {
      message.warning('请上传至少 2 张图片');
      return;
    }
    try {
      const result = await startMode3Select({
        discovery_task_id: discoveryTaskId,
        ref_image_id: refImage.id, ref_image_path: refImage.path,
        selected_instance_id: selectedInstanceId,
        target_images: targetImages,
      });
      setTask(result.task_id, 'pending');
      message.info(`模式3 跨图标注已提交（${targetImages.length} 张待标注）`);
    } catch (err: any) {
      message.error(`标注失败：${err.message}`);
    }
  }, [selectedInstanceId, selectedImageId, discoveryTaskId, images, setTask]);

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
      {/* 隐藏的文件输入 */}
      <input
        ref={fileInputRef}
        type="file"
        multiple
        accept="image/*"
        style={{ display: 'none' }}
        onChange={handleFileChange}
      />

      {/* 上传按钮 */}
      <Button icon={<UploadOutlined />} onClick={handleUploadClick} block>
        上传图片
      </Button>
      <div style={{ color: '#555', fontSize: 11, textAlign: 'center' }}>
        或拖拽图片到此处
      </div>

      <Divider style={{ margin: '4px 0', borderColor: '#444' }} />

      {/* 模式切换 */}
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
            YOLO-World 检测 → SAM3 精准分割
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
            </div>
          )}
          <Button
            type="primary"
            icon={<PlayCircleOutlined />}
            onClick={handleMode2Annotate}
            disabled={isAnnotating || !bbox}
            loading={isAnnotating}
            block
            style={{ marginTop: 4 }}
          >
            {isAnnotating ? '标注中...' : '批量标注'}
          </Button>
          <div style={{ color: '#666', fontSize: 11 }}>
            SAM3 分割 → DINOv3 匹配 → 批量 SAM3
          </div>
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

          {/* 实例选择列表 */}
          {instanceMasks.length > 0 && (
            <>
              <div style={{ color: '#999', fontSize: 12, marginTop: 8 }}>
                步骤2：选择目标实例（点击选中）
              </div>
              <div style={{ display: 'flex', flexWrap: 'wrap', gap: 4 }}>
                {instanceMasks.map((inst) => (
                  <Button
                    key={inst.id}
                    size="small"
                    type={selectedInstanceId === inst.id ? 'primary' : 'default'}
                    onClick={() => setSelectedInstanceId(inst.id)}
                    style={{
                      borderColor: `rgb(${inst.color.join(',')})`,
                      minWidth: 40,
                    }}
                  >
                    #{inst.id}
                  </Button>
                ))}
              </div>

              <div style={{ color: '#999', fontSize: 12, marginTop: 8 }}>
                步骤3：跨图批量标注
              </div>
              <Button
                type="primary"
                icon={<PlayCircleOutlined />}
                onClick={handleMode3Select}
                disabled={isAnnotating || selectedInstanceId === null}
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
        </>
      )}

      {/* 底部通用区域 */}
      <div style={{ flex: 1 }} />

      <Divider style={{ margin: '4px 0', borderColor: '#444' }} />

      {/* Mask 透明度调节 */}
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
        onClick={resetTask}
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
