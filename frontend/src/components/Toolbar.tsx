/**
 * 左侧工具栏
 * 功能：
 *   - 模式切换（模式1 文本标注 / 模式2 框选批量标注）
 *   - 模式1：文本输入框 + 一键标注按钮
 *   - 模式2：框选/平移/缩放工具 + 批量标注按钮
 *   - 通用：上传图片、重置任务
 */
import React, { useCallback } from 'react';
import { Button, Divider, Tooltip, Input, message, Radio, Space } from 'antd';
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
} from '@ant-design/icons';
import { useAppStore, ToolType, AnnotationMode } from '../store/useAppStore';
import { uploadImages, startMode1Annotation, startMode2Annotation } from '../api';

const { TextArea } = Input;

const Toolbar: React.FC = () => {
  const {
    currentMode, setCurrentMode,
    textPrompt, setTextPrompt,
    activeTool, setActiveTool,
    images, selectedImageId, bbox,
    taskStatus,
    addImages, selectImage,
    setTask, resetTask,
  } = useAppStore();

  const isAnnotating = taskStatus === 'pending' || taskStatus === 'processing';

  /** 工具按钮配置（模式2 使用） */
  const tools: { key: ToolType; icon: React.ReactNode; label: string }[] = [
    { key: 'select', icon: <SelectOutlined />, label: '框选工具' },
    { key: 'pan', icon: <DragOutlined />, label: '平移工具' },
    { key: 'zoom', icon: <ZoomInOutlined />, label: '缩放工具' },
  ];

  /** 上传图片 */
  const handleUpload = useCallback(() => {
    const input = document.createElement('input');
    input.type = 'file';
    input.multiple = true;
    input.accept = 'image/*';
    input.onchange = async (e) => {
      const files = Array.from((e.target as HTMLInputElement).files || []);
      if (files.length === 0) return;

      try {
        message.loading({ content: `正在上传 ${files.length} 张图片...`, key: 'upload' });
        const results = await uploadImages(files);
        const newImages = results.map((r) => ({
          id: r.image_id,
          filename: r.filename,
          url: r.url,
          path: r.path,
        }));
        addImages(newImages);
        if (!selectedImageId && newImages.length > 0) {
          selectImage(newImages[0].id);
        }
        message.success({ content: `上传成功：${results.length} 张`, key: 'upload' });
      } catch (err: any) {
        message.error({ content: `上传失败：${err.message}`, key: 'upload' });
      }
    };
    input.click();
  }, [addImages, selectImage, selectedImageId]);

  /** 模式1：文本提示一键标注 */
  const handleMode1Annotate = useCallback(async () => {
    const text = textPrompt.trim();
    if (!text) {
      message.warning('请输入文本提示（如：person, car, dog）');
      return;
    }
    if (images.length === 0) {
      message.warning('请先上传图片');
      return;
    }

    try {
      const result = await startMode1Annotation({
        text_prompt: text,
        image_ids: images.map((img) => img.id),
        image_paths: images.map((img) => img.path),
      });
      setTask(result.task_id, 'pending');
      message.info(`模式1 任务已提交（${images.length} 张图片，提示：${text}）`);
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

    const targetImages = images
      .filter((img) => img.id !== selectedImageId)
      .map((img) => ({ id: img.id, path: img.path }));

    if (targetImages.length === 0) {
      message.warning('请上传至少 2 张图片（1 张参考图 + N 张目标图）');
      return;
    }

    try {
      const bboxCoords: [number, number, number, number] = [
        bbox.x, bbox.y, bbox.x + bbox.width, bbox.y + bbox.height,
      ];

      const result = await startMode2Annotation({
        ref_image_id: refImage.id,
        ref_image_path: refImage.path,
        bbox: bboxCoords,
        target_images: targetImages,
      });

      setTask(result.task_id, 'pending');
      message.info(`模式2 任务已提交（${targetImages.length} 张待标注）`);
    } catch (err: any) {
      message.error(`标注失败：${err.message}`);
    }
  }, [selectedImageId, bbox, images, setTask]);

  return (
    <div style={{
      width: 220,
      background: '#1f1f1f',
      display: 'flex',
      flexDirection: 'column',
      padding: '12px',
      gap: 8,
      borderRight: '1px solid #333',
      overflow: 'auto',
    }}>
      {/* 上传按钮 */}
      <Button
        icon={<UploadOutlined />}
        onClick={handleUpload}
        block
        style={{ marginBottom: 4 }}
      >
        上传图片
      </Button>

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
        <Radio.Button value="mode1" style={{ width: '50%', textAlign: 'center' }}>
          <FontSizeOutlined /> 文本
        </Radio.Button>
        <Radio.Button value="mode2" style={{ width: '50%', textAlign: 'center' }}>
          <AimOutlined /> 框选
        </Radio.Button>
      </Radio.Group>

      <Divider style={{ margin: '4px 0', borderColor: '#444' }} />

      {/* ===== 模式1：文本标注 ===== */}
      {currentMode === 'mode1' && (
        <>
          <div style={{ color: '#999', fontSize: 12 }}>
            文本提示（逗号分隔多个类别）
          </div>
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

          {/* bbox 坐标显示 */}
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

      {/* 底部：重置 */}
      <div style={{ flex: 1 }} />
      <Divider style={{ margin: '4px 0', borderColor: '#444' }} />
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
