/**
 * 左侧工具栏
 * 工具切换（框选/平移/缩放）、批量标注触发、重置。
 */
import React, { useCallback } from 'react';
import { Button, Divider, Tooltip, Space, message } from 'antd';
import {
  SelectOutlined,
  DragOutlined,
  ZoomInOutlined,
  PlayCircleOutlined,
  ClearOutlined,
  UploadOutlined,
} from '@ant-design/icons';
import { useAppStore, ToolType } from '../store/useAppStore';
import { uploadImages, startMode2Annotation } from '../api';

const Toolbar: React.FC = () => {
  const {
    activeTool, setActiveTool,
    images, selectedImageId, bbox,
    taskStatus,
    addImages, selectImage,
    setTask, resetTask,
  } = useAppStore();

  /** 工具按钮配置 */
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
        // 自动选中第一张
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

  /** 触发批量标注 */
  const handleStartAnnotation = useCallback(async () => {
    if (!selectedImageId || !bbox) {
      message.warning('请先选择参考图并框选目标区域');
      return;
    }

    const refImage = images.find((img) => img.id === selectedImageId);
    if (!refImage) return;

    // 目标图 = 除参考图外的所有图
    const targetImages = images
      .filter((img) => img.id !== selectedImageId)
      .map((img) => ({ id: img.id, path: img.path }));

    if (targetImages.length === 0) {
      message.warning('请上传至少 2 张图片（1 张参考图 + N 张目标图）');
      return;
    }

    try {
      // bbox 转为 [x1, y1, x2, y2]
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
      message.info(`任务已提交 (${targetImages.length} 张待标注)`);
    } catch (err: any) {
      message.error(`标注失败：${err.message}`);
    }
  }, [selectedImageId, bbox, images, setTask]);

  const isAnnotating = taskStatus === 'pending' || taskStatus === 'processing';

  return (
    <div style={{
      width: 64,
      background: '#1f1f1f',
      display: 'flex',
      flexDirection: 'column',
      alignItems: 'center',
      padding: '12px 0',
      gap: 4,
    }}>
      {/* 上传按钮 */}
      <Tooltip title="上传图片" placement="right">
        <Button
          type="text"
          icon={<UploadOutlined style={{ color: '#fff', fontSize: 20 }} />}
          onClick={handleUpload}
          style={{ width: 48, height: 48 }}
        />
      </Tooltip>

      <Divider style={{ margin: '8px 0', borderColor: '#444', minWidth: 40 }} />

      {/* 工具按钮 */}
      {tools.map((tool) => (
        <Tooltip key={tool.key} title={tool.label} placement="right">
          <Button
            type="text"
            icon={React.cloneElement(tool.icon as React.ReactElement, {
              style: {
                color: activeTool === tool.key ? '#1890ff' : '#aaa',
                fontSize: 20,
              },
            })}
            onClick={() => setActiveTool(tool.key)}
            style={{
              width: 48,
              height: 48,
              background: activeTool === tool.key ? '#333' : 'transparent',
              borderRadius: 8,
            }}
          />
        </Tooltip>
      ))}

      <Divider style={{ margin: '8px 0', borderColor: '#444', minWidth: 40 }} />

      {/* 批量标注按钮 */}
      <Tooltip title="批量标注" placement="right">
        <Button
          type="text"
          icon={<PlayCircleOutlined style={{
            color: isAnnotating ? '#faad14' : '#52c41a',
            fontSize: 22,
          }} />}
          onClick={handleStartAnnotation}
          disabled={isAnnotating}
          style={{ width: 48, height: 48 }}
        />
      </Tooltip>

      {/* 重置 */}
      <Tooltip title="重置任务" placement="right">
        <Button
          type="text"
          icon={<ClearOutlined style={{ color: '#aaa', fontSize: 18 }} />}
          onClick={resetTask}
          style={{ width: 48, height: 48 }}
        />
      </Tooltip>
    </div>
  );
};

export default Toolbar;
