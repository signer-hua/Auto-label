/**
 * 右侧面板
 * 包含：图片列表缩略图（含删除/设为参考图）、批量进度条、一键导出按钮。
 */
import React, { useEffect, useRef, useCallback } from 'react';
import { Card, Progress, Button, Tag, Typography, message, Empty, Popconfirm } from 'antd';
import {
  DownloadOutlined,
  CheckCircleOutlined,
  DeleteOutlined,
  StarOutlined,
  StarFilled,
} from '@ant-design/icons';
import { useAppStore } from '../store/useAppStore';
import { getTaskStatus, deleteImage, listImages } from '../api';

const { Text } = Typography;

const RightPanel: React.FC = () => {
  const {
    images, selectedImageId, viewingImageId,
    taskId, taskStatus, taskProgress, taskTotal, taskMessage,
    maskUrls, exportUrl,
    selectImage, setViewingImage, removeImage, setImages, addImages,
    updateTaskProgress, setTaskStatus, setMaskUrls, setExportUrl,
  } = useAppStore();

  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const initRef = useRef(false);

  // ==================== 页面加载时恢复图片列表 ====================
  useEffect(() => {
    if (initRef.current) return;
    initRef.current = true;

    listImages()
      .then((serverImages) => {
        if (serverImages.length > 0 && images.length === 0) {
          const items = serverImages.map((img) => ({
            id: img.image_id,
            filename: img.filename,
            url: img.url,
            path: img.path,
          }));
          addImages(items);
          selectImage(items[0].id);
        }
      })
      .catch(() => {
        // 后端未启动时静默忽略
      });
  }, []);

  // ==================== 任务状态轮询 ====================
  useEffect(() => {
    if (pollRef.current) {
      clearInterval(pollRef.current);
      pollRef.current = null;
    }

    if (!taskId || (taskStatus !== 'pending' && taskStatus !== 'processing')) {
      return;
    }

    pollRef.current = setInterval(async () => {
      try {
        const res = await getTaskStatus(taskId);
        updateTaskProgress(res.progress, res.total, res.message);

        if (res.status === 'success') {
          setTaskStatus('success', res.message);
          if (res.mask_urls) setMaskUrls(res.mask_urls);
          if (res.export_url) setExportUrl(res.export_url);
          message.success('批量标注完成！');
          if (pollRef.current) clearInterval(pollRef.current);
        } else if (res.status === 'failed') {
          setTaskStatus('failed', res.message);
          message.error(`标注失败：${res.message}`);
          if (pollRef.current) clearInterval(pollRef.current);
        }
      } catch (err: any) {
        console.error('Poll error:', err);
      }
    }, 1000);

    return () => {
      if (pollRef.current) clearInterval(pollRef.current);
    };
  }, [taskId, taskStatus]);

  /** 删除图片 */
  const handleDelete = useCallback(async (imageId: string, e: React.MouseEvent) => {
    e.stopPropagation(); // 阻止触发点击选中
    try {
      await deleteImage(imageId);
      removeImage(imageId);
      message.success('已删除');
    } catch (err: any) {
      message.error(`删除失败：${err.message}`);
    }
  }, [removeImage]);

  /** 设为参考图 */
  const handleSetRef = useCallback((imageId: string, e: React.MouseEvent) => {
    e.stopPropagation();
    selectImage(imageId);
    message.info('已设为参考图');
  }, [selectImage]);

  /** 下载 COCO 导出 */
  const handleExport = useCallback(() => {
    if (exportUrl) {
      window.open(exportUrl, '_blank');
    }
  }, [exportUrl]);

  const progressPercent = taskTotal > 0 ? Math.round((taskProgress / taskTotal) * 100) : 0;

  const statusColor: Record<string, string> = {
    idle: 'default', pending: 'orange', processing: 'blue', success: 'green', failed: 'red',
  };

  return (
    <div style={{
      width: 280,
      background: '#1a1a1a',
      display: 'flex',
      flexDirection: 'column',
      borderLeft: '1px solid #333',
      overflow: 'hidden',
    }}>
      {/* 任务状态区域 */}
      <Card
        size="small"
        title="任务状态"
        style={{ background: '#222', borderBottom: '1px solid #333' }}
        headStyle={{ color: '#ddd', borderBottom: '1px solid #333' }}
        bodyStyle={{ padding: '12px' }}
        extra={<Tag color={statusColor[taskStatus]}>{taskStatus}</Tag>}
      >
        {taskStatus !== 'idle' && (
          <>
            <Progress
              percent={progressPercent}
              size="small"
              status={taskStatus === 'failed' ? 'exception' : taskStatus === 'success' ? 'success' : 'active'}
              format={() => `${taskProgress}/${taskTotal}`}
            />
            <Text style={{ color: '#999', fontSize: 12, display: 'block', marginTop: 4 }}>
              {taskMessage}
            </Text>
          </>
        )}

        {taskStatus === 'success' && exportUrl && (
          <Button
            type="primary"
            icon={<DownloadOutlined />}
            size="small"
            onClick={handleExport}
            style={{ marginTop: 8, width: '100%' }}
          >
            导出 COCO JSON
          </Button>
        )}
      </Card>

      {/* 图片列表 */}
      <div style={{ flex: 1, overflow: 'auto', padding: '8px' }}>
        <Text style={{ color: '#999', fontSize: 12, marginBottom: 8, display: 'block' }}>
          图片列表 ({images.length})
        </Text>

        {images.length === 0 ? (
          <Empty
            description={<Text style={{ color: '#666' }}>暂无图片</Text>}
            image={Empty.PRESENTED_IMAGE_SIMPLE}
          />
        ) : (
          images.map((img) => {
            const isRef = img.id === selectedImageId;
            const isViewing = img.id === viewingImageId;
            const hasMask = (maskUrls[img.id] || []).length > 0;

            return (
              <div
                key={img.id}
                onClick={() => setViewingImage(img.id)}
                style={{
                  display: 'flex',
                  alignItems: 'center',
                  gap: 8,
                  padding: '6px 8px',
                  marginBottom: 4,
                  borderRadius: 6,
                  cursor: 'pointer',
                  background: isViewing ? '#333' : 'transparent',
                  border: isRef ? '1px solid #1890ff' : '1px solid transparent',
                  position: 'relative',
                }}
              >
                {/* 缩略图 */}
                <img
                  src={img.url}
                  alt={img.filename}
                  style={{
                    width: 48,
                    height: 48,
                    objectFit: 'cover',
                    borderRadius: 4,
                    flexShrink: 0,
                  }}
                />

                {/* 文件名 + 标签 */}
                <div style={{ flex: 1, minWidth: 0 }}>
                  <Text
                    ellipsis
                    style={{ color: '#ddd', fontSize: 12, display: 'block' }}
                  >
                    {img.filename}
                  </Text>
                  <div style={{ display: 'flex', gap: 4, marginTop: 2, flexWrap: 'wrap' }}>
                    {isRef && (
                      <Tag color="blue" style={{ fontSize: 10, lineHeight: '16px', padding: '0 4px' }}>
                        参考图
                      </Tag>
                    )}
                    {hasMask && (
                      <Tag color="green" style={{ fontSize: 10, lineHeight: '16px', padding: '0 4px' }}>
                        <CheckCircleOutlined /> {maskUrls[img.id].length}
                      </Tag>
                    )}
                  </div>
                </div>

                {/* 操作按钮 */}
                <div style={{ display: 'flex', flexDirection: 'column', gap: 2, flexShrink: 0 }}>
                  {/* 设为参考图 */}
                  <Button
                    type="text"
                    size="small"
                    icon={isRef
                      ? <StarFilled style={{ color: '#faad14', fontSize: 14 }} />
                      : <StarOutlined style={{ color: '#666', fontSize: 14 }} />
                    }
                    onClick={(e) => handleSetRef(img.id, e)}
                    title="设为参考图"
                    style={{ width: 24, height: 24, padding: 0, minWidth: 24 }}
                  />
                  {/* 删除 */}
                  <Popconfirm
                    title="确定删除此图片？"
                    onConfirm={(e) => handleDelete(img.id, e as any)}
                    okText="删除"
                    cancelText="取消"
                    placement="left"
                  >
                    <Button
                      type="text"
                      size="small"
                      danger
                      icon={<DeleteOutlined style={{ fontSize: 14 }} />}
                      onClick={(e) => e.stopPropagation()}
                      title="删除图片"
                      style={{ width: 24, height: 24, padding: 0, minWidth: 24 }}
                    />
                  </Popconfirm>
                </div>
              </div>
            );
          })
        )}
      </div>
    </div>
  );
};

export default RightPanel;
