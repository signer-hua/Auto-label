/**
 * 右侧面板
 * 包含：
 *   - 任务状态区域（进度条 + 暂停/恢复/取消按钮 + GPU OOM 提示）
 *   - 多格式导出（COCO / VOC / YOLO）
 *   - 图片列表缩略图（删除/设为参考图）
 *   - 页面加载时恢复图片列表
 */
import React, { useEffect, useRef, useCallback, useState } from 'react';
import {
  Card, Progress, Button, Tag, Typography, message, Empty,
  Popconfirm, Modal, Select, Space,
} from 'antd';
import {
  DownloadOutlined,
  CheckCircleOutlined,
  DeleteOutlined,
  StarOutlined,
  StarFilled,
  PauseCircleOutlined,
  CaretRightOutlined,
  StopOutlined,
  ExclamationCircleOutlined,
} from '@ant-design/icons';
import { useAppStore } from '../store/useAppStore';
import {
  getTaskStatus, deleteImage, listImages,
  pauseTask, resumeTask, cancelTask,
  exportAnnotations,
} from '../api';

const { Text } = Typography;

const RightPanel: React.FC = () => {
  const {
    images, selectedImageId, viewingImageId,
    taskId, taskStatus, taskProgress, taskTotal, taskMessage, errorType,
    maskUrls, exportUrl,
    selectImage, setViewingImage, removeImage, addImages,
    updateTaskProgress, setTaskStatus, setMaskUrls, setExportUrl,
    setErrorType, setInstanceMasks, setTask,
  } = useAppStore();

  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const initRef = useRef(false);
  const [exportFormat, setExportFormat] = useState<'coco' | 'voc' | 'yolo'>('coco');

  // ==================== 页面加载时恢复图片列表 ====================
  useEffect(() => {
    if (initRef.current) return;
    initRef.current = true;
    listImages()
      .then((serverImages) => {
        if (serverImages.length > 0 && images.length === 0) {
          const items = serverImages.map((img) => ({
            id: img.image_id, filename: img.filename, url: img.url, path: img.path,
          }));
          addImages(items);
          selectImage(items[0].id);
        }
      })
      .catch(() => {});
  }, []);

  // ==================== 任务状态轮询 ====================
  useEffect(() => {
    if (pollRef.current) {
      clearInterval(pollRef.current);
      pollRef.current = null;
    }

    const pollableStatuses = ['pending', 'processing', 'paused'];
    if (!taskId || !pollableStatuses.includes(taskStatus)) return;

    pollRef.current = setInterval(async () => {
      try {
        const res = await getTaskStatus(taskId);
        updateTaskProgress(res.progress, res.total, res.message);

        if (res.error_type) {
          setErrorType(res.error_type);
        }

        if (res.status === 'success') {
          setTaskStatus('success', res.message);
          if (res.mask_urls) setMaskUrls(res.mask_urls);
          if (res.export_url) setExportUrl(res.export_url);
          message.success('标注完成！');
          if (pollRef.current) clearInterval(pollRef.current);
        } else if (res.status === 'failed') {
          setTaskStatus('failed', res.message);
          if (res.error_type) setErrorType(res.error_type);
          message.error(`标注失败：${res.message}`);
          if (pollRef.current) clearInterval(pollRef.current);
        } else if (res.status === 'canceled') {
          setTaskStatus('canceled', res.message);
          message.info('任务已取消');
          if (pollRef.current) clearInterval(pollRef.current);
        } else if (res.status === 'instance_ready') {
          // 模式3 阶段1 完成：实例就绪
          setTaskStatus('instance_ready', res.message);
          if (res.instance_masks) {
            setInstanceMasks(res.instance_masks);
          }
          message.success('实例生成完成，请选择目标实例');
          if (pollRef.current) clearInterval(pollRef.current);
        } else if (res.status === 'paused') {
          setTaskStatus('paused', res.message || '任务已暂停');
        }
      } catch (err: any) {
        console.error('Poll error:', err);
      }
    }, 1000);

    return () => {
      if (pollRef.current) clearInterval(pollRef.current);
    };
  }, [taskId, taskStatus]);

  // ==================== GPU OOM 弹窗 ====================
  useEffect(() => {
    if (errorType === 'gpu_oom') {
      Modal.warning({
        title: 'GPU 显存不足',
        icon: <ExclamationCircleOutlined />,
        content: '显存不足，建议压缩图片分辨率或关闭其他程序后重试。',
        okText: '知道了',
        onOk: () => setErrorType(null),
      });
    }
  }, [errorType]);

  /** 暂停任务 */
  const handlePause = useCallback(async () => {
    if (!taskId) return;
    try {
      await pauseTask(taskId);
      setTaskStatus('paused', '任务已暂停');
      message.info('任务已暂停');
    } catch (err: any) {
      message.error(`暂停失败：${err.message}`);
    }
  }, [taskId, setTaskStatus]);

  /** 恢复任务 */
  const handleResume = useCallback(async () => {
    if (!taskId) return;
    try {
      await resumeTask(taskId);
      setTaskStatus('processing', '任务已恢复');
      message.info('任务已恢复');
    } catch (err: any) {
      message.error(`恢复失败：${err.message}`);
    }
  }, [taskId, setTaskStatus]);

  /** 取消任务 */
  const handleCancel = useCallback(async () => {
    if (!taskId) return;
    try {
      await cancelTask(taskId);
      setTaskStatus('canceled', '任务已取消');
      message.info('任务已取消');
    } catch (err: any) {
      message.error(`取消失败：${err.message}`);
    }
  }, [taskId, setTaskStatus]);

  /** 删除图片 */
  const handleDelete = useCallback(async (imageId: string, e?: React.MouseEvent) => {
    e?.stopPropagation();
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

  /** 多格式导出 */
  const handleExport = useCallback(async () => {
    if (!taskId) return;
    try {
      const data = await exportAnnotations(taskId, exportFormat);
      // 下载为文件
      const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `${taskId}_${exportFormat}.json`;
      a.click();
      URL.revokeObjectURL(url);
      message.success(`${exportFormat.toUpperCase()} 导出成功`);
    } catch (err: any) {
      message.error(`导出失败：${err.message}`);
    }
  }, [taskId, exportFormat]);

  const progressPercent = taskTotal > 0 ? Math.round((taskProgress / taskTotal) * 100) : 0;
  const isActive = taskStatus === 'processing' || taskStatus === 'pending';
  const isPaused = taskStatus === 'paused';

  const statusColor: Record<string, string> = {
    idle: 'default', pending: 'orange', processing: 'blue',
    success: 'green', failed: 'red', paused: 'gold',
    canceled: 'default', instance_ready: 'cyan',
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
        extra={<Tag color={statusColor[taskStatus] || 'default'}>{taskStatus}</Tag>}
      >
        {taskStatus !== 'idle' && (
          <>
            <Progress
              percent={progressPercent}
              size="small"
              status={
                taskStatus === 'failed' ? 'exception'
                : taskStatus === 'success' ? 'success'
                : taskStatus === 'paused' ? 'normal'
                : 'active'
              }
              format={() => `${taskProgress}/${taskTotal}`}
            />
            <Text style={{ color: '#999', fontSize: 12, display: 'block', marginTop: 4 }}>
              {taskMessage}
            </Text>

            {/* 任务控制按钮 */}
            {(isActive || isPaused) && (
              <Space style={{ marginTop: 8 }}>
                {isActive && (
                  <Button size="small" icon={<PauseCircleOutlined />} onClick={handlePause}>
                    暂停
                  </Button>
                )}
                {isPaused && (
                  <Button size="small" type="primary" icon={<CaretRightOutlined />} onClick={handleResume}>
                    恢复
                  </Button>
                )}
                <Popconfirm
                  title="取消任务将终止推理并清理中间文件，确定取消？"
                  onConfirm={handleCancel}
                  okText="确定取消"
                  cancelText="继续"
                >
                  <Button size="small" danger icon={<StopOutlined />}>
                    取消
                  </Button>
                </Popconfirm>
              </Space>
            )}
          </>
        )}

        {/* 多格式导出 */}
        {taskStatus === 'success' && taskId && (
          <div style={{ marginTop: 8 }}>
            <Space>
              <Select
                value={exportFormat}
                onChange={(v) => setExportFormat(v)}
                size="small"
                style={{ width: 90 }}
                options={[
                  { value: 'coco', label: 'COCO' },
                  { value: 'voc', label: 'VOC' },
                  { value: 'yolo', label: 'YOLO' },
                ]}
              />
              <Button
                type="primary"
                icon={<DownloadOutlined />}
                size="small"
                onClick={handleExport}
              >
                导出
              </Button>
            </Space>
          </div>
        )}
      </Card>

      {/* 图片列表 */}
      <div style={{ flex: 1, overflow: 'auto', padding: '8px' }}>
        <Text style={{ color: '#999', fontSize: 12, marginBottom: 8, display: 'block' }}>
          图片列表 ({images.length})
        </Text>

        {images.length === 0 ? (
          <Empty
            description={<Text style={{ color: '#666' }}>暂无图片，请上传</Text>}
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
                }}
              >
                <img
                  src={img.url}
                  alt={img.filename}
                  style={{
                    width: 48, height: 48,
                    objectFit: 'cover', borderRadius: 4, flexShrink: 0,
                  }}
                />
                <div style={{ flex: 1, minWidth: 0 }}>
                  <Text ellipsis style={{ color: '#ddd', fontSize: 12, display: 'block' }}>
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
                <div style={{ display: 'flex', flexDirection: 'column', gap: 2, flexShrink: 0 }}>
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
                  <Popconfirm
                    title="删除后关联 Mask/标注结果将一并清除，确定删除？"
                    onConfirm={() => handleDelete(img.id)}
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
