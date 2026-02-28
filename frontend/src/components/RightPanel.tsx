/**
 * 右侧面板
 * 包含：图片列表缩略图、批量进度条、一键导出按钮。
 * 点击缩略图切换查看对应图片的 Mask 结果。
 */
import React, { useEffect, useRef, useCallback } from 'react';
import { Card, Progress, Button, List, Tag, Typography, message, Empty } from 'antd';
import { DownloadOutlined, CheckCircleOutlined, LoadingOutlined } from '@ant-design/icons';
import { useAppStore } from '../store/useAppStore';
import { getTaskStatus } from '../api';

const { Text } = Typography;

const RightPanel: React.FC = () => {
  const {
    images, selectedImageId, viewingImageId,
    taskId, taskStatus, taskProgress, taskTotal, taskMessage,
    maskUrls, exportUrl,
    selectImage, setViewingImage,
    updateTaskProgress, setTaskStatus, setMaskUrls, setExportUrl,
  } = useAppStore();

  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null);

  // ==================== 任务状态轮询 ====================
  useEffect(() => {
    // 清除旧的轮询
    if (pollRef.current) {
      clearInterval(pollRef.current);
      pollRef.current = null;
    }

    // 只在 pending/processing 状态下轮询
    if (!taskId || (taskStatus !== 'pending' && taskStatus !== 'processing')) {
      return;
    }

    // 每 1 秒轮询一次
    pollRef.current = setInterval(async () => {
      try {
        const res = await getTaskStatus(taskId);

        updateTaskProgress(res.progress, res.total, res.message);

        if (res.status === 'success') {
          setTaskStatus('success', res.message);
          if (res.mask_urls) setMaskUrls(res.mask_urls);
          if (res.export_url) setExportUrl(res.export_url);
          message.success('批量标注完成！');
          // 停止轮询
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

  /** 下载 COCO 导出 */
  const handleExport = useCallback(() => {
    if (exportUrl) {
      window.open(exportUrl, '_blank');
    }
  }, [exportUrl]);

  /** 进度百分比 */
  const progressPercent = taskTotal > 0 ? Math.round((taskProgress / taskTotal) * 100) : 0;

  /** 状态标签颜色 */
  const statusColor: Record<string, string> = {
    idle: 'default',
    pending: 'orange',
    processing: 'blue',
    success: 'green',
    failed: 'red',
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

        {/* 导出按钮 */}
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
          <List
            dataSource={images}
            renderItem={(img) => {
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
                  {/* 缩略图 */}
                  <img
                    src={img.url}
                    alt={img.filename}
                    style={{
                      width: 48,
                      height: 48,
                      objectFit: 'cover',
                      borderRadius: 4,
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
                    <div style={{ display: 'flex', gap: 4, marginTop: 2 }}>
                      {isRef && <Tag color="blue" style={{ fontSize: 10, lineHeight: '16px', padding: '0 4px' }}>参考图</Tag>}
                      {hasMask && (
                        <Tag color="green" style={{ fontSize: 10, lineHeight: '16px', padding: '0 4px' }}>
                          <CheckCircleOutlined /> {maskUrls[img.id].length}
                        </Tag>
                      )}
                    </div>
                  </div>
                </div>
              );
            }}
          />
        )}
      </div>
    </div>
  );
};

export default RightPanel;
