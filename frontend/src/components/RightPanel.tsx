/**
 * 右侧面板（v3 增强版）
 *
 * v3 新增：
 *   - 置信分数列（0~100 分颜色区分：绿/黄/红）
 *   - 分数筛选（仅看高分/中分/低分）+ 排序
 *   - 图片详情分项评分悬浮提示
 *   - 参考图库管理（星标多选参考图 + 权重设置）
 */
import React, { useEffect, useRef, useCallback, useState } from 'react';
import {
  Card, Progress, Button, Tag, Typography, message, Empty,
  Popconfirm, Modal, Select, Space, Tooltip, Slider,
} from 'antd';
import {
  DownloadOutlined, CheckCircleOutlined, DeleteOutlined,
  StarOutlined, StarFilled, PauseCircleOutlined,
  CaretRightOutlined, StopOutlined, ExclamationCircleOutlined,
  PushpinOutlined, PushpinFilled,
} from '@ant-design/icons';
import { useAppStore, ScoreFilter, SortOrder, ImageScore } from '../store/useAppStore';
import {
  getTaskStatus, deleteImage, listImages,
  pauseTask, resumeTask, cancelTask, exportAnnotations,
} from '../api';

const { Text } = Typography;

const scoreColor = (score: number) => {
  if (score >= 85) return '#52c41a';
  if (score >= 70) return '#faad14';
  return '#ff4d4f';
};

const RightPanel: React.FC = () => {
  const {
    images, selectedImageId, viewingImageId,
    taskId, taskStatus, taskProgress, taskTotal, taskMessage, errorType,
    maskUrls, imageScores, scoreFilter, sortOrder,
    refImages,
    selectImage, setViewingImage, removeImage, addImages,
    updateTaskProgress, setTaskStatus, setMaskUrls, mergeMaskUrls, setExportUrl,
    setErrorType, setInstanceMasks, setImageScores,
    setScoreFilter, setSortOrder,
    addRefImage, removeRefImage, setRefImageWeight,
  } = useAppStore();

  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const initRef = useRef(false);
  const [exportFormat, setExportFormat] = useState<'coco' | 'voc' | 'yolo'>('coco');

  useEffect(() => {
    if (initRef.current) return;
    initRef.current = true;
    listImages().then((serverImages) => {
      if (serverImages.length > 0 && images.length === 0) {
        const items = serverImages.map((img) => ({
          id: img.image_id, filename: img.filename, url: img.url, path: img.path,
        }));
        addImages(items);
        selectImage(items[0].id);
      }
    }).catch(() => {});
  }, []);

  useEffect(() => {
    if (pollRef.current) { clearInterval(pollRef.current); pollRef.current = null; }
    const pollable = ['pending', 'processing', 'paused'];
    if (!taskId || !pollable.includes(taskStatus)) return;
    pollRef.current = setInterval(async () => {
      try {
        const res = await getTaskStatus(taskId);
        updateTaskProgress(res.progress, res.total, res.message);
        if (res.error_type) setErrorType(res.error_type);

        if (res.status === 'success') {
          setTaskStatus('success', res.message);
          if (res.mask_urls) mergeMaskUrls(res.mask_urls);
          if (res.export_url) setExportUrl(res.export_url);
          if (res.image_scores) setImageScores(res.image_scores);
          message.success('标注完成！');
          if (pollRef.current) clearInterval(pollRef.current);
        } else if (res.status === 'failed') {
          setTaskStatus('failed', res.message);
          message.error(`标注失败：${res.message}`);
          if (pollRef.current) clearInterval(pollRef.current);
        } else if (res.status === 'canceled') {
          setTaskStatus('canceled', res.message);
          if (pollRef.current) clearInterval(pollRef.current);
        } else if (res.status === 'instance_ready') {
          setTaskStatus('instance_ready', res.message);
          if (res.instance_masks) setInstanceMasks(res.instance_masks, selectedImageId || undefined);
          message.success('实例生成完成');
          if (pollRef.current) clearInterval(pollRef.current);
        }
      } catch (err) { console.error('Poll error:', err); }
    }, 1000);
    return () => { if (pollRef.current) clearInterval(pollRef.current); };
  }, [taskId, taskStatus]);

  useEffect(() => {
    if (errorType === 'gpu_oom') {
      Modal.warning({
        title: 'GPU 显存不足', icon: <ExclamationCircleOutlined />,
        content: '显存不足，建议压缩图片分辨率或关闭其他程序后重试。',
        okText: '知道了', onOk: () => setErrorType(null),
      });
    }
  }, [errorType]);

  const handlePause = useCallback(async () => { if (taskId) { await pauseTask(taskId); setTaskStatus('paused'); } }, [taskId]);
  const handleResume = useCallback(async () => { if (taskId) { await resumeTask(taskId); setTaskStatus('processing'); } }, [taskId]);
  const handleCancel = useCallback(async () => { if (taskId) { await cancelTask(taskId); setTaskStatus('canceled'); } }, [taskId]);
  const handleDelete = useCallback(async (imageId: string, e?: React.MouseEvent) => {
    e?.stopPropagation();
    await deleteImage(imageId); removeImage(imageId); message.success('已删除');
  }, [removeImage]);
  const handleSetRef = useCallback((imageId: string, e: React.MouseEvent) => {
    e.stopPropagation(); selectImage(imageId); message.info('已设为参考图');
  }, [selectImage]);
  const handleToggleRefLib = useCallback((imageId: string, e: React.MouseEvent) => {
    e.stopPropagation();
    const isRef = refImages.some(r => r.imageId === imageId);
    if (isRef) { removeRefImage(imageId); } else { addRefImage(imageId); message.info('已加入参考图库'); }
  }, [refImages, addRefImage, removeRefImage]);
  const handleExport = useCallback(async () => {
    if (!taskId) return;
    const data = await exportAnnotations(taskId, exportFormat);
    const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url; a.download = `${taskId}_${exportFormat}.json`; a.click();
    URL.revokeObjectURL(url); message.success(`${exportFormat.toUpperCase()} 导出成功`);
  }, [taskId, exportFormat]);

  const progressPercent = taskTotal > 0 ? Math.round((taskProgress / taskTotal) * 100) : 0;
  const isActive = taskStatus === 'processing' || taskStatus === 'pending';
  const isPaused = taskStatus === 'paused';

  // 筛选 + 排序
  let filteredImages = [...images];
  if (scoreFilter !== 'all' && Object.keys(imageScores).length > 0) {
    filteredImages = filteredImages.filter((img) => {
      const sc = imageScores[img.id];
      if (!sc) return scoreFilter === 'low';
      return sc.level === scoreFilter;
    });
  }
  if (sortOrder === 'score_asc') {
    filteredImages.sort((a, b) => (imageScores[a.id]?.total ?? 0) - (imageScores[b.id]?.total ?? 0));
  } else if (sortOrder === 'score_desc') {
    filteredImages.sort((a, b) => (imageScores[b.id]?.total ?? 0) - (imageScores[a.id]?.total ?? 0));
  }

  const statusColor: Record<string, string> = {
    idle: 'default', pending: 'orange', processing: 'blue',
    success: 'green', failed: 'red', paused: 'gold',
    canceled: 'default', instance_ready: 'cyan',
  };

  const renderScoreTooltip = (sc: ImageScore) => (
    <div style={{ fontSize: 11, lineHeight: '18px' }}>
      <div>匹配度: <b>{sc.similarity}</b></div>
      <div>Mask完整: <b>{sc.mask_coverage}</b></div>
      <div>面积合理: <b>{sc.area}</b></div>
      <div>检测完整: <b>{sc.detection}</b></div>
    </div>
  );

  return (
    <div style={{ width: 280, background: '#1a1a1a', display: 'flex', flexDirection: 'column', borderLeft: '1px solid #333', overflow: 'hidden' }}>
      <Card size="small" title="任务状态" style={{ background: '#222', borderBottom: '1px solid #333' }}
        headStyle={{ color: '#ddd', borderBottom: '1px solid #333' }} bodyStyle={{ padding: '12px' }}
        extra={<Tag color={statusColor[taskStatus] || 'default'}>{taskStatus}</Tag>}>
        {taskStatus !== 'idle' && (
          <>
            <Progress percent={progressPercent} size="small"
              status={taskStatus === 'failed' ? 'exception' : taskStatus === 'success' ? 'success' : 'active'}
              format={() => `${taskProgress}/${taskTotal}`} />
            <Text style={{ color: '#999', fontSize: 12, display: 'block', marginTop: 4 }}>{taskMessage}</Text>
            {(isActive || isPaused) && (
              <Space style={{ marginTop: 8 }}>
                {isActive && <Button size="small" icon={<PauseCircleOutlined />} onClick={handlePause}>暂停</Button>}
                {isPaused && <Button size="small" type="primary" icon={<CaretRightOutlined />} onClick={handleResume}>恢复</Button>}
                <Popconfirm title="确定取消？" onConfirm={handleCancel} okText="确定" cancelText="继续">
                  <Button size="small" danger icon={<StopOutlined />}>取消</Button>
                </Popconfirm>
              </Space>
            )}
          </>
        )}
        {taskStatus === 'success' && taskId && (
          <div style={{ marginTop: 8 }}>
            <Space>
              <Select value={exportFormat} onChange={(v) => setExportFormat(v)} size="small" style={{ width: 90 }}
                options={[{ value: 'coco', label: 'COCO' }, { value: 'voc', label: 'VOC' }, { value: 'yolo', label: 'YOLO' }]} />
              <Button type="primary" icon={<DownloadOutlined />} size="small" onClick={handleExport}>导出</Button>
            </Space>
          </div>
        )}
      </Card>

      {/* 评分筛选/排序 */}
      {Object.keys(imageScores).length > 0 && (
        <div style={{ padding: '6px 8px', borderBottom: '1px solid #333', display: 'flex', gap: 4, flexWrap: 'wrap' }}>
          <Select value={scoreFilter} onChange={(v) => setScoreFilter(v as ScoreFilter)} size="small" style={{ width: 80 }}
            options={[
              { value: 'all', label: '全部' },
              { value: 'high', label: '高分' },
              { value: 'medium', label: '中分' },
              { value: 'low', label: '低分' },
            ]} />
          <Select value={sortOrder} onChange={(v) => setSortOrder(v as SortOrder)} size="small" style={{ flex: 1 }}
            options={[
              { value: 'default', label: '默认' },
              { value: 'score_desc', label: '分数↓' },
              { value: 'score_asc', label: '分数↑' },
            ]} />
        </div>
      )}

      <div style={{ flex: 1, overflow: 'auto', padding: '8px' }}>
        <Text style={{ color: '#999', fontSize: 12, marginBottom: 8, display: 'block' }}>
          图片列表 ({filteredImages.length}/{images.length})
        </Text>
        {filteredImages.length === 0 ? (
          <Empty description={<Text style={{ color: '#666' }}>暂无图片</Text>} image={Empty.PRESENTED_IMAGE_SIMPLE} />
        ) : filteredImages.map((img) => {
          const isRef = img.id === selectedImageId;
          const isViewing = img.id === viewingImageId;
          const hasMask = (maskUrls[img.id] || []).length > 0;
          const sc = imageScores[img.id];
          const isInRefLib = refImages.some(r => r.imageId === img.id);

          return (
            <div key={img.id} onClick={() => setViewingImage(img.id)}
              style={{
                display: 'flex', alignItems: 'center', gap: 6,
                padding: '5px 6px', marginBottom: 3, borderRadius: 6, cursor: 'pointer',
                background: isViewing ? '#333' : 'transparent',
                border: isRef ? '1px solid #1890ff' : '1px solid transparent',
              }}>
              <img src={img.url} alt={img.filename}
                style={{ width: 42, height: 42, objectFit: 'cover', borderRadius: 4, flexShrink: 0 }} />
              <div style={{ flex: 1, minWidth: 0 }}>
                <Text ellipsis style={{ color: '#ddd', fontSize: 11, display: 'block' }}>{img.filename}</Text>
                <div style={{ display: 'flex', gap: 3, marginTop: 2, flexWrap: 'wrap', alignItems: 'center' }}>
                  {isRef && <Tag color="blue" style={{ fontSize: 9, lineHeight: '14px', padding: '0 3px' }}>参考</Tag>}
                  {isInRefLib && <Tag color="purple" style={{ fontSize: 9, lineHeight: '14px', padding: '0 3px' }}>图库</Tag>}
                  {hasMask && <Tag color="green" style={{ fontSize: 9, lineHeight: '14px', padding: '0 3px' }}><CheckCircleOutlined /> {maskUrls[img.id].length}</Tag>}
                  {sc && (
                    <Tooltip title={renderScoreTooltip(sc)} placement="left">
                      <Tag color={scoreColor(sc.total)} style={{ fontSize: 9, lineHeight: '14px', padding: '0 4px', fontWeight: 'bold', cursor: 'help' }}>
                        {Math.round(sc.total)}分
                      </Tag>
                    </Tooltip>
                  )}
                </div>
              </div>
              <div style={{ display: 'flex', flexDirection: 'column', gap: 1, flexShrink: 0 }}>
                <Button type="text" size="small"
                  icon={isRef ? <StarFilled style={{ color: '#faad14', fontSize: 13 }} /> : <StarOutlined style={{ color: '#666', fontSize: 13 }} />}
                  onClick={(e) => handleSetRef(img.id, e)} title="设为参考图"
                  style={{ width: 22, height: 22, padding: 0, minWidth: 22 }} />
                <Button type="text" size="small"
                  icon={isInRefLib ? <PushpinFilled style={{ color: '#b37feb', fontSize: 13 }} /> : <PushpinOutlined style={{ color: '#666', fontSize: 13 }} />}
                  onClick={(e) => handleToggleRefLib(img.id, e)} title="加入/移出参考图库"
                  style={{ width: 22, height: 22, padding: 0, minWidth: 22 }} />
                <Popconfirm title="确定删除？" onConfirm={() => handleDelete(img.id)} okText="删除" cancelText="取消" placement="left">
                  <Button type="text" size="small" danger icon={<DeleteOutlined style={{ fontSize: 13 }} />}
                    onClick={(e) => e.stopPropagation()} style={{ width: 22, height: 22, padding: 0, minWidth: 22 }} />
                </Popconfirm>
              </div>
            </div>
          );
        })}
      </div>

      {/* 参考图库权重设置 */}
      {refImages.length > 0 && (
        <div style={{ borderTop: '1px solid #333', padding: '8px', maxHeight: 120, overflow: 'auto' }}>
          <Text style={{ color: '#999', fontSize: 11, display: 'block', marginBottom: 4 }}>
            参考图库 ({refImages.length}/5)
          </Text>
          {refImages.map((ri) => {
            const img = images.find(i => i.id === ri.imageId);
            return (
              <div key={ri.imageId} style={{ display: 'flex', alignItems: 'center', gap: 4, marginBottom: 2 }}>
                <Text ellipsis style={{ color: '#aaa', fontSize: 10, width: 60 }}>{img?.filename || ri.imageId}</Text>
                <Slider min={0.1} max={2.0} step={0.1} value={ri.weight}
                  onChange={(v) => setRefImageWeight(ri.imageId, v)}
                  style={{ flex: 1 }} tooltip={{ formatter: (v) => `${v}` }} />
                <Text style={{ color: '#888', fontSize: 10, width: 24, textAlign: 'right' }}>{ri.weight}</Text>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
};

export default RightPanel;
