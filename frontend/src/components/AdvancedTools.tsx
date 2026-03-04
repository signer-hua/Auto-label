/**
 * 高级工具面板：LoRA 微调入口
 * 选择类别 + 已标注图片 → 启动 SAM3 LoRA 微调
 */
import React, { useCallback, useState } from 'react';
import { Button, Select, InputNumber, message, Tooltip } from 'antd';
import { ThunderboltOutlined } from '@ant-design/icons';
import { useAppStore } from '../store/useAppStore';
import { startLoraFinetune } from '../api';

const AdvancedTools: React.FC = () => {
  const { categories, activeCategoryId, images, maskUrls, setActiveCategoryId, setTask } = useAppStore();
  const [epochs, setEpochs] = useState(10);
  const [loading, setLoading] = useState(false);

  const annotatedImages = images.filter(img => (maskUrls[img.id] || []).length > 0);

  const handleFinetune = useCallback(async () => {
    if (!activeCategoryId) { message.warning('请先选择类别'); return; }
    if (annotatedImages.length < 5) {
      message.warning(`至少需要 5 张已标注图片（当前 ${annotatedImages.length} 张）`);
      return;
    }
    setLoading(true);
    try {
      const result = await startLoraFinetune({
        image_ids: annotatedImages.map(img => img.id),
        category_id: activeCategoryId,
        epochs,
      });
      setTask(result.task_id, 'pending');
      message.info(`LoRA 微调已提交（${annotatedImages.length} 张，${epochs} 轮）`);
    } catch (err: any) {
      message.error(`微调启动失败：${err?.response?.data?.detail || err.message}`);
    } finally {
      setLoading(false);
    }
  }, [activeCategoryId, annotatedImages, epochs, setTask]);

  return (
    <div style={{ borderTop: '1px solid #444', padding: '8px 0' }}>
      <div style={{ color: '#999', fontSize: 12, marginBottom: 6 }}>高级工具</div>
      <div style={{ color: '#888', fontSize: 11, marginBottom: 4 }}>
        SAM3 LoRA 微调（提升特定类别精度）
      </div>
      <Select
        value={activeCategoryId || undefined}
        onChange={(v) => setActiveCategoryId(v)}
        placeholder="选择微调类别"
        size="small"
        style={{ width: '100%', marginBottom: 4 }}
        options={categories.map(c => ({
          value: c.id,
          label: <span style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
            <span style={{ width: 10, height: 10, borderRadius: 2, background: c.color, display: 'inline-block' }} />{c.name}</span>,
        }))}
      />
      <div style={{ display: 'flex', gap: 4, alignItems: 'center', marginBottom: 4 }}>
        <span style={{ color: '#888', fontSize: 11 }}>轮次:</span>
        <InputNumber size="small" min={1} max={50} value={epochs}
          onChange={(v) => setEpochs(v || 10)} style={{ width: 60 }} />
        <span style={{ color: '#666', fontSize: 10 }}>已标注 {annotatedImages.length} 张</span>
      </div>
      <Tooltip title={annotatedImages.length < 5 ? '至少需要 5 张已标注图片' : ''}>
        <Button size="small" icon={<ThunderboltOutlined />}
          onClick={handleFinetune} loading={loading}
          disabled={loading || !activeCategoryId || annotatedImages.length < 5}
          block style={{ background: '#722ed1', borderColor: '#722ed1', color: '#fff' }}>
          启动 LoRA 微调
        </Button>
      </Tooltip>
    </div>
  );
};

export default AdvancedTools;
