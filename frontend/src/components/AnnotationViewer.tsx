/**
 * 标注结果可视化组件
 * 在图像上叠加显示边界框和分割多边形
 */
import React, { useRef, useEffect, useState } from 'react';
import { Tag, Typography, Table } from 'antd';
import type { Annotation } from '../api';

const { Text } = Typography;

/** 标注颜色表 */
const COLORS = [
  '#1890ff', '#52c41a', '#faad14', '#f5222d', '#722ed1',
  '#13c2c2', '#eb2f96', '#fa8c16', '#a0d911', '#2f54eb',
  '#ff7a45', '#36cfc9', '#9254de', '#ffc53d', '#73d13d',
];

interface Props {
  imageUrl: string;
  annotations: Annotation[];
}

const AnnotationViewer: React.FC<Props> = ({ imageUrl, annotations }) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [imgSize, setImgSize] = useState({ width: 0, height: 0 });
  const [displaySize, setDisplaySize] = useState({ width: 0, height: 0 });

  /** 绘制标注 */
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const img = new Image();
    img.crossOrigin = 'anonymous';
    img.onload = () => {
      setImgSize({ width: img.naturalWidth, height: img.naturalHeight });

      // 计算显示尺寸（最大宽度 800px）
      const maxWidth = 800;
      const scale = Math.min(1, maxWidth / img.naturalWidth);
      const dw = img.naturalWidth * scale;
      const dh = img.naturalHeight * scale;
      setDisplaySize({ width: dw, height: dh });

      canvas.width = dw;
      canvas.height = dh;
      const ctx = canvas.getContext('2d');
      if (!ctx) return;

      // 绘制图像
      ctx.drawImage(img, 0, 0, dw, dh);

      // 绘制标注
      annotations.forEach((ann, idx) => {
        const color = COLORS[idx % COLORS.length];
        ctx.strokeStyle = color;
        ctx.lineWidth = 2;
        ctx.font = '14px sans-serif';

        // 绘制分割多边形
        if (ann.segmentation && ann.segmentation.length > 0) {
          ctx.fillStyle = color + '33'; // 半透明填充
          ann.segmentation.forEach((polygon) => {
            if (polygon.length < 6) return;
            ctx.beginPath();
            ctx.moveTo(polygon[0] * scale, polygon[1] * scale);
            for (let i = 2; i < polygon.length; i += 2) {
              ctx.lineTo(polygon[i] * scale, polygon[i + 1] * scale);
            }
            ctx.closePath();
            ctx.fill();
            ctx.stroke();
          });
        }

        // 绘制边界框
        if (ann.bbox && ann.bbox.length >= 4) {
          const [x, y, w, h] = ann.bbox;
          ctx.strokeRect(x * scale, y * scale, w * scale, h * scale);

          // 绘制标签
          const label = `${ann.label || ''} ${(ann.score ?? 0).toFixed(2)}`;
          const textWidth = ctx.measureText(label).width;
          ctx.fillStyle = color;
          ctx.fillRect(x * scale, y * scale - 18, textWidth + 8, 18);
          ctx.fillStyle = '#fff';
          ctx.fillText(label, x * scale + 4, y * scale - 4);
        }
      });
    };
    img.src = imageUrl;
  }, [imageUrl, annotations]);

  /** 标注列表表格列 */
  const columns = [
    {
      title: '#',
      dataIndex: 'index',
      key: 'index',
      width: 40,
      render: (_: any, __: any, idx: number) => idx + 1,
    },
    {
      title: '类别',
      dataIndex: 'label',
      key: 'label',
      render: (label: string, _: any, idx: number) => (
        <Tag color={COLORS[idx % COLORS.length]}>{label || '未知'}</Tag>
      ),
    },
    {
      title: '置信度',
      dataIndex: 'score',
      key: 'score',
      width: 80,
      render: (score: number) => (score ?? 0).toFixed(3),
    },
    {
      title: '面积',
      dataIndex: 'area',
      key: 'area',
      width: 80,
      render: (area: number) => (area ?? 0).toFixed(0),
    },
  ];

  return (
    <div>
      {/* 画布 */}
      <div style={{ textAlign: 'center', marginBottom: 12 }}>
        <canvas
          ref={canvasRef}
          style={{
            maxWidth: '100%',
            border: '1px solid #d9d9d9',
            borderRadius: 4,
          }}
        />
      </div>

      {/* 标注统计 */}
      <Text type="secondary" style={{ marginBottom: 8, display: 'block' }}>
        共 {annotations.length} 个标注实例
        {imgSize.width > 0 && ` | 图像尺寸: ${imgSize.width}×${imgSize.height}`}
      </Text>

      {/* 标注列表 */}
      {annotations.length > 0 && (
        <Table
          dataSource={annotations}
          columns={columns}
          size="small"
          pagination={false}
          rowKey={(_, idx) => String(idx)}
          scroll={{ y: 200 }}
        />
      )}
    </div>
  );
};

export default AnnotationViewer;
