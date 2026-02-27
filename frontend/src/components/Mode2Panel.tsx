/**
 * 模式2：人工预标注 → 批量自动标注
 * 用户在参考图上框选实例 → DINOv3 构建特征模板 → 跨图匹配 → SAM3 分割
 */
import React, { useState, useRef, useCallback } from 'react';
import {
  Card, Button, Select, Slider, Row, Col,
  Spin, message, Typography, Divider, Tag,
} from 'antd';
import { AimOutlined, PlayCircleOutlined, ClearOutlined } from '@ant-design/icons';
import { mode2Annotate, getImageUrl, Annotation } from '../api';
import AnnotationViewer from './AnnotationViewer';

const { Text } = Typography;

interface Props {
  images: { name: string; path: string; size: number }[];
  onRefresh: () => void;
}

interface Box {
  x1: number;
  y1: number;
  x2: number;
  y2: number;
}

const Mode2Panel: React.FC<Props> = ({ images, onRefresh }) => {
  // 参考图像
  const [refImage, setRefImage] = useState<string>('');
  // 目标图像
  const [targetImages, setTargetImages] = useState<string[]>([]);
  // 用户框选的边界框
  const [userBoxes, setUserBoxes] = useState<Box[]>([]);
  // 正在绘制的框
  const [drawing, setDrawing] = useState(false);
  const [startPos, setStartPos] = useState<{ x: number; y: number } | null>(null);
  const [currentBox, setCurrentBox] = useState<Box | null>(null);
  // 相似度阈值
  const [threshold, setThreshold] = useState(0.8);
  // 加载状态
  const [loading, setLoading] = useState(false);
  // 结果
  const [results, setResults] = useState<any>(null);
  const [viewIndex, setViewIndex] = useState(0);

  const canvasRef = useRef<HTMLDivElement>(null);
  const imgRef = useRef<HTMLImageElement>(null);

  /** 获取鼠标在图像上的坐标 */
  const getImageCoords = useCallback(
    (e: React.MouseEvent) => {
      if (!imgRef.current) return { x: 0, y: 0 };
      const rect = imgRef.current.getBoundingClientRect();
      const scaleX = imgRef.current.naturalWidth / rect.width;
      const scaleY = imgRef.current.naturalHeight / rect.height;
      return {
        x: (e.clientX - rect.left) * scaleX,
        y: (e.clientY - rect.top) * scaleY,
      };
    },
    []
  );

  /** 鼠标按下：开始绘制框 */
  const handleMouseDown = useCallback(
    (e: React.MouseEvent) => {
      if (!refImage) return;
      const pos = getImageCoords(e);
      setStartPos(pos);
      setDrawing(true);
    },
    [refImage, getImageCoords]
  );

  /** 鼠标移动：更新当前框 */
  const handleMouseMove = useCallback(
    (e: React.MouseEvent) => {
      if (!drawing || !startPos) return;
      const pos = getImageCoords(e);
      setCurrentBox({
        x1: Math.min(startPos.x, pos.x),
        y1: Math.min(startPos.y, pos.y),
        x2: Math.max(startPos.x, pos.x),
        y2: Math.max(startPos.y, pos.y),
      });
    },
    [drawing, startPos, getImageCoords]
  );

  /** 鼠标松开：完成框选 */
  const handleMouseUp = useCallback(() => {
    if (currentBox && currentBox.x2 - currentBox.x1 > 5 && currentBox.y2 - currentBox.y1 > 5) {
      setUserBoxes((prev) => [...prev, currentBox]);
    }
    setDrawing(false);
    setStartPos(null);
    setCurrentBox(null);
  }, [currentBox]);

  /** 执行批量标注 */
  const handleAnnotate = async () => {
    if (!refImage) {
      message.warning('请选择参考图像');
      return;
    }
    if (userBoxes.length === 0) {
      message.warning('请在参考图上框选至少一个目标实例');
      return;
    }
    if (targetImages.length === 0) {
      message.warning('请选择待标注的目标图像');
      return;
    }

    setLoading(true);
    try {
      const data = await mode2Annotate({
        ref_image_name: refImage,
        user_boxes: userBoxes.map((b) => [b.x1, b.y1, b.x2, b.y2]),
        target_image_names: targetImages,
        similarity_threshold: threshold,
      });
      setResults(data);
      setViewIndex(0);
      message.success(`批量标注完成，共 ${data.total_annotations} 个实例`);
    } catch (err: any) {
      message.error('标注失败: ' + (err.response?.data?.detail || err.message));
    } finally {
      setLoading(false);
    }
  };

  /** 将框坐标转为 SVG 显示坐标 */
  const boxToSvgRect = (box: Box) => {
    if (!imgRef.current) return null;
    const rect = imgRef.current.getBoundingClientRect();
    const scaleX = rect.width / imgRef.current.naturalWidth;
    const scaleY = rect.height / imgRef.current.naturalHeight;
    return {
      x: box.x1 * scaleX,
      y: box.y1 * scaleY,
      width: (box.x2 - box.x1) * scaleX,
      height: (box.y2 - box.y1) * scaleY,
    };
  };

  return (
    <div>
      <Row gutter={16}>
        {/* 左侧：参考图 + 框选 */}
        <Col span={12}>
          <Card
            title="📌 参考图像（框选目标实例）"
            size="small"
            extra={
              <Button
                icon={<ClearOutlined />}
                size="small"
                onClick={() => setUserBoxes([])}
                disabled={userBoxes.length === 0}
              >
                清除框选
              </Button>
            }
          >
            {/* 参考图选择 */}
            <Select
              style={{ width: '100%', marginBottom: 8 }}
              placeholder="选择参考图像"
              value={refImage || undefined}
              onChange={(v) => {
                setRefImage(v);
                setUserBoxes([]);
              }}
              options={images.map((img) => ({
                label: img.name,
                value: img.name,
              }))}
            />

            {/* 参考图 + 框选交互 */}
            {refImage && (
              <div
                ref={canvasRef}
                className="annotation-canvas"
                onMouseDown={handleMouseDown}
                onMouseMove={handleMouseMove}
                onMouseUp={handleMouseUp}
                style={{ position: 'relative', userSelect: 'none' }}
              >
                <img
                  ref={imgRef}
                  src={getImageUrl(refImage)}
                  alt="参考图"
                  style={{ maxWidth: '100%', display: 'block' }}
                  draggable={false}
                />
                {/* SVG 覆盖层：显示已框选的框 */}
                <svg
                  style={{
                    position: 'absolute',
                    top: 0,
                    left: 0,
                    width: '100%',
                    height: '100%',
                    pointerEvents: 'none',
                  }}
                >
                  {userBoxes.map((box, i) => {
                    const r = boxToSvgRect(box);
                    return r ? (
                      <rect
                        key={i}
                        x={r.x}
                        y={r.y}
                        width={r.width}
                        height={r.height}
                        fill="rgba(24, 144, 255, 0.2)"
                        stroke="#1890ff"
                        strokeWidth={2}
                      />
                    ) : null;
                  })}
                  {currentBox && (() => {
                    const r = boxToSvgRect(currentBox);
                    return r ? (
                      <rect
                        x={r.x}
                        y={r.y}
                        width={r.width}
                        height={r.height}
                        fill="rgba(255, 77, 79, 0.2)"
                        stroke="#ff4d4f"
                        strokeWidth={2}
                        strokeDasharray="4"
                      />
                    ) : null;
                  })()}
                </svg>
              </div>
            )}

            <div style={{ marginTop: 8 }}>
              {userBoxes.map((_, i) => (
                <Tag color="blue" key={i}>框选 #{i + 1}</Tag>
              ))}
              {userBoxes.length === 0 && refImage && (
                <Text type="secondary">在图像上拖拽鼠标框选目标实例（1-3个）</Text>
              )}
            </div>
          </Card>

          {/* 参数配置 */}
          <Card title="⚙️ 参数配置" size="small" style={{ marginTop: 12 }}>
            <div style={{ marginBottom: 12 }}>
              <Text strong>目标图像</Text>
              <Select
                mode="multiple"
                style={{ width: '100%', marginTop: 4 }}
                placeholder="选择待标注的目标图像"
                value={targetImages}
                onChange={setTargetImages}
                options={images
                  .filter((img) => img.name !== refImage)
                  .map((img) => ({ label: img.name, value: img.name }))}
                maxTagCount={3}
              />
            </div>
            <div style={{ marginBottom: 12 }}>
              <Text strong>相似度阈值: {threshold.toFixed(2)}</Text>
              <Slider
                min={0.5}
                max={0.99}
                step={0.01}
                value={threshold}
                onChange={setThreshold}
              />
            </div>
            <Button
              type="primary"
              icon={<PlayCircleOutlined />}
              onClick={handleAnnotate}
              loading={loading}
              block
              size="large"
            >
              批量自动标注
            </Button>
          </Card>
        </Col>

        {/* 右侧：结果展示 */}
        <Col span={12}>
          <Card
            title="🖼️ 标注结果"
            size="small"
            extra={
              results?.target_results?.length > 1 && (
                <Select
                  value={viewIndex}
                  onChange={setViewIndex}
                  style={{ width: 200 }}
                  options={results.target_results.map((r: any, i: number) => ({
                    label: `${r.image} (${r.count}个)`,
                    value: i,
                  }))}
                />
              )
            }
          >
            <Spin spinning={loading} tip="正在批量标注...">
              {results?.target_results?.[viewIndex] ? (
                <AnnotationViewer
                  imageUrl={getImageUrl(results.target_results[viewIndex].image)}
                  annotations={results.target_results[viewIndex].annotations}
                />
              ) : (
                <div
                  style={{
                    height: 400,
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    color: '#999',
                  }}
                >
                  {loading ? '' : '在参考图上框选目标，选择目标图像后点击"批量自动标注"'}
                </div>
              )}
            </Spin>
          </Card>
        </Col>
      </Row>
    </div>
  );
};

export default Mode2Panel;
