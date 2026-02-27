/**
 * 模式1：文本提示一键自动标注
 * 用户输入文本提示 → YOLO-World 检测 → DINOv3 增强 → SAM3 分割
 */
import React, { useState } from 'react';
import {
  Card, Input, Button, Select, Slider, Row, Col,
  Spin, message, Tag, List, Image as AntImage, Typography,
} from 'antd';
import { PlayCircleOutlined } from '@ant-design/icons';
import { mode1Annotate, getImageUrl, Annotation } from '../api';
import AnnotationViewer from './AnnotationViewer';

const { Text } = Typography;

interface Props {
  images: { name: string; path: string; size: number }[];
  onRefresh: () => void;
}

/** 单张图像的标注结果 */
interface ImageResult {
  image: string;
  annotations: Annotation[];
  count: number;
}

const Mode1Panel: React.FC<Props> = ({ images, onRefresh }) => {
  // 文本提示输入
  const [textInput, setTextInput] = useState('');
  // 选中的图像
  const [selectedImages, setSelectedImages] = useState<string[]>([]);
  // 检测阈值
  const [scoreThr, setScoreThr] = useState(0.3);
  // 导出格式
  const [exportFormat, setExportFormat] = useState('coco');
  // 加载状态
  const [loading, setLoading] = useState(false);
  // 标注结果
  const [results, setResults] = useState<ImageResult[]>([]);
  // 当前查看的图像索引
  const [viewIndex, setViewIndex] = useState(0);

  /** 解析文本提示（逗号/空格分隔） */
  const parsePrompts = (text: string): string[] => {
    return text
      .split(/[,，\s]+/)
      .map((s) => s.trim())
      .filter((s) => s.length > 0);
  };

  /** 执行标注 */
  const handleAnnotate = async () => {
    const prompts = parsePrompts(textInput);
    if (prompts.length === 0) {
      message.warning('请输入文本提示（如：person, car, dog）');
      return;
    }
    if (selectedImages.length === 0) {
      message.warning('请选择至少一张图像');
      return;
    }

    setLoading(true);
    try {
      const data = await mode1Annotate({
        image_names: selectedImages,
        text_prompts: prompts,
        score_thr: scoreThr,
        export_format: exportFormat,
      });
      setResults(data.results || []);
      setViewIndex(0);
      message.success(`标注完成，共 ${data.total_annotations} 个实例`);
    } catch (err: any) {
      message.error('标注失败: ' + (err.response?.data?.detail || err.message));
    } finally {
      setLoading(false);
    }
  };

  return (
    <div>
      <Row gutter={16}>
        {/* 左侧：参数配置 */}
        <Col span={8}>
          <Card title="📝 文本提示配置" size="small">
            {/* 文本输入 */}
            <div style={{ marginBottom: 12 }}>
              <Text strong>文本提示（逗号分隔）</Text>
              <Input.TextArea
                rows={3}
                placeholder="输入目标类别，如：person, car, dog"
                value={textInput}
                onChange={(e) => setTextInput(e.target.value)}
                style={{ marginTop: 4 }}
              />
              <div style={{ marginTop: 4 }}>
                {parsePrompts(textInput).map((p) => (
                  <Tag color="blue" key={p}>{p}</Tag>
                ))}
              </div>
            </div>

            {/* 图像选择 */}
            <div style={{ marginBottom: 12 }}>
              <Text strong>选择图像</Text>
              <Select
                mode="multiple"
                style={{ width: '100%', marginTop: 4 }}
                placeholder="选择要标注的图像"
                value={selectedImages}
                onChange={setSelectedImages}
                options={images.map((img) => ({
                  label: img.name,
                  value: img.name,
                }))}
                maxTagCount={3}
              />
            </div>

            {/* 检测阈值 */}
            <div style={{ marginBottom: 12 }}>
              <Text strong>检测置信度阈值: {scoreThr.toFixed(2)}</Text>
              <Slider
                min={0.05}
                max={0.95}
                step={0.05}
                value={scoreThr}
                onChange={setScoreThr}
              />
            </div>

            {/* 导出格式 */}
            <div style={{ marginBottom: 12 }}>
              <Text strong>导出格式</Text>
              <Select
                style={{ width: '100%', marginTop: 4 }}
                value={exportFormat}
                onChange={setExportFormat}
                options={[
                  { label: 'COCO JSON', value: 'coco' },
                  { label: 'VOC XML', value: 'voc' },
                ]}
              />
            </div>

            {/* 执行按钮 */}
            <Button
              type="primary"
              icon={<PlayCircleOutlined />}
              onClick={handleAnnotate}
              loading={loading}
              block
              size="large"
            >
              一键自动标注
            </Button>
          </Card>
        </Col>

        {/* 右侧：结果展示 */}
        <Col span={16}>
          <Card
            title="🖼️ 标注结果"
            size="small"
            extra={
              results.length > 1 && (
                <Select
                  value={viewIndex}
                  onChange={setViewIndex}
                  style={{ width: 200 }}
                  options={results.map((r, i) => ({
                    label: `${r.image} (${r.count}个)`,
                    value: i,
                  }))}
                />
              )
            }
          >
            <Spin spinning={loading} tip="正在标注，请稍候...">
              {results.length > 0 && results[viewIndex] ? (
                <AnnotationViewer
                  imageUrl={getImageUrl(results[viewIndex].image)}
                  annotations={results[viewIndex].annotations}
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
                  {loading ? '' : '输入文本提示并选择图像，点击"一键自动标注"开始'}
                </div>
              )}
            </Spin>
          </Card>
        </Col>
      </Row>
    </div>
  );
};

export default Mode1Panel;
