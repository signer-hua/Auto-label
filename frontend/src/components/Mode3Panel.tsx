/**
 * 模式3：选实例 → 跨图批量标注
 * 全图聚类粗分割 → 用户选中实例 → DINOv3 跨图匹配 → SAM3 精准分割
 */
import React, { useState } from 'react';
import {
  Card, Button, Select, Slider, Row, Col,
  Spin, message, Typography, Tag, Divider,
} from 'antd';
import {
  AppstoreOutlined, PlayCircleOutlined, SelectOutlined,
} from '@ant-design/icons';
import { mode3Cluster, mode3Annotate, getImageUrl, ClusterData } from '../api';
import AnnotationViewer from './AnnotationViewer';

const { Text } = Typography;

interface Props {
  images: { name: string; path: string; size: number }[];
  onRefresh: () => void;
}

/** 聚类颜色表 */
const CLUSTER_COLORS = [
  '#1890ff', '#52c41a', '#faad14', '#f5222d', '#722ed1',
  '#13c2c2', '#eb2f96', '#fa8c16', '#a0d911', '#2f54eb',
];

const Mode3Panel: React.FC<Props> = ({ images, onRefresh }) => {
  // 选中的图像
  const [selectedImage, setSelectedImage] = useState<string>('');
  // 聚类数
  const [nClusters, setNClusters] = useState(10);
  // 聚类结果
  const [clusters, setClusters] = useState<ClusterData[]>([]);
  // 选中的聚类实例
  const [selectedCluster, setSelectedCluster] = useState<number | null>(null);
  // 目标图像
  const [targetImages, setTargetImages] = useState<string[]>([]);
  // 相似度阈值
  const [threshold, setThreshold] = useState(0.8);
  // 加载状态
  const [clusterLoading, setClusterLoading] = useState(false);
  const [annotateLoading, setAnnotateLoading] = useState(false);
  // 跨图标注结果
  const [results, setResults] = useState<any>(null);
  const [viewIndex, setViewIndex] = useState(0);

  /** 第一步：全图聚类 */
  const handleCluster = async () => {
    if (!selectedImage) {
      message.warning('请选择图像');
      return;
    }

    setClusterLoading(true);
    setClusters([]);
    setSelectedCluster(null);
    try {
      const data = await mode3Cluster({
        image_name: selectedImage,
        n_clusters: nClusters,
      });
      setClusters(data.clusters || []);
      message.success(`聚类完成，共 ${data.cluster_count} 个区域`);
    } catch (err: any) {
      message.error('聚类失败: ' + (err.response?.data?.detail || err.message));
    } finally {
      setClusterLoading(false);
    }
  };

  /** 第二步：跨图标注 */
  const handleAnnotate = async () => {
    if (selectedCluster === null) {
      message.warning('请先选中一个聚类实例');
      return;
    }
    if (targetImages.length === 0) {
      message.warning('请选择目标图像');
      return;
    }

    const feature = clusters[selectedCluster]?.feature;
    if (!feature || feature.length === 0) {
      message.error('选中实例无有效特征');
      return;
    }

    setAnnotateLoading(true);
    try {
      const data = await mode3Annotate({
        selected_feature: feature,
        target_image_names: targetImages,
        similarity_threshold: threshold,
      });
      setResults(data);
      setViewIndex(0);
      message.success(`跨图标注完成，共 ${data.total_annotations} 个实例`);
    } catch (err: any) {
      message.error('标注失败: ' + (err.response?.data?.detail || err.message));
    } finally {
      setAnnotateLoading(false);
    }
  };

  return (
    <div>
      <Row gutter={16}>
        {/* 左侧：聚类 + 选择 */}
        <Col span={12}>
          {/* 第一步：聚类 */}
          <Card title="🔍 第一步：全图聚类粗分割" size="small">
            <div style={{ marginBottom: 8 }}>
              <Text strong>选择图像</Text>
              <Select
                style={{ width: '100%', marginTop: 4 }}
                placeholder="选择要聚类的图像"
                value={selectedImage || undefined}
                onChange={(v) => {
                  setSelectedImage(v);
                  setClusters([]);
                  setSelectedCluster(null);
                }}
                options={images.map((img) => ({
                  label: img.name,
                  value: img.name,
                }))}
              />
            </div>
            <div style={{ marginBottom: 8 }}>
              <Text strong>聚类数: {nClusters}</Text>
              <Slider
                min={3}
                max={20}
                value={nClusters}
                onChange={setNClusters}
              />
            </div>
            <Button
              type="primary"
              icon={<AppstoreOutlined />}
              onClick={handleCluster}
              loading={clusterLoading}
              block
            >
              执行聚类
            </Button>
          </Card>

          {/* 聚类结果：选择实例 */}
          {clusters.length > 0 && (
            <Card
              title="🎯 第二步：选择目标实例"
              size="small"
              style={{ marginTop: 12 }}
            >
              <div style={{ marginBottom: 8 }}>
                <Text type="secondary">
                  点击选中要跨图标注的实例区域
                </Text>
              </div>

              {/* 聚类列表 */}
              <div style={{ display: 'flex', flexWrap: 'wrap', gap: 8 }}>
                {clusters.map((cluster, i) => (
                  <Tag
                    key={i}
                    color={selectedCluster === i ? CLUSTER_COLORS[i % CLUSTER_COLORS.length] : undefined}
                    style={{
                      cursor: 'pointer',
                      border: selectedCluster === i ? '2px solid #000' : undefined,
                      padding: '4px 12px',
                      fontSize: 14,
                    }}
                    onClick={() => setSelectedCluster(i)}
                  >
                    区域 #{i + 1} (面积: {cluster.area.toFixed(0)})
                  </Tag>
                ))}
              </div>

              {/* 聚类可视化 */}
              {selectedImage && (
                <div style={{ marginTop: 12, position: 'relative' }}>
                  <img
                    src={getImageUrl(selectedImage)}
                    alt="聚类结果"
                    style={{ maxWidth: '100%', display: 'block' }}
                  />
                  {/* SVG 覆盖：显示聚类边界框 */}
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
                    {clusters.map((cluster, i) => {
                      if (!cluster.bbox || cluster.bbox.length < 4) return null;
                      const [x, y, w, h] = cluster.bbox;
                      const color = CLUSTER_COLORS[i % CLUSTER_COLORS.length];
                      const isSelected = selectedCluster === i;
                      return (
                        <rect
                          key={i}
                          x={`${(x / 100) * 100}%`}
                          y={`${(y / 100) * 100}%`}
                          width={`${(w / 100) * 100}%`}
                          height={`${(h / 100) * 100}%`}
                          fill={isSelected ? `${color}44` : `${color}22`}
                          stroke={color}
                          strokeWidth={isSelected ? 3 : 1}
                        />
                      );
                    })}
                  </svg>
                </div>
              )}

              <Divider />

              {/* 跨图标注配置 */}
              <div style={{ marginBottom: 8 }}>
                <Text strong>目标图像</Text>
                <Select
                  mode="multiple"
                  style={{ width: '100%', marginTop: 4 }}
                  placeholder="选择待标注的目标图像"
                  value={targetImages}
                  onChange={setTargetImages}
                  options={images
                    .filter((img) => img.name !== selectedImage)
                    .map((img) => ({ label: img.name, value: img.name }))}
                  maxTagCount={3}
                />
              </div>
              <div style={{ marginBottom: 8 }}>
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
                icon={<SelectOutlined />}
                onClick={handleAnnotate}
                loading={annotateLoading}
                disabled={selectedCluster === null}
                block
                size="large"
              >
                跨图批量标注
              </Button>
            </Card>
          )}
        </Col>

        {/* 右侧：跨图标注结果 */}
        <Col span={12}>
          <Card
            title="🖼️ 跨图标注结果"
            size="small"
            extra={
              results?.results?.length > 1 && (
                <Select
                  value={viewIndex}
                  onChange={setViewIndex}
                  style={{ width: 200 }}
                  options={results.results.map((r: any, i: number) => ({
                    label: `${r.image} (${r.count}个)`,
                    value: i,
                  }))}
                />
              )
            }
          >
            <Spin spinning={annotateLoading} tip="正在跨图标注...">
              {results?.results?.[viewIndex] ? (
                <AnnotationViewer
                  imageUrl={getImageUrl(results.results[viewIndex].image)}
                  annotations={results.results[viewIndex].annotations}
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
                  {annotateLoading
                    ? ''
                    : '先执行聚类 → 选中实例 → 选择目标图像 → 跨图标注'}
                </div>
              )}
            </Spin>
          </Card>
        </Col>
      </Row>
    </div>
  );
};

export default Mode3Panel;
