/**
 * App 主组件
 * 提供标签页切换三种标注模式，以及图像上传和结果导出
 */
import React, { useState, useCallback } from 'react';
import { Layout, Tabs, Upload, Button, message, Space, Typography } from 'antd';
import {
  UploadOutlined,
  EditOutlined,
  AimOutlined,
  AppstoreOutlined,
  DownloadOutlined,
} from '@ant-design/icons';
import type { UploadFile } from 'antd';
import { uploadImages, listImages, getExportAllUrl } from './api';
import Mode1Panel from './components/Mode1Panel';
import Mode2Panel from './components/Mode2Panel';
import Mode3Panel from './components/Mode3Panel';

const { Header, Content, Sider } = Layout;
const { Title, Text } = Typography;

/** 已上传图像信息 */
interface ImageInfo {
  name: string;
  path: string;
  size: number;
}

const App: React.FC = () => {
  const [images, setImages] = useState<ImageInfo[]>([]);
  const [uploading, setUploading] = useState(false);
  const [activeTab, setActiveTab] = useState('mode1');

  /** 刷新图像列表 */
  const refreshImages = useCallback(async () => {
    try {
      const data = await listImages();
      setImages(data.images || []);
    } catch {
      message.error('获取图像列表失败');
    }
  }, []);

  /** 处理图像上传 */
  const handleUpload = useCallback(
    async (options: any) => {
      const { file, onSuccess, onError } = options;
      setUploading(true);
      try {
        await uploadImages([file]);
        onSuccess?.({}, file);
        message.success(`${file.name} 上传成功`);
        await refreshImages();
      } catch (err) {
        onError?.(err);
        message.error(`${file.name} 上传失败`);
      } finally {
        setUploading(false);
      }
    },
    [refreshImages]
  );

  /** 标签页配置 */
  const tabItems = [
    {
      key: 'mode1',
      label: (
        <span>
          <EditOutlined /> 文本提示标注
        </span>
      ),
      children: <Mode1Panel images={images} onRefresh={refreshImages} />,
    },
    {
      key: 'mode2',
      label: (
        <span>
          <AimOutlined /> 人工预标注
        </span>
      ),
      children: <Mode2Panel images={images} onRefresh={refreshImages} />,
    },
    {
      key: 'mode3',
      label: (
        <span>
          <AppstoreOutlined /> 选实例跨图标
        </span>
      ),
      children: <Mode3Panel images={images} onRefresh={refreshImages} />,
    },
  ];

  return (
    <Layout style={{ minHeight: '100vh' }}>
      {/* 顶部导航 */}
      <Header
        style={{
          background: '#001529',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          padding: '0 24px',
        }}
      >
        <Title level={4} style={{ color: '#fff', margin: 0 }}>
          🏷️ Auto-label 人机协同图像自动标注工具
        </Title>
        <Space>
          <Upload customRequest={handleUpload} showUploadList={false} multiple accept="image/*">
            <Button icon={<UploadOutlined />} loading={uploading} type="primary">
              上传图像
            </Button>
          </Upload>
          <Button
            icon={<DownloadOutlined />}
            href={getExportAllUrl()}
            target="_blank"
          >
            导出全部
          </Button>
        </Space>
      </Header>

      {/* 主内容区 */}
      <Content style={{ padding: '16px 24px' }}>
        {/* 图像数量提示 */}
        <div style={{ marginBottom: 12 }}>
          <Text type="secondary">
            已上传 {images.length} 张图像
            {images.length === 0 && ' — 请先上传图像'}
          </Text>
          <Button type="link" size="small" onClick={refreshImages}>
            刷新
          </Button>
        </div>

        {/* 三种标注模式标签页 */}
        <Tabs
          activeKey={activeTab}
          onChange={setActiveTab}
          items={tabItems}
          type="card"
          size="large"
        />
      </Content>
    </Layout>
  );
};

export default App;
