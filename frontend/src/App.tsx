/**
 * Auto-label 主应用
 * 布局：左侧 Toolbar | 中间 MainCanvas | 右侧 RightPanel
 */
import React from 'react';
import { ConfigProvider, theme } from 'antd';
import Toolbar from './components/Toolbar';
import MainCanvas from './components/MainCanvas';
import RightPanel from './components/RightPanel';
import './styles/global.css';

const App: React.FC = () => {
  return (
    <ConfigProvider
      theme={{
        algorithm: theme.darkAlgorithm,
        token: {
          colorPrimary: '#1890ff',
          borderRadius: 6,
        },
      }}
    >
      <div style={{
        display: 'flex',
        height: '100vh',
        width: '100vw',
        overflow: 'hidden',
        background: '#1a1a1a',
      }}>
        {/* 左侧工具栏 */}
        <Toolbar />

        {/* 中间主画布 */}
        <MainCanvas />

        {/* 右侧面板 */}
        <RightPanel />
      </div>
    </ConfigProvider>
  );
};

export default App;
