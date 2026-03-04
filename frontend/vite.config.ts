import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      // API 请求代理到后端
      '/api': {
        target: 'http://localhost:8000',
        changeOrigin: true,
      },
      // 静态文件（图片、Mask、导出）代理到后端
      '/data': {
        target: 'http://localhost:8000',
        changeOrigin: true,
      },
    },
  },
});
