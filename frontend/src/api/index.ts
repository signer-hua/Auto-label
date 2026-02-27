/**
 * 后端 API 接口封装
 * 统一管理所有与后端的通信
 */
import axios from 'axios';

const api = axios.create({
  baseURL: '/api',
  timeout: 300000, // 5分钟超时（模型推理可能较慢）
});

// ==================== 类型定义 ====================

/** 标注结果 */
export interface Annotation {
  segmentation?: number[][];
  bbox?: number[];
  area?: number;
  label?: string;
  label_id?: number;
  score?: number;
  dino_feature?: number[];
  mask_score?: number;
}

/** 聚类结果 */
export interface ClusterData {
  cluster_id: number;
  segmentation: number[][];
  bbox: number[];
  area: number;
  feature: number[];
}

// ==================== 文件管理 API ====================

/** 上传图像 */
export async function uploadImages(files: File[]) {
  const formData = new FormData();
  files.forEach((f) => formData.append('files', f));
  const res = await api.post('/files/upload', formData, {
    headers: { 'Content-Type': 'multipart/form-data' },
  });
  return res.data;
}

/** 获取图像列表 */
export async function listImages() {
  const res = await api.get('/files/list');
  return res.data;
}

/** 获取图像 URL */
export function getImageUrl(filename: string) {
  return `/uploads/${filename}`;
}

/** 删除图像 */
export async function deleteImage(filename: string) {
  const res = await api.delete(`/files/delete/${filename}`);
  return res.data;
}

// ==================== 标注 API ====================

/** 模式1：文本提示一键标注 */
export async function mode1Annotate(params: {
  image_names: string[];
  text_prompts: string[];
  score_thr?: number;
  export_format?: string;
}) {
  const res = await api.post('/annotate/mode1', params);
  return res.data;
}

/** 模式2：人工预标注 → 批量标注 */
export async function mode2Annotate(params: {
  ref_image_name: string;
  user_boxes: number[][];
  target_image_names: string[];
  similarity_threshold?: number;
  export_format?: string;
}) {
  const res = await api.post('/annotate/mode2', params);
  return res.data;
}

/** 模式3第一步：全图聚类 */
export async function mode3Cluster(params: {
  image_name: string;
  n_clusters?: number;
}) {
  const res = await api.post('/annotate/mode3/cluster', params);
  return res.data;
}

/** 模式3第二步：跨图标注 */
export async function mode3Annotate(params: {
  selected_feature: number[];
  target_image_names: string[];
  similarity_threshold?: number;
  export_format?: string;
}) {
  const res = await api.post('/annotate/mode3/annotate', params);
  return res.data;
}

/** 更新标注参数 */
export async function updateParams(params: {
  similarity_threshold?: number;
  kmeans_clusters?: number;
  score_thr?: number;
}) {
  const res = await api.post('/annotate/update_params', params);
  return res.data;
}

// ==================== 导出 API ====================

/** 列出导出结果 */
export async function listExports() {
  const res = await api.get('/export/list');
  return res.data;
}

/** 下载标注结果 */
export function getExportDownloadUrl(filename: string) {
  return `/api/export/download/${filename}`;
}

/** 下载所有标注结果（ZIP） */
export function getExportAllUrl() {
  return '/api/export/download_all';
}

export default api;
