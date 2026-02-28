/**
 * 后端 API 接口封装
 * 所有请求通过 Axios 发送，baseURL 由 Vite proxy 代理到后端。
 */
import axios from 'axios';

const api = axios.create({
  baseURL: '/api',
  timeout: 30000,
});

// ==================== 类型定义 ====================

export interface UploadResponse {
  image_id: string;
  filename: string;
  url: string;
  path: string;
}

export interface TaskStatusResponse {
  task_id: string;
  status: 'pending' | 'processing' | 'success' | 'failed';
  progress: number;
  total: number;
  message: string;
  mask_urls: Record<string, string[]> | null;
  export_url: string | null;
}

// ==================== 接口方法 ====================

/**
 * 上传图像文件
 */
export async function uploadImage(file: File): Promise<UploadResponse> {
  const formData = new FormData();
  formData.append('file', file);
  const { data } = await api.post<UploadResponse>('/upload', formData, {
    headers: { 'Content-Type': 'multipart/form-data' },
  });
  return data;
}

/**
 * 批量上传图像
 */
export async function uploadImages(files: File[]): Promise<UploadResponse[]> {
  const results: UploadResponse[] = [];
  for (const file of files) {
    const res = await uploadImage(file);
    results.push(res);
  }
  return results;
}

/**
 * 触发模式2异步批量标注
 * 返回 task_id，后续通过 pollTaskStatus 轮询进度。
 */
export async function startMode2Annotation(params: {
  ref_image_id: string;
  ref_image_path: string;
  bbox: [number, number, number, number]; // [x1, y1, x2, y2]
  target_images: { id: string; path: string }[];
}): Promise<{ task_id: string; status: string }> {
  const { data } = await api.post('/annotate/mode2', params);
  return data;
}

/**
 * 查询任务状态（前端每 1 秒轮询）
 */
export async function getTaskStatus(taskId: string): Promise<TaskStatusResponse> {
  const { data } = await api.get<TaskStatusResponse>(`/tasks/${taskId}`);
  return data;
}

/**
 * 获取已上传的图片列表（页面刷新后恢复）
 */
export async function listImages(): Promise<UploadResponse[]> {
  const { data } = await api.get<{ images: UploadResponse[]; count: number }>('/images');
  return data.images;
}

/**
 * 删除指定图片
 */
export async function deleteImage(imageId: string): Promise<void> {
  await api.delete(`/images/${imageId}`);
}

export default api;
