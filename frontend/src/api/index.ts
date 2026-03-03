/**
 * 后端 API 接口封装（增强版）
 * v2 增强：模式2/3 多类别标注 API 支持
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
  status: string;
  progress: number;
  total: number;
  message: string;
  mode?: string;
  mask_urls: Record<string, string[]> | null;
  export_url: string | null;
  instance_masks: Array<{
    id: number;
    mask_url: string;
    bbox: number[];
    color: number[];
  }> | null;
  error_type?: string;
}

/** 模式2 多类别框选参考 */
export interface CategoryBboxRef {
  name: string;
  bboxes: [number, number, number, number][];
}

/** 模式3 多类别实例参考 */
export interface CategoryInstanceRef {
  name: string;
  instance_ids: number[];
}

// ==================== 上传 ====================

export async function uploadImage(file: File): Promise<UploadResponse> {
  const formData = new FormData();
  formData.append('file', file);
  const { data } = await api.post<UploadResponse>('/upload', formData, {
    headers: { 'Content-Type': 'multipart/form-data' },
  });
  return data;
}

export async function uploadImages(files: File[]): Promise<UploadResponse[]> {
  const results: UploadResponse[] = [];
  for (const file of files) {
    const result = await uploadImage(file);
    results.push(result);
  }
  return results;
}

// ==================== 模式1：文本标注 ====================

export async function startMode1Annotation(params: {
  text_prompt: string;
  image_ids: string[];
  image_paths: string[];
}): Promise<{ task_id: string; status: string; mode: string }> {
  const { data } = await api.post('/annotate/mode1', params);
  return data;
}

// ==================== 模式2：框选批量标注（支持多类别） ====================

export async function startMode2Annotation(params: {
  ref_image_id: string;
  ref_image_path: string;
  bbox: [number, number, number, number];
  target_images: Array<{ id: string; path: string }>;
  categories?: CategoryBboxRef[];
}): Promise<{ task_id: string; status: string; mode: string }> {
  const { data } = await api.post('/annotate/mode2', params);
  return data;
}

// ==================== 模式3：选实例跨图标注（支持多类别 + 多实例） ====================

export async function startMode3Discovery(params: {
  ref_image_id: string;
  ref_image_path: string;
}): Promise<{ task_id: string; status: string; mode: string }> {
  const { data } = await api.post('/annotate/mode3', params);
  return data;
}

export async function startMode3Select(params: {
  discovery_task_id: string;
  ref_image_id: string;
  ref_image_path: string;
  selected_instance_id: number;
  target_images: Array<{ id: string; path: string }>;
  categories?: CategoryInstanceRef[];
}): Promise<{ task_id: string; status: string; mode: string }> {
  const { data } = await api.post('/annotate/mode3/select', params);
  return data;
}

// ==================== 任务控制 ====================

export async function pauseTask(taskId: string): Promise<void> {
  await api.post(`/tasks/${taskId}/pause`);
}

export async function resumeTask(taskId: string): Promise<void> {
  await api.post(`/tasks/${taskId}/resume`);
}

export async function cancelTask(taskId: string): Promise<void> {
  await api.post(`/tasks/${taskId}/cancel`);
}

// ==================== 任务查询 ====================

export async function getTaskStatus(taskId: string): Promise<TaskStatusResponse> {
  const { data } = await api.get<TaskStatusResponse>(`/tasks/${taskId}`);
  return data;
}

// ==================== 图片管理 ====================

export async function listImages(): Promise<UploadResponse[]> {
  const { data } = await api.get<{ images: UploadResponse[]; count: number }>('/images');
  return data.images;
}

export async function deleteImage(imageId: string): Promise<void> {
  await api.delete(`/images/${imageId}`);
}

// ==================== 多格式导出 ====================

export async function exportAnnotations(taskId: string, format: 'coco' | 'voc' | 'yolo'): Promise<any> {
  const { data } = await api.get(`/export/${taskId}/${format}`);
  return data;
}

export default api;
