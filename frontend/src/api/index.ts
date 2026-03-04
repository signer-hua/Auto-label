/**
 * 后端 API 接口封装（v3 增强版）
 * v3 新增：手动标注 SAM API + 多参考图 + 评分数据
 */
import axios from 'axios';

const api = axios.create({ baseURL: '/api', timeout: 30000 });

export interface UploadResponse {
  image_id: string;
  filename: string;
  url: string;
  path: string;
}

export interface ImageScoreData {
  total: number;
  similarity: number;
  mask_coverage: number;
  area: number;
  detection: number;
  level: 'high' | 'medium' | 'low';
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
  instance_masks: Array<{ id: number; mask_url: string; bbox: number[]; color: number[] }> | null;
  error_type?: string;
  image_scores?: Record<string, ImageScoreData> | null;
  mask_url?: string | null;
}

export interface RefImageParam {
  path: string;
  bbox: [number, number, number, number];
  weight: number;
}

export interface CategoryBboxRef {
  name: string;
  bboxes: [number, number, number, number][];
  ref_images?: RefImageParam[];
}

export interface CategoryInstanceRef {
  name: string;
  instance_ids: number[];
}

// ===== 上传 =====
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
  for (const file of files) results.push(await uploadImage(file));
  return results;
}

// ===== 模式1（支持全局类别绑定） =====
export async function startMode1Annotation(params: {
  text_prompt: string;
  image_ids: string[];
  image_paths: string[];
  category_name?: string | null;
  category_color?: string | null;
}): Promise<{ task_id: string; status: string; mode: string }> {
  const { data } = await api.post('/annotate/mode1', params);
  return data;
}

// ===== 模式1 目标预览（轻量检测，不生成 Mask） =====
export async function previewMode1(params: {
  text_prompt: string;
  image_id: string;
  image_path: string;
}): Promise<{ detections: Array<{ box: number[]; label: string; score: number }> }> {
  const { data } = await api.post('/annotate/preview', params);
  return data;
}

// ===== 模式2（多参考图） =====
export async function startMode2Annotation(params: {
  ref_image_id: string;
  ref_image_path: string;
  bbox: [number, number, number, number];
  target_images: Array<{ id: string; path: string }>;
  categories?: CategoryBboxRef[];
  ref_images?: RefImageParam[];
}): Promise<{ task_id: string; status: string; mode: string }> {
  const { data } = await api.post('/annotate/mode2', params);
  return data;
}

// ===== 模式3 =====
export async function startMode3Discovery(params: {
  ref_image_id: string; ref_image_path: string;
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
  ref_images?: RefImageParam[];
}): Promise<{ task_id: string; status: string; mode: string }> {
  const { data } = await api.post('/annotate/mode3/select', params);
  return data;
}

// ===== 手动标注（支持类别颜色） =====
export async function startManualSam(params: {
  image_id: string;
  image_path: string;
  bbox: [number, number, number, number];
  category_color?: string | null;
  category_name?: string | null;
}): Promise<{ task_id: string; status: string; mode: string }> {
  const { data } = await api.post('/annotate/manual', params);
  return data;
}

// ===== 人机协同负向提示修正 =====
export async function startCorrectMask(params: {
  image_id: string;
  image_path: string;
  positive_boxes: [number, number, number, number][];
  negative_boxes: [number, number, number, number][];
  category_color?: string | null;
  category_name?: string | null;
}): Promise<{ task_id: string; status: string; mode: string }> {
  const { data } = await api.post('/annotate/correct', params);
  return data;
}

// ===== 任务控制 =====
export async function pauseTask(taskId: string): Promise<void> { await api.post(`/tasks/${taskId}/pause`); }
export async function resumeTask(taskId: string): Promise<void> { await api.post(`/tasks/${taskId}/resume`); }
export async function cancelTask(taskId: string): Promise<void> { await api.post(`/tasks/${taskId}/cancel`); }

export async function getTaskStatus(taskId: string): Promise<TaskStatusResponse> {
  const { data } = await api.get<TaskStatusResponse>(`/tasks/${taskId}`);
  return data;
}

// ===== 图片管理 =====
export async function listImages(): Promise<UploadResponse[]> {
  const { data } = await api.get<{ images: UploadResponse[]; count: number }>('/images');
  return data.images;
}

export async function deleteImage(imageId: string): Promise<void> {
  await api.delete(`/images/${imageId}`);
}

// ===== 导出 =====
export async function exportAnnotations(taskId: string, format: 'coco' | 'voc' | 'yolo'): Promise<any> {
  const { data } = await api.get(`/export/${taskId}/${format}`);
  return data;
}

export default api;
