/**
 * Zustand 全局状态管理
 *
 * 管理：三种标注模式（mode1/mode2/mode3）、文本提示、上传图片列表、
 *       框选坐标、task_id、任务状态、进度、Mask URL 映射表、
 *       模式3 实例选择、Mask 透明度、任务控制（暂停/取消）。
 */
import { create } from 'zustand';

/** 上传的图像信息 */
export interface ImageItem {
  id: string;
  filename: string;
  url: string;
  path: string;
}

/** 框选坐标 */
export interface BBox {
  x: number;
  y: number;
  width: number;
  height: number;
}

/** 模式3 粗分割实例 */
export interface InstanceItem {
  id: number;
  mask_url: string;
  bbox: number[];
  color: number[];
}

/** 工具类型 */
export type ToolType = 'select' | 'pan' | 'zoom';

/** 标注模式 */
export type AnnotationMode = 'mode1' | 'mode2' | 'mode3';

/** 任务状态 */
export type TaskStatus =
  | 'idle' | 'pending' | 'processing' | 'success' | 'failed'
  | 'paused' | 'canceled' | 'instance_ready';

/** 全局状态 */
interface AppState {
  // ===== 标注模式 =====
  currentMode: AnnotationMode;
  textPrompt: string;

  // ===== 图片管理 =====
  images: ImageItem[];
  selectedImageId: string | null;
  viewingImageId: string | null;

  // ===== 交互工具 =====
  activeTool: ToolType;
  bbox: BBox | null;
  isDrawing: boolean;

  // ===== 异步任务 =====
  taskId: string | null;
  taskStatus: TaskStatus;
  taskProgress: number;
  taskTotal: number;
  taskMessage: string;
  errorType: string | null;

  // ===== Mask 结果 =====
  maskUrls: Record<string, string[]>;
  exportUrl: string | null;
  maskOpacity: number;  // Mask 透明度 0~1

  // ===== 模式3 实例选择 =====
  discoveryTaskId: string | null;
  instanceMasks: InstanceItem[];
  selectedInstanceId: number | null;

  // ===== 画布状态 =====
  stageScale: number;
  stagePosition: { x: number; y: number };

  // ===== Actions =====
  setCurrentMode: (mode: AnnotationMode) => void;
  setTextPrompt: (text: string) => void;
  addImage: (img: ImageItem) => void;
  addImages: (imgs: ImageItem[]) => void;
  setImages: (imgs: ImageItem[]) => void;
  removeImage: (id: string) => void;
  selectImage: (id: string | null) => void;
  setViewingImage: (id: string | null) => void;
  setActiveTool: (tool: ToolType) => void;
  setBBox: (bbox: BBox | null) => void;
  setIsDrawing: (drawing: boolean) => void;
  setTask: (taskId: string, status: TaskStatus) => void;
  updateTaskProgress: (progress: number, total: number, message: string) => void;
  setTaskStatus: (status: TaskStatus, message?: string) => void;
  setErrorType: (errorType: string | null) => void;
  setMaskUrls: (urls: Record<string, string[]>) => void;
  setExportUrl: (url: string | null) => void;
  setMaskOpacity: (opacity: number) => void;
  setDiscoveryTaskId: (id: string | null) => void;
  setInstanceMasks: (instances: InstanceItem[]) => void;
  setSelectedInstanceId: (id: number | null) => void;
  setStageScale: (scale: number) => void;
  setStagePosition: (pos: { x: number; y: number }) => void;
  resetTask: () => void;
}

export const useAppStore = create<AppState>((set) => ({
  // 初始状态
  currentMode: 'mode1',
  textPrompt: '',
  images: [],
  selectedImageId: null,
  viewingImageId: null,
  activeTool: 'select',
  bbox: null,
  isDrawing: false,
  taskId: null,
  taskStatus: 'idle',
  taskProgress: 0,
  taskTotal: 0,
  taskMessage: '',
  errorType: null,
  maskUrls: {},
  exportUrl: null,
  maskOpacity: 0.7,
  discoveryTaskId: null,
  instanceMasks: [],
  selectedInstanceId: null,
  stageScale: 1,
  stagePosition: { x: 0, y: 0 },

  // Actions
  setCurrentMode: (mode) =>
    set({ currentMode: mode, bbox: null, instanceMasks: [], selectedInstanceId: null }),

  setTextPrompt: (text) =>
    set({ textPrompt: text }),

  addImage: (img) =>
    set((s) => ({ images: [...s.images, img] })),

  addImages: (imgs) =>
    set((s) => ({ images: [...s.images, ...imgs] })),

  setImages: (imgs) =>
    set({ images: imgs }),

  removeImage: (id) =>
    set((s) => {
      const images = s.images.filter((img) => img.id !== id);
      const updates: Partial<AppState> = { images };
      if (s.selectedImageId === id) updates.selectedImageId = images[0]?.id || null;
      if (s.viewingImageId === id) updates.viewingImageId = null;
      return updates;
    }),

  selectImage: (id) =>
    set({ selectedImageId: id, bbox: null }),

  setViewingImage: (id) =>
    set({ viewingImageId: id }),

  setActiveTool: (tool) =>
    set({ activeTool: tool }),

  setBBox: (bbox) =>
    set({ bbox }),

  setIsDrawing: (drawing) =>
    set({ isDrawing: drawing }),

  setTask: (taskId, status) =>
    set({ taskId, taskStatus: status, taskProgress: 0, taskTotal: 0, taskMessage: '', errorType: null }),

  updateTaskProgress: (progress, total, message) =>
    set({ taskProgress: progress, taskTotal: total, taskMessage: message }),

  setTaskStatus: (status, message) =>
    set((s) => ({ taskStatus: status, taskMessage: message ?? s.taskMessage })),

  setErrorType: (errorType) =>
    set({ errorType }),

  setMaskUrls: (urls) =>
    set({ maskUrls: urls }),

  setExportUrl: (url) =>
    set({ exportUrl: url }),

  setMaskOpacity: (opacity) =>
    set({ maskOpacity: opacity }),

  setDiscoveryTaskId: (id) =>
    set({ discoveryTaskId: id }),

  setInstanceMasks: (instances) =>
    set({ instanceMasks: instances }),

  setSelectedInstanceId: (id) =>
    set({ selectedInstanceId: id }),

  setStageScale: (scale) =>
    set({ stageScale: scale }),

  setStagePosition: (pos) =>
    set({ stagePosition: pos }),

  resetTask: () =>
    set({
      taskId: null,
      taskStatus: 'idle',
      taskProgress: 0,
      taskTotal: 0,
      taskMessage: '',
      errorType: null,
      maskUrls: {},
      exportUrl: null,
      discoveryTaskId: null,
      instanceMasks: [],
      selectedInstanceId: null,
    }),
}));
