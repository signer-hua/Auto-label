/**
 * Zustand 全局状态管理
 *
 * 管理：标注模式（mode1/mode2）、文本提示、上传图片列表、
 *       当前选中图片、框选坐标、task_id、任务状态、进度、Mask URL 映射表。
 */
import { create } from 'zustand';

/** 上传的图像信息 */
export interface ImageItem {
  id: string;
  filename: string;
  url: string;       // 原图访问 URL
  path: string;      // 后端本地路径
}

/** 框选坐标 */
export interface BBox {
  x: number;
  y: number;
  width: number;
  height: number;
}

/** 工具类型 */
export type ToolType = 'select' | 'pan' | 'zoom';

/** 标注模式 */
export type AnnotationMode = 'mode1' | 'mode2';

/** 任务状态 */
export type TaskStatus = 'idle' | 'pending' | 'processing' | 'success' | 'failed';

/** 全局状态 */
interface AppState {
  // ===== 标注模式 =====
  currentMode: AnnotationMode;            // 当前标注模式
  textPrompt: string;                     // 模式1 文本提示

  // ===== 图片管理 =====
  images: ImageItem[];                    // 已上传图片列表
  selectedImageId: string | null;         // 当前选中的图片 ID（参考图 / 主画布显示）
  viewingImageId: string | null;          // 右侧面板正在查看的图片 ID

  // ===== 交互工具 =====
  activeTool: ToolType;                   // 当前激活的工具
  bbox: BBox | null;                      // 用户框选的坐标（画布像素坐标）
  isDrawing: boolean;                     // 是否正在框选

  // ===== 异步任务 =====
  taskId: string | null;                  // 当前任务 ID
  taskStatus: TaskStatus;                 // 任务状态
  taskProgress: number;                   // 已完成数量
  taskTotal: number;                      // 总数量
  taskMessage: string;                    // 状态消息

  // ===== Mask 结果 =====
  maskUrls: Record<string, string[]>;     // {image_id: [mask_url, ...]}
  exportUrl: string | null;               // COCO 导出 URL

  // ===== 画布状态 =====
  stageScale: number;                     // 缩放比例
  stagePosition: { x: number; y: number }; // 平移偏移

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
  setMaskUrls: (urls: Record<string, string[]>) => void;
  setExportUrl: (url: string | null) => void;
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
  maskUrls: {},
  exportUrl: null,
  stageScale: 1,
  stagePosition: { x: 0, y: 0 },

  // Actions
  setCurrentMode: (mode) =>
    set({ currentMode: mode, bbox: null }),

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
    set({ taskId, taskStatus: status, taskProgress: 0, taskTotal: 0, taskMessage: '' }),

  updateTaskProgress: (progress, total, message) =>
    set({ taskProgress: progress, taskTotal: total, taskMessage: message }),

  setTaskStatus: (status, message) =>
    set((s) => ({ taskStatus: status, taskMessage: message ?? s.taskMessage })),

  setMaskUrls: (urls) =>
    set({ maskUrls: urls }),

  setExportUrl: (url) =>
    set({ exportUrl: url }),

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
      maskUrls: {},
      exportUrl: null,
    }),
}));
