/**
 * Zustand 全局状态管理（增强版）
 *
 * v2 增强：
 *   - 图层隔离修复：bbox/实例与 imageId 强绑定，切图不污染
 *   - 多实例选择：selectedInstanceIds 替代 selectedInstanceId
 *   - 多类别管理：categories + activeCategoryId + 类别-参考绑定
 *   - 模式切换自动清空非当前模式画布元素
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

/** 用户定义的类别 */
export interface CategoryItem {
  id: string;
  name: string;
  color: string;
}

/** 模式2 类别-框选绑定 */
export interface Mode2CategoryRef {
  categoryId: string;
  bboxes: BBox[];
}

/** 模式3 类别-实例绑定 */
export interface Mode3CategoryRef {
  categoryId: string;
  instanceIds: number[];
}

/** 工具类型 */
export type ToolType = 'select' | 'pan' | 'zoom';

/** 标注模式 */
export type AnnotationMode = 'mode1' | 'mode2' | 'mode3';

/** 任务状态 */
export type TaskStatus =
  | 'idle' | 'pending' | 'processing' | 'success' | 'failed'
  | 'paused' | 'canceled' | 'instance_ready';

/** 多类别颜色预设 */
const CATEGORY_COLORS = [
  '#FF5050', '#50C850', '#5078FF', '#FFC800',
  '#C850FF', '#00DCB4', '#FF7800', '#B4B400',
  '#6496FF', '#FF6464',
];

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
  bboxImageId: string | null;  // bbox 绑定的图片 ID（图层隔离）
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
  maskOpacity: number;

  // ===== 模式3 实例选择（多选） =====
  discoveryTaskId: string | null;
  instanceMasks: InstanceItem[];
  instanceMasksImageId: string | null;  // 实例绑定的图片 ID（图层隔离）
  selectedInstanceIds: number[];  // 多选（替代原 selectedInstanceId）

  // ===== 多类别管理 =====
  categories: CategoryItem[];
  activeCategoryId: string | null;
  mode2CategoryRefs: Mode2CategoryRef[];  // 模式2 类别-框选绑定
  mode3CategoryRefs: Mode3CategoryRef[];  // 模式3 类别-实例绑定

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
  setBBoxImageId: (id: string | null) => void;
  setIsDrawing: (drawing: boolean) => void;
  setTask: (taskId: string, status: TaskStatus) => void;
  updateTaskProgress: (progress: number, total: number, message: string) => void;
  setTaskStatus: (status: TaskStatus, message?: string) => void;
  setErrorType: (errorType: string | null) => void;
  setMaskUrls: (urls: Record<string, string[]>) => void;
  setExportUrl: (url: string | null) => void;
  setMaskOpacity: (opacity: number) => void;
  setDiscoveryTaskId: (id: string | null) => void;
  setInstanceMasks: (instances: InstanceItem[], imageId?: string) => void;
  toggleInstanceId: (id: number) => void;
  setSelectedInstanceIds: (ids: number[]) => void;
  setStageScale: (scale: number) => void;
  setStagePosition: (pos: { x: number; y: number }) => void;
  resetTask: () => void;
  // 多类别
  addCategory: (name: string) => void;
  removeCategory: (id: string) => void;
  setActiveCategoryId: (id: string | null) => void;
  clearCategories: () => void;
  confirmMode2Bbox: () => void;  // 将当前 bbox 添加到 activeCategoryId
  setMode3CategoryInstances: (categoryId: string, instanceIds: number[]) => void;
}

let _nextCategoryId = 1;

export const useAppStore = create<AppState>((set, get) => ({
  // 初始状态
  currentMode: 'mode1',
  textPrompt: '',
  images: [],
  selectedImageId: null,
  viewingImageId: null,
  activeTool: 'select',
  bbox: null,
  bboxImageId: null,
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
  instanceMasksImageId: null,
  selectedInstanceIds: [],
  categories: [],
  activeCategoryId: null,
  mode2CategoryRefs: [],
  mode3CategoryRefs: [],
  stageScale: 1,
  stagePosition: { x: 0, y: 0 },

  // ========== Actions ==========

  setCurrentMode: (mode) =>
    set({
      currentMode: mode,
      bbox: null,
      bboxImageId: null,
      instanceMasks: [],
      instanceMasksImageId: null,
      selectedInstanceIds: [],
      isDrawing: false,
    }),

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
      if (s.bboxImageId === id) {
        updates.bbox = null;
        updates.bboxImageId = null;
      }
      if (s.instanceMasksImageId === id) {
        updates.instanceMasks = [];
        updates.instanceMasksImageId = null;
        updates.selectedInstanceIds = [];
      }
      return updates;
    }),

  selectImage: (id) =>
    set({ selectedImageId: id }),

  setViewingImage: (id) =>
    set({ viewingImageId: id }),

  setActiveTool: (tool) =>
    set({ activeTool: tool }),

  setBBox: (bbox) => {
    const state = get();
    const displayId = state.viewingImageId || state.selectedImageId;
    set({ bbox, bboxImageId: displayId });
  },

  setBBoxImageId: (id) =>
    set({ bboxImageId: id }),

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

  setInstanceMasks: (instances, imageId) => {
    const state = get();
    const imgId = imageId || state.viewingImageId || state.selectedImageId;
    set({
      instanceMasks: instances,
      instanceMasksImageId: imgId,
      selectedInstanceIds: [],
    });
  },

  toggleInstanceId: (id) =>
    set((s) => {
      const ids = s.selectedInstanceIds.includes(id)
        ? s.selectedInstanceIds.filter((i) => i !== id)
        : [...s.selectedInstanceIds, id];
      return { selectedInstanceIds: ids };
    }),

  setSelectedInstanceIds: (ids) =>
    set({ selectedInstanceIds: ids }),

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
      instanceMasksImageId: null,
      selectedInstanceIds: [],
      bbox: null,
      bboxImageId: null,
    }),

  // ===== 多类别管理 =====
  addCategory: (name) =>
    set((s) => {
      const id = `cat_${_nextCategoryId++}`;
      const colorIdx = s.categories.length % CATEGORY_COLORS.length;
      const newCat: CategoryItem = { id, name, color: CATEGORY_COLORS[colorIdx] };
      return {
        categories: [...s.categories, newCat],
        activeCategoryId: s.activeCategoryId || id,
      };
    }),

  removeCategory: (id) =>
    set((s) => {
      const categories = s.categories.filter((c) => c.id !== id);
      const activeCategoryId = s.activeCategoryId === id
        ? (categories[0]?.id || null)
        : s.activeCategoryId;
      return {
        categories,
        activeCategoryId,
        mode2CategoryRefs: s.mode2CategoryRefs.filter((r) => r.categoryId !== id),
        mode3CategoryRefs: s.mode3CategoryRefs.filter((r) => r.categoryId !== id),
      };
    }),

  setActiveCategoryId: (id) =>
    set({ activeCategoryId: id }),

  clearCategories: () =>
    set({
      categories: [],
      activeCategoryId: null,
      mode2CategoryRefs: [],
      mode3CategoryRefs: [],
    }),

  confirmMode2Bbox: () =>
    set((s) => {
      if (!s.bbox || !s.activeCategoryId) return {};
      const existing = s.mode2CategoryRefs.find((r) => r.categoryId === s.activeCategoryId);
      const bboxCopy: BBox = { ...s.bbox };
      let refs: Mode2CategoryRef[];
      if (existing) {
        refs = s.mode2CategoryRefs.map((r) =>
          r.categoryId === s.activeCategoryId
            ? { ...r, bboxes: [...r.bboxes, bboxCopy] }
            : r
        );
      } else {
        refs = [...s.mode2CategoryRefs, { categoryId: s.activeCategoryId!, bboxes: [bboxCopy] }];
      }
      return { mode2CategoryRefs: refs, bbox: null };
    }),

  setMode3CategoryInstances: (categoryId, instanceIds) =>
    set((s) => {
      const existing = s.mode3CategoryRefs.find((r) => r.categoryId === categoryId);
      let refs: Mode3CategoryRef[];
      if (existing) {
        refs = s.mode3CategoryRefs.map((r) =>
          r.categoryId === categoryId ? { ...r, instanceIds } : r
        );
      } else {
        refs = [...s.mode3CategoryRefs, { categoryId, instanceIds }];
      }
      return { mode3CategoryRefs: refs };
    }),
}));
