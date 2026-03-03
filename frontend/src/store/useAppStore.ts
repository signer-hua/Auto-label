/**
 * Zustand 全局状态管理（v3 增强版）
 *
 * v3 新增：
 *   - 多参考图库：refImageIds + refImageWeights + refImageBboxes
 *   - 置信度评分：imageScores + 筛选/排序
 *   - 手动标注工具：manualTool + brushSize + 撤销/重做历史
 *   - 图层隔离强化
 */
import { create } from 'zustand';

export interface ImageItem {
  id: string;
  filename: string;
  url: string;
  path: string;
}

export interface BBox {
  x: number;
  y: number;
  width: number;
  height: number;
}

export interface InstanceItem {
  id: number;
  mask_url: string;
  bbox: number[];
  color: number[];
}

export interface CategoryItem {
  id: string;
  name: string;
  color: string;
}

export interface Mode2CategoryRef {
  categoryId: string;
  bboxes: BBox[];
}

export interface Mode3CategoryRef {
  categoryId: string;
  instanceIds: number[];
}

/** 单图置信度评分 */
export interface ImageScore {
  total: number;
  similarity: number;
  mask_coverage: number;
  area: number;
  detection: number;
  level: 'high' | 'medium' | 'low';
}

/** 参考图信息 */
export interface RefImageInfo {
  imageId: string;
  weight: number;
  bbox: BBox | null;
}

/** 手动标注操作记录（撤销/重做） */
export interface HistoryEntry {
  type: 'add_mask' | 'delete_mask';
  imageId: string;
  maskUrl: string;
}

export type ToolType = 'select' | 'pan' | 'zoom';
export type ManualTool = 'rect_manual' | 'brush' | 'eraser' | null;
export type AnnotationMode = 'mode1' | 'mode2' | 'mode3';
export type TaskStatus =
  | 'idle' | 'pending' | 'processing' | 'success' | 'failed'
  | 'paused' | 'canceled' | 'instance_ready';
export type ScoreFilter = 'all' | 'high' | 'medium' | 'low';
export type SortOrder = 'default' | 'score_asc' | 'score_desc';

const CATEGORY_COLORS = [
  '#FF5050', '#50C850', '#5078FF', '#FFC800',
  '#C850FF', '#00DCB4', '#FF7800', '#B4B400',
  '#6496FF', '#FF6464',
];

interface AppState {
  currentMode: AnnotationMode;
  textPrompt: string;

  images: ImageItem[];
  selectedImageId: string | null;
  viewingImageId: string | null;

  activeTool: ToolType;
  bbox: BBox | null;
  bboxImageId: string | null;
  isDrawing: boolean;

  taskId: string | null;
  taskStatus: TaskStatus;
  taskProgress: number;
  taskTotal: number;
  taskMessage: string;
  errorType: string | null;

  maskUrls: Record<string, string[]>;
  exportUrl: string | null;
  maskOpacity: number;

  discoveryTaskId: string | null;
  instanceMasks: InstanceItem[];
  instanceMasksImageId: string | null;
  selectedInstanceIds: number[];

  categories: CategoryItem[];
  activeCategoryId: string | null;
  mode2CategoryRefs: Mode2CategoryRef[];
  mode3CategoryRefs: Mode3CategoryRef[];

  stageScale: number;
  stagePosition: { x: number; y: number };

  // v3: 多参考图库
  refImages: RefImageInfo[];

  // v3: 置信度评分
  imageScores: Record<string, ImageScore>;
  scoreFilter: ScoreFilter;
  sortOrder: SortOrder;

  // v3: 手动标注工具
  manualTool: ManualTool;
  brushSize: number;

  // v3: 撤销/重做
  undoStack: HistoryEntry[];
  redoStack: HistoryEntry[];

  // Actions
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
  addCategory: (name: string) => void;
  removeCategory: (id: string) => void;
  setActiveCategoryId: (id: string | null) => void;
  clearCategories: () => void;
  confirmMode2Bbox: () => void;
  setMode3CategoryInstances: (categoryId: string, instanceIds: number[]) => void;
  // v3 actions
  addRefImage: (imageId: string) => void;
  removeRefImage: (imageId: string) => void;
  setRefImageWeight: (imageId: string, weight: number) => void;
  setRefImageBbox: (imageId: string, bbox: BBox | null) => void;
  clearRefImages: () => void;
  setImageScores: (scores: Record<string, ImageScore>) => void;
  setScoreFilter: (filter: ScoreFilter) => void;
  setSortOrder: (order: SortOrder) => void;
  setManualTool: (tool: ManualTool) => void;
  setBrushSize: (size: number) => void;
  addMaskToImage: (imageId: string, maskUrl: string) => void;
  removeMaskFromImage: (imageId: string, maskUrl: string) => void;
  clearImageMasks: (imageId: string) => void;
  undo: () => void;
  redo: () => void;
}

let _nextCategoryId = 1;

export const useAppStore = create<AppState>((set, get) => ({
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
  refImages: [],
  imageScores: {},
  scoreFilter: 'all',
  sortOrder: 'default',
  manualTool: null,
  brushSize: 10,
  undoStack: [],
  redoStack: [],

  setCurrentMode: (mode) =>
    set({
      currentMode: mode, bbox: null, bboxImageId: null,
      instanceMasks: [], instanceMasksImageId: null,
      selectedInstanceIds: [], isDrawing: false, manualTool: null,
    }),

  setTextPrompt: (text) => set({ textPrompt: text }),
  addImage: (img) => set((s) => ({ images: [...s.images, img] })),
  addImages: (imgs) => set((s) => ({ images: [...s.images, ...imgs] })),
  setImages: (imgs) => set({ images: imgs }),

  removeImage: (id) =>
    set((s) => {
      const images = s.images.filter((img) => img.id !== id);
      const updates: Partial<AppState> = { images };
      if (s.selectedImageId === id) updates.selectedImageId = images[0]?.id || null;
      if (s.viewingImageId === id) updates.viewingImageId = null;
      if (s.bboxImageId === id) { updates.bbox = null; updates.bboxImageId = null; }
      if (s.instanceMasksImageId === id) {
        updates.instanceMasks = [];
        updates.instanceMasksImageId = null;
        updates.selectedInstanceIds = [];
      }
      updates.refImages = s.refImages.filter((r) => r.imageId !== id);
      return updates;
    }),

  selectImage: (id) => set({ selectedImageId: id }),
  setViewingImage: (id) => set({ viewingImageId: id }),
  setActiveTool: (tool) => set({ activeTool: tool, manualTool: null }),

  setBBox: (bbox) => {
    const state = get();
    const displayId = state.viewingImageId || state.selectedImageId;
    set({ bbox, bboxImageId: displayId });
  },

  setBBoxImageId: (id) => set({ bboxImageId: id }),
  setIsDrawing: (drawing) => set({ isDrawing: drawing }),

  setTask: (taskId, status) =>
    set({ taskId, taskStatus: status, taskProgress: 0, taskTotal: 0, taskMessage: '', errorType: null }),

  updateTaskProgress: (progress, total, message) =>
    set({ taskProgress: progress, taskTotal: total, taskMessage: message }),

  setTaskStatus: (status, message) =>
    set((s) => ({ taskStatus: status, taskMessage: message ?? s.taskMessage })),

  setErrorType: (errorType) => set({ errorType }),
  setMaskUrls: (urls) => set({ maskUrls: urls }),
  setExportUrl: (url) => set({ exportUrl: url }),
  setMaskOpacity: (opacity) => set({ maskOpacity: opacity }),
  setDiscoveryTaskId: (id) => set({ discoveryTaskId: id }),

  setInstanceMasks: (instances, imageId) => {
    const state = get();
    const imgId = imageId || state.viewingImageId || state.selectedImageId;
    set({ instanceMasks: instances, instanceMasksImageId: imgId, selectedInstanceIds: [] });
  },

  toggleInstanceId: (id) =>
    set((s) => ({
      selectedInstanceIds: s.selectedInstanceIds.includes(id)
        ? s.selectedInstanceIds.filter((i) => i !== id)
        : [...s.selectedInstanceIds, id],
    })),

  setSelectedInstanceIds: (ids) => set({ selectedInstanceIds: ids }),
  setStageScale: (scale) => set({ stageScale: scale }),
  setStagePosition: (pos) => set({ stagePosition: pos }),

  resetTask: () =>
    set({
      taskId: null, taskStatus: 'idle', taskProgress: 0, taskTotal: 0,
      taskMessage: '', errorType: null, maskUrls: {}, exportUrl: null,
      discoveryTaskId: null, instanceMasks: [], instanceMasksImageId: null,
      selectedInstanceIds: [], bbox: null, bboxImageId: null,
      imageScores: {}, undoStack: [], redoStack: [],
    }),

  addCategory: (name) =>
    set((s) => {
      const id = `cat_${_nextCategoryId++}`;
      const newCat: CategoryItem = { id, name, color: CATEGORY_COLORS[s.categories.length % CATEGORY_COLORS.length] };
      return { categories: [...s.categories, newCat], activeCategoryId: s.activeCategoryId || id };
    }),

  removeCategory: (id) =>
    set((s) => {
      const categories = s.categories.filter((c) => c.id !== id);
      return {
        categories,
        activeCategoryId: s.activeCategoryId === id ? (categories[0]?.id || null) : s.activeCategoryId,
        mode2CategoryRefs: s.mode2CategoryRefs.filter((r) => r.categoryId !== id),
        mode3CategoryRefs: s.mode3CategoryRefs.filter((r) => r.categoryId !== id),
      };
    }),

  setActiveCategoryId: (id) => set({ activeCategoryId: id }),
  clearCategories: () => set({ categories: [], activeCategoryId: null, mode2CategoryRefs: [], mode3CategoryRefs: [] }),

  confirmMode2Bbox: () =>
    set((s) => {
      if (!s.bbox || !s.activeCategoryId) return {};
      const existing = s.mode2CategoryRefs.find((r) => r.categoryId === s.activeCategoryId);
      const bboxCopy: BBox = { ...s.bbox };
      let refs: Mode2CategoryRef[];
      if (existing) {
        refs = s.mode2CategoryRefs.map((r) =>
          r.categoryId === s.activeCategoryId ? { ...r, bboxes: [...r.bboxes, bboxCopy] } : r);
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
        refs = s.mode3CategoryRefs.map((r) => r.categoryId === categoryId ? { ...r, instanceIds } : r);
      } else {
        refs = [...s.mode3CategoryRefs, { categoryId, instanceIds }];
      }
      return { mode3CategoryRefs: refs };
    }),

  // ===== v3: 多参考图库 =====
  addRefImage: (imageId) =>
    set((s) => {
      if (s.refImages.length >= 5 || s.refImages.some(r => r.imageId === imageId)) return {};
      return { refImages: [...s.refImages, { imageId, weight: 1.0, bbox: null }] };
    }),

  removeRefImage: (imageId) =>
    set((s) => ({ refImages: s.refImages.filter((r) => r.imageId !== imageId) })),

  setRefImageWeight: (imageId, weight) =>
    set((s) => ({
      refImages: s.refImages.map((r) => r.imageId === imageId ? { ...r, weight } : r),
    })),

  setRefImageBbox: (imageId, bbox) =>
    set((s) => ({
      refImages: s.refImages.map((r) => r.imageId === imageId ? { ...r, bbox } : r),
    })),

  clearRefImages: () => set({ refImages: [] }),

  // ===== v3: 置信度评分 =====
  setImageScores: (scores) => set({ imageScores: scores }),
  setScoreFilter: (filter) => set({ scoreFilter: filter }),
  setSortOrder: (order) => set({ sortOrder: order }),

  // ===== v3: 手动标注工具 =====
  setManualTool: (tool) => set({ manualTool: tool, activeTool: 'select' }),
  setBrushSize: (size) => set({ brushSize: size }),

  addMaskToImage: (imageId, maskUrl) =>
    set((s) => {
      const existing = s.maskUrls[imageId] || [];
      return {
        maskUrls: { ...s.maskUrls, [imageId]: [...existing, maskUrl] },
        undoStack: [...s.undoStack.slice(-9), { type: 'add_mask', imageId, maskUrl }],
        redoStack: [],
      };
    }),

  removeMaskFromImage: (imageId, maskUrl) =>
    set((s) => {
      const existing = s.maskUrls[imageId] || [];
      return {
        maskUrls: { ...s.maskUrls, [imageId]: existing.filter((u) => u !== maskUrl) },
        undoStack: [...s.undoStack.slice(-9), { type: 'delete_mask', imageId, maskUrl }],
        redoStack: [],
      };
    }),

  clearImageMasks: (imageId) =>
    set((s) => ({
      maskUrls: { ...s.maskUrls, [imageId]: [] },
    })),

  undo: () =>
    set((s) => {
      if (s.undoStack.length === 0) return {};
      const entry = s.undoStack[s.undoStack.length - 1];
      const existing = s.maskUrls[entry.imageId] || [];
      let newMasks: string[];
      if (entry.type === 'add_mask') {
        newMasks = existing.filter((u) => u !== entry.maskUrl);
      } else {
        newMasks = [...existing, entry.maskUrl];
      }
      return {
        maskUrls: { ...s.maskUrls, [entry.imageId]: newMasks },
        undoStack: s.undoStack.slice(0, -1),
        redoStack: [...s.redoStack, entry],
      };
    }),

  redo: () =>
    set((s) => {
      if (s.redoStack.length === 0) return {};
      const entry = s.redoStack[s.redoStack.length - 1];
      const existing = s.maskUrls[entry.imageId] || [];
      let newMasks: string[];
      if (entry.type === 'add_mask') {
        newMasks = [...existing, entry.maskUrl];
      } else {
        newMasks = existing.filter((u) => u !== entry.maskUrl);
      }
      return {
        maskUrls: { ...s.maskUrls, [entry.imageId]: newMasks },
        redoStack: s.redoStack.slice(0, -1),
        undoStack: [...s.undoStack, entry],
      };
    }),
}));
