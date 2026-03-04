/**
 * Zustand 全局状态管理（v4）
 *
 * v4 修复与增强：
 *   - 全局类别管理：categories 持久化到 localStorage，模式1/2/3 统一使用
 *   - 模式1 类别绑定：mode1CategoryId 绑定标注结果至指定全局类别
 *   - 模式2 框选残留修复：Mode2CategoryRef 增加 imageId 字段，confirmedBboxes 按图片过滤
 *   - clearBboxAfterAnnotate：标注提交后自动清空框选状态
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

/** 全局类别（跨模式共享，持久化到 localStorage） */
export interface CategoryItem {
  id: string;
  name: string;
  color: string;
}

/** 模式2 类别-框选绑定（含 imageId 防跨图残留） */
export interface Mode2CategoryRef {
  categoryId: string;
  imageId: string;
  bboxes: BBox[];
}

export interface Mode3CategoryRef {
  categoryId: string;
  instanceIds: number[];
}

export interface ImageScore {
  total: number;
  similarity: number;
  mask_coverage: number;
  area: number;
  detection: number;
  level: 'high' | 'medium' | 'low';
}

export interface RefImageInfo {
  imageId: string;
  weight: number;
  bbox: BBox | null;
}

export interface HistoryEntry {
  type: 'add_mask' | 'delete_mask';
  imageId: string;
  maskUrl: string;
}

export type ToolType = 'select' | 'pan' | 'zoom';
export type ManualTool = 'rect_manual' | 'negative_box' | 'brush' | 'eraser' | null;
export type AnnotationMode = 'mode1' | 'mode2' | 'mode3';
export type TaskStatus =
  | 'idle' | 'pending' | 'processing' | 'success' | 'failed'
  | 'paused' | 'canceled' | 'instance_ready';
export type ScoreFilter = 'all' | 'high' | 'medium' | 'low';
export type SortOrder = 'default' | 'score_asc' | 'score_desc';

/** 预设类别颜色 */
const PRESET_COLORS = [
  '#FF5050', '#50C850', '#5078FF', '#FFC800',
  '#C850FF', '#00DCB4', '#FF7800', '#B4B400',
  '#6496FF', '#FF6464',
];

/** 从 localStorage 加载持久化类别 */
function loadCategories(): CategoryItem[] {
  try {
    const raw = localStorage.getItem('autolabel_categories');
    if (raw) return JSON.parse(raw);
  } catch { /* ignore */ }
  return [];
}

/** 保存类别到 localStorage */
function saveCategories(cats: CategoryItem[]) {
  try { localStorage.setItem('autolabel_categories', JSON.stringify(cats)); } catch { /* ignore */ }
}

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
  mode1CategoryId: string | null;  // 模式1 绑定的全局类别
  mode2CategoryRefs: Mode2CategoryRef[];
  mode3CategoryRefs: Mode3CategoryRef[];
  stageScale: number;
  stagePosition: { x: number; y: number };
  // 图片自适应渲染参数（切图时自动重算）
  imageFitScale: number;
  imageFitOffsetX: number;
  imageFitOffsetY: number;
  refImages: RefImageInfo[];
  imageScores: Record<string, ImageScore>;
  scoreFilter: ScoreFilter;
  sortOrder: SortOrder;
  manualTool: ManualTool;
  brushSize: number;
  undoStack: HistoryEntry[];
  redoStack: HistoryEntry[];
  negativeBoxes: BBox[];
  mode2TextHint: string;

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
  // 全局类别管理（持久化）
  addCategory: (name: string, color?: string) => void;
  updateCategory: (id: string, name: string, color: string) => void;
  removeCategory: (id: string) => void;
  setActiveCategoryId: (id: string | null) => void;
  clearCategories: () => void;
  setMode1CategoryId: (id: string | null) => void;
  // 模式2 框选管理（含 imageId 绑定）
  confirmMode2Bbox: () => void;
  clearBboxAfterAnnotate: () => void;
  setMode3CategoryInstances: (categoryId: string, instanceIds: number[]) => void;
  setImageFit: (scale: number, offsetX: number, offsetY: number) => void;
  // 多参考图
  addRefImage: (imageId: string) => void;
  removeRefImage: (imageId: string) => void;
  setRefImageWeight: (imageId: string, weight: number) => void;
  setRefImageBbox: (imageId: string, bbox: BBox | null) => void;
  clearRefImages: () => void;
  // 评分
  setImageScores: (scores: Record<string, ImageScore>) => void;
  setScoreFilter: (filter: ScoreFilter) => void;
  setSortOrder: (order: SortOrder) => void;
  // 手动标注
  setManualTool: (tool: ManualTool) => void;
  setBrushSize: (size: number) => void;
  addMaskToImage: (imageId: string, maskUrl: string) => void;
  removeMaskFromImage: (imageId: string, maskUrl: string) => void;
  clearImageMasks: (imageId: string) => void;
  undo: () => void;
  redo: () => void;
  addNegativeBox: (box: BBox) => void;
  clearNegativeBoxes: () => void;
  setMode2TextHint: (text: string) => void;
}

let _nextCatId = Date.now();

const initialCategories = loadCategories();

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
  categories: initialCategories,
  activeCategoryId: initialCategories[0]?.id || null,
  mode1CategoryId: null,
  mode2CategoryRefs: [],
  mode3CategoryRefs: [],
  stageScale: 1,
  stagePosition: { x: 0, y: 0 },
  imageFitScale: 1,
  imageFitOffsetX: 0,
  imageFitOffsetY: 0,
  refImages: [],
  imageScores: {},
  scoreFilter: 'all',
  sortOrder: 'default',
  manualTool: null,
  brushSize: 10,
  undoStack: [],
  redoStack: [],
  negativeBoxes: [],
  mode2TextHint: '',

  // ===== 模式切换：清空所有模式特有状态 =====
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
        updates.instanceMasks = []; updates.instanceMasksImageId = null; updates.selectedInstanceIds = [];
      }
      updates.refImages = s.refImages.filter((r) => r.imageId !== id);
      updates.mode2CategoryRefs = s.mode2CategoryRefs.filter((r) => r.imageId !== id);
      return updates;
    }),

  selectImage: (id) => set({ selectedImageId: id }),
  setViewingImage: (id) => set({ viewingImageId: id }),
  setActiveTool: (tool) => set({ activeTool: tool, manualTool: null }),

  setBBox: (bbox) => {
    const s = get();
    set({ bbox, bboxImageId: s.viewingImageId || s.selectedImageId });
  },
  setBBoxImageId: (id) => set({ bboxImageId: id }),
  setIsDrawing: (drawing) => set({ isDrawing: drawing }),

  setTask: (taskId, status) =>
    set({ taskId, taskStatus: status, taskProgress: 0, taskTotal: 0, taskMessage: '', errorType: null }),
  updateTaskProgress: (progress, total, message) => set({ taskProgress: progress, taskTotal: total, taskMessage: message }),
  setTaskStatus: (status, message) => set((s) => ({ taskStatus: status, taskMessage: message ?? s.taskMessage })),
  setErrorType: (errorType) => set({ errorType }),
  setMaskUrls: (urls) => set({ maskUrls: urls }),
  setExportUrl: (url) => set({ exportUrl: url }),
  setMaskOpacity: (opacity) => set({ maskOpacity: opacity }),
  setDiscoveryTaskId: (id) => set({ discoveryTaskId: id }),

  setInstanceMasks: (instances, imageId) => {
    const s = get();
    set({ instanceMasks: instances, instanceMasksImageId: imageId || s.viewingImageId || s.selectedImageId, selectedInstanceIds: [] });
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
      mode2CategoryRefs: [], mode3CategoryRefs: [],
    }),

  // ===== 全局类别管理（持久化） =====
  addCategory: (name, color) =>
    set((s) => {
      const id = `cat_${_nextCatId++}`;
      const c = color || PRESET_COLORS[s.categories.length % PRESET_COLORS.length];
      const newCat: CategoryItem = { id, name, color: c };
      const cats = [...s.categories, newCat];
      saveCategories(cats);
      return { categories: cats, activeCategoryId: s.activeCategoryId || id };
    }),

  updateCategory: (id, name, color) =>
    set((s) => {
      const cats = s.categories.map((c) => c.id === id ? { ...c, name, color } : c);
      saveCategories(cats);
      return { categories: cats };
    }),

  removeCategory: (id) =>
    set((s) => {
      const cats = s.categories.filter((c) => c.id !== id);
      saveCategories(cats);
      return {
        categories: cats,
        activeCategoryId: s.activeCategoryId === id ? (cats[0]?.id || null) : s.activeCategoryId,
        mode1CategoryId: s.mode1CategoryId === id ? null : s.mode1CategoryId,
        mode2CategoryRefs: s.mode2CategoryRefs.filter((r) => r.categoryId !== id),
        mode3CategoryRefs: s.mode3CategoryRefs.filter((r) => r.categoryId !== id),
      };
    }),

  setActiveCategoryId: (id) => set({ activeCategoryId: id }),

  clearCategories: () => {
    saveCategories([]);
    set({ categories: [], activeCategoryId: null, mode1CategoryId: null, mode2CategoryRefs: [], mode3CategoryRefs: [] });
  },

  setMode1CategoryId: (id) => set({ mode1CategoryId: id }),

  // ===== 模式2 框选管理（含 imageId 绑定，防跨图残留） =====
  // 存储时将画布坐标转为原图坐标，避免渲染/提交时坐标偏移
  confirmMode2Bbox: () =>
    set((s) => {
      if (!s.bbox || !s.activeCategoryId) return {};
      const displayId = s.viewingImageId || s.selectedImageId || '';
      const sc = s.imageFitScale || 1;
      const ox = s.imageFitOffsetX || 0;
      const oy = s.imageFitOffsetY || 0;
      const imgBbox: BBox = {
        x: (s.bbox.x - ox) / sc,
        y: (s.bbox.y - oy) / sc,
        width: s.bbox.width / sc,
        height: s.bbox.height / sc,
      };
      const existing = s.mode2CategoryRefs.find(
        (r) => r.categoryId === s.activeCategoryId && r.imageId === displayId
      );
      let refs: Mode2CategoryRef[];
      if (existing) {
        refs = s.mode2CategoryRefs.map((r) =>
          r.categoryId === s.activeCategoryId && r.imageId === displayId
            ? { ...r, bboxes: [...r.bboxes, imgBbox] }
            : r
        );
      } else {
        refs = [...s.mode2CategoryRefs, { categoryId: s.activeCategoryId!, imageId: displayId, bboxes: [imgBbox] }];
      }
      return { mode2CategoryRefs: refs, bbox: null };
    }),

  /** 标注提交后立即清空框选状态（解决框选残留） */
  clearBboxAfterAnnotate: () =>
    set({ bbox: null, bboxImageId: null, isDrawing: false }),

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

  setImageFit: (scale, offsetX, offsetY) =>
    set({ imageFitScale: scale, imageFitOffsetX: offsetX, imageFitOffsetY: offsetY }),

  // ===== 多参考图 =====
  addRefImage: (imageId) =>
    set((s) => {
      if (s.refImages.length >= 5 || s.refImages.some(r => r.imageId === imageId)) return {};
      return { refImages: [...s.refImages, { imageId, weight: 1.0, bbox: null }] };
    }),
  removeRefImage: (imageId) => set((s) => ({ refImages: s.refImages.filter((r) => r.imageId !== imageId) })),
  setRefImageWeight: (imageId, weight) =>
    set((s) => ({ refImages: s.refImages.map((r) => r.imageId === imageId ? { ...r, weight } : r) })),
  setRefImageBbox: (imageId, bbox) =>
    set((s) => ({ refImages: s.refImages.map((r) => r.imageId === imageId ? { ...r, bbox } : r) })),
  clearRefImages: () => set({ refImages: [] }),

  // ===== 评分 =====
  setImageScores: (scores) => set({ imageScores: scores }),
  setScoreFilter: (filter) => set({ scoreFilter: filter }),
  setSortOrder: (order) => set({ sortOrder: order }),

  // ===== 手动标注 =====
  setManualTool: (tool) => set({ manualTool: tool, activeTool: 'select' }),
  setBrushSize: (size) => set({ brushSize: size }),
  addMaskToImage: (imageId, maskUrl) =>
    set((s) => ({
      maskUrls: { ...s.maskUrls, [imageId]: [...(s.maskUrls[imageId] || []), maskUrl] },
      undoStack: [...s.undoStack.slice(-9), { type: 'add_mask', imageId, maskUrl }],
      redoStack: [],
    })),
  removeMaskFromImage: (imageId, maskUrl) =>
    set((s) => ({
      maskUrls: { ...s.maskUrls, [imageId]: (s.maskUrls[imageId] || []).filter((u) => u !== maskUrl) },
      undoStack: [...s.undoStack.slice(-9), { type: 'delete_mask', imageId, maskUrl }],
      redoStack: [],
    })),
  clearImageMasks: (imageId) => set((s) => ({ maskUrls: { ...s.maskUrls, [imageId]: [] } })),
  undo: () =>
    set((s) => {
      if (s.undoStack.length === 0) return {};
      const entry = s.undoStack[s.undoStack.length - 1];
      const existing = s.maskUrls[entry.imageId] || [];
      const newMasks = entry.type === 'add_mask'
        ? existing.filter((u) => u !== entry.maskUrl)
        : [...existing, entry.maskUrl];
      return { maskUrls: { ...s.maskUrls, [entry.imageId]: newMasks }, undoStack: s.undoStack.slice(0, -1), redoStack: [...s.redoStack, entry] };
    }),
  redo: () =>
    set((s) => {
      if (s.redoStack.length === 0) return {};
      const entry = s.redoStack[s.redoStack.length - 1];
      const existing = s.maskUrls[entry.imageId] || [];
      const newMasks = entry.type === 'add_mask'
        ? [...existing, entry.maskUrl]
        : existing.filter((u) => u !== entry.maskUrl);
      return { maskUrls: { ...s.maskUrls, [entry.imageId]: newMasks }, redoStack: s.redoStack.slice(0, -1), undoStack: [...s.undoStack, entry] };
    }),
  addNegativeBox: (box) => set((s) => ({ negativeBoxes: [...s.negativeBoxes, box] })),
  clearNegativeBoxes: () => set({ negativeBoxes: [] }),
  setMode2TextHint: (text) => set({ mode2TextHint: text }),
}));
