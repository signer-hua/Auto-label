/**
 * 类别-颜色映射工具类
 * 封装类别 ID 与 RGB 颜色的映射、查询、持久化逻辑。
 * 所有图层渲染统一调用此工具获取颜色，杜绝同类别颜色不一致。
 */

/** 预设颜色池（10 色，循环分配） */
const PRESET_COLORS = [
  '#FF5733', '#33FF57', '#3357FF', '#FFD700',
  '#FF33FF', '#00CED1', '#FF8C00', '#8A2BE2',
  '#00FF7F', '#DC143C',
];

const STORAGE_KEY = 'autolabel_categories';

export interface CategoryDef {
  id: string;
  name: string;
  color: string; // hex, 如 #FF5733
}

/** 从 localStorage 加载类别列表 */
export function loadCategoriesFromStorage(): CategoryDef[] {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    return raw ? JSON.parse(raw) : [];
  } catch {
    return [];
  }
}

/** 保存类别列表到 localStorage */
export function saveCategoriesToStorage(cats: CategoryDef[]): void {
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(cats));
  } catch { /* ignore */ }
}

/** 根据类别 ID 获取颜色（hex） */
export function getCategoryColor(categories: CategoryDef[], categoryId: string): string {
  const cat = categories.find(c => c.id === categoryId);
  return cat?.color || '#888888';
}

/** 根据类别 ID 获取颜色（RGB 元组） */
export function getCategoryColorRGB(categories: CategoryDef[], categoryId: string): [number, number, number] {
  const hex = getCategoryColor(categories, categoryId);
  return hexToRGB(hex);
}

/** hex → RGB 元组 */
export function hexToRGB(hex: string): [number, number, number] {
  const h = hex.replace('#', '');
  return [
    parseInt(h.substring(0, 2), 16),
    parseInt(h.substring(2, 4), 16),
    parseInt(h.substring(4, 6), 16),
  ];
}

/** RGB 元组 → hex */
export function rgbToHex(r: number, g: number, b: number): string {
  return `#${[r, g, b].map(v => v.toString(16).padStart(2, '0')).join('')}`.toUpperCase();
}

/** 自动分配一个未被使用的预设颜色 */
export function getNextAvailableColor(usedColors: string[]): string {
  const usedSet = new Set(usedColors.map(c => c.toUpperCase()));
  for (const c of PRESET_COLORS) {
    if (!usedSet.has(c.toUpperCase())) return c;
  }
  const r = Math.floor(Math.random() * 200) + 30;
  const g = Math.floor(Math.random() * 200) + 30;
  const b = Math.floor(Math.random() * 200) + 30;
  return rgbToHex(r, g, b);
}

/** 根据类别名称查找类别（模糊匹配） */
export function findCategoryByName(categories: CategoryDef[], name: string): CategoryDef | undefined {
  const lower = name.toLowerCase().trim();
  return categories.find(c => c.name.toLowerCase().trim() === lower);
}

/**
 * 确保自动/手动标注同类别颜色统一。
 * 根据类别名称或ID返回一致的颜色值。
 */
export function getUnifiedColor(
  categories: CategoryDef[],
  categoryId?: string,
  categoryName?: string,
): string {
  if (categoryId) {
    const cat = categories.find(c => c.id === categoryId);
    if (cat) return cat.color;
  }
  if (categoryName) {
    const cat = findCategoryByName(categories, categoryName);
    if (cat) return cat.color;
  }
  return '#888888';
}

/**
 * 同步类别列表到 localStorage 并返回更新后的列表。
 * 如果后端返回了新的类别名称，自动添加到本地类别列表。
 */
export function syncCategoriesFromAnnotation(
  existingCategories: CategoryDef[],
  annotationCategories: Array<{ name: string; color?: string }>,
): CategoryDef[] {
  const updated = [...existingCategories];
  let changed = false;

  for (const annCat of annotationCategories) {
    const existing = findCategoryByName(updated, annCat.name);
    if (!existing) {
      const usedColors = updated.map(c => c.color);
      const color = annCat.color || getNextAvailableColor(usedColors);
      updated.push({
        id: `cat_sync_${Date.now()}_${Math.random().toString(36).slice(2, 6)}`,
        name: annCat.name,
        color,
      });
      changed = true;
    }
  }

  if (changed) {
    saveCategoriesToStorage(updated);
  }
  return updated;
}
