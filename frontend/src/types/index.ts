/**
 * 类型定义
 */

/** 已上传图像信息 */
export interface ImageInfo {
  name: string;
  path: string;
  size: number;
}

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

/** 单张图像的标注结果 */
export interface ImageResult {
  image: string;
  annotations: Annotation[];
  count: number;
}

/** 模式1 响应 */
export interface Mode1Response {
  mode: string;
  results: ImageResult[];
  total_annotations: number;
  export_path: string;
  export_format: string;
}

/** 模式2 响应 */
export interface Mode2Response {
  mode: string;
  ref_annotations: Annotation[];
  target_results: ImageResult[];
  total_annotations: number;
  template_features: number[][];
  export_path: string;
  export_format: string;
}

/** 模式3 聚类响应 */
export interface Mode3ClusterResponse {
  mode: string;
  image: string;
  clusters: ClusterData[];
  cluster_count: number;
  image_size: number[];
}

/** 模式3 标注响应 */
export interface Mode3AnnotateResponse {
  mode: string;
  results: ImageResult[];
  total_annotations: number;
  export_path: string;
  export_format: string;
}

/** 边界框 */
export interface Box {
  x1: number;
  y1: number;
  x2: number;
  y2: number;
}
