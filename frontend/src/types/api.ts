export interface PolygonGeometry {
  type: 'Polygon'
  coordinates: number[][][]
}

export interface ClassifyRequest {
  polygon: PolygonGeometry
  satellite: 'sentinel2' | 'landsat8'
  classifier: 'random_forest' | 'svm' | 'unet_cnn'
  start_date: string
  end_date: string
  max_cloud: number
  include_topography?: boolean
  include_crop?: boolean
}

export interface ClassStats {
  class_name: string
  pixels: number
  percent: number
}

export interface MeasurementStats {
  total_area_ha: number
  perimeter_m: number
  total_pixels: number
  parcels: number
}

export interface ClassArea {
  class_name: string
  area_ha: number
  pixels: number
  percent: number
  parcels: number
}

export interface SlopeClassArea {
  class_name: string
  area_ha: number
  percent: number
}

export interface TopographyStats {
  elevation_min: number | null
  elevation_max: number | null
  elevation_mean: number | null
  relief_m: number | null
  slope_min: number | null
  slope_mean: number | null
  slope_max: number | null
  dominant_class: string | null
  classes: SlopeClassArea[]
}

export interface LayerLegendItem {
  label: string
  color: string
}

export interface LayerInfo {
  name: string
  label: string
  image_url: string
  legend: LayerLegendItem[]
}

export interface ClassificationResult {
  job_id: string
  status: 'completed'
  satellite: string
  classifier: string
  start_date: string
  end_date: string
  num_images: number
  stats: ClassStats[]
  measurements: MeasurementStats | null
  class_areas: ClassArea[]
  topography: TopographyStats | null
  crop_stats: ClassArea[] | null
  layers: LayerInfo[]
  classification_image_url: string
  rgb_image_url: string | null
  overlay_image_url: string | null
  bounds: [[number, number], [number, number]] | null
}

export interface ClassificationError {
  job_id: string
  status: 'failed'
  error: string
}

export interface ClassificationPending {
  job_id: string
  status: 'pending' | 'processing'
}

export type ClassifyResponse = ClassificationResult | ClassificationError | ClassificationPending

export interface ModelsStatusResponse {
  random_forest: boolean
  svm: boolean
  unet_cnn: boolean
  crop: boolean
}

export interface HealthResponse {
  status: 'healthy' | 'degraded'
  gee_initialized: boolean
}
