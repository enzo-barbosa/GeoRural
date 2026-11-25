export interface PolygonGeometry {
  type: 'Polygon'
  coordinates: number[][][]
}

export type ClassifierName =
  | 'auto'
  | 'ensemble'
  | 'random_forest'
  | 'svm'
  | 'unet_cnn'
  | 'brazil_rf'
  | 'brazil_svm'
  | 'brazil_unet'
  | 'global_rf'
  | 'global_svm'
  | 'sp_rf'
  | 'sp_svm'
  | 'sp_ensemble'

export interface ClassifyRequest {
  polygon: PolygonGeometry
  satellite: 'sentinel2' | 'landsat8'
  classifier: ClassifierName
  start_date: string
  end_date: string
  max_cloud: number
  include_topography?: boolean
  include_crop?: boolean
  postprocess?: boolean
  area_id?: number | null
  name?: string | null
}

export interface CoverageRequest {
  polygon: PolygonGeometry
}

export interface CoverageResponse {
  coverage: 'brazil' | 'global'
  label_source: string
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
  dem_source: string | null
  dem_resolution_m: number | null
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
  coverage: 'brazil' | 'global'
  label_source: string | null
  models_used: string[]
  weights: Record<string, number>
  confidence_mean: number | null
  start_date: string
  end_date: string
  num_images: number
  stats: ClassStats[]
  measurements: MeasurementStats | null
  class_areas: ClassArea[]
  topography: TopographyStats | null
  crop_stats: ClassArea[] | null
  postprocessed?: boolean
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
  brazil_rf: boolean
  brazil_svm: boolean
  brazil_unet: boolean
  global_rf: boolean
  global_svm: boolean
  sp_rf: boolean
  sp_svm: boolean
  crop: boolean
}

export interface HealthResponse {
  status: 'healthy' | 'degraded'
  gee_initialized: boolean
}

export interface GeocodeResult {
  display_name: string
  lat: number
  lon: number
  /** [south, north, west, east] como no Nominatim */
  bbox?: [number, number, number, number] | null
}

export interface GeocodeResponse {
  results: GeocodeResult[]
}

export interface AuthUser {
  id: number
  email: string
  name: string
  is_active: boolean
  created_at: string
}

export interface LoginRequest {
  email: string
  password: string
}

export interface RegisterRequest extends LoginRequest {
  name: string
}

export interface Area {
  id: number
  name: string
  description: string | null
  geojson: PolygonGeometry
  area_ha: number | null
  created_at: string
  updated_at: string
}

export interface AreaCreateRequest {
  name: string
  description?: string | null
  polygon: PolygonGeometry
}

export interface AreaUpdateRequest {
  name?: string
  description?: string | null
}

export type AnalysisStatus = 'queued' | 'running' | 'completed' | 'failed'

export interface AnalysisSummary {
  id: string
  name: string | null
  status: AnalysisStatus
  stage: string | null
  progress: number
  error: string | null
  satellite: string
  classifier: string
  coverage: 'brazil' | 'global' | null
  start_date: string | null
  end_date: string | null
  area_id: number | null
  area_name: string | null
  area_ha: number | null
  num_images: number | null
  confidence_mean: number | null
  image_url: string | null
  created_at: string
  updated_at: string
}

export interface AnalysisDetail extends AnalysisSummary {
  label_source: string | null
  max_cloud: number | null
  include_topography: boolean
  include_crop: boolean
  postprocess: boolean
  models_used: string[] | null
  weights: Record<string, number> | null
  geojson: PolygonGeometry | null
  result: ClassificationResult | null
}
