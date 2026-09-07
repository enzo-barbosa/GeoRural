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
}

export interface ClassStats {
  class_name: string
  pixels: number
  percent: number
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
  classification_image_url: string
  rgb_image_url: string | null
  overlay_image_url: string | null
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
}

export interface HealthResponse {
  status: 'healthy' | 'degraded'
  gee_initialized: boolean
}