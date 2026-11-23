import axios from 'axios'
import type {
  ClassifyRequest,
  ClassifyResponse,
  CoverageRequest,
  CoverageResponse,
  GeocodeResponse,
  GeocodeResult,
  ModelsStatusResponse,
  HealthResponse,
} from '@/types/api'

const api = axios.create({
  baseURL: '/api',
  headers: {
    'Content-Type': 'application/json',
  },
  timeout: 300000, // 5 minutes for classification
})

export const classifyApi = {
  classify: async (request: ClassifyRequest): Promise<ClassifyResponse> => {
    const response = await api.post<ClassifyResponse>('/classify', request)
    return response.data
  },

  getModelsStatus: async (): Promise<ModelsStatusResponse> => {
    const response = await api.get<ModelsStatusResponse>('/classify/models')
    return response.data
  },

  getCoverage: async (request: CoverageRequest): Promise<CoverageResponse> => {
    const response = await api.post<CoverageResponse>('/classify/coverage', request)
    return response.data
  },
}

export const healthApi = {
  check: async (): Promise<HealthResponse> => {
    const response = await api.get<HealthResponse>('/health')
    return response.data
  },
}

export const geocodeApi = {
  search: async (q: string, limit = 5): Promise<GeocodeResult[]> => {
    const response = await api.get<GeocodeResponse>('/geocode', {
      params: { q, limit },
    })
    return response.data.results
  },
}

export default api