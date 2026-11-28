import axios from 'axios'
import type {
  AnalysisDetail,
  AnalysisSummary,
  Area,
  AreaCreateRequest,
  AreaUpdateRequest,
  AuthUser,
  ClassificationPending,
  ClassifyRequest,
  ClassifyResponse,
  CoverageRequest,
  CoverageResponse,
  GeocodeResponse,
  GeocodeResult,
  LoginRequest,
  ModelsStatusResponse,
  HealthResponse,
  RegisterRequest,
} from '@/types/api'

const api = axios.create({
  baseURL: '/api',
  headers: {
    'Content-Type': 'application/json',
  },
  withCredentials: true,
  timeout: 300000, // 5 minutes for classification
})

api.interceptors.response.use(
  (response) => response,
  (error: unknown) => {
    if (axios.isAxiosError(error) && error.response?.status === 401) {
      const url = error.config?.url ?? ''
      if (!url.startsWith('/auth/')) {
        window.dispatchEvent(new Event('auth:unauthorized'))
      }
    }
    return Promise.reject(error)
  }
)

export const authApi = {
  register: async (request: RegisterRequest): Promise<AuthUser> => {
    const response = await api.post<AuthUser>('/auth/register', request)
    return response.data
  },

  login: async (request: LoginRequest): Promise<AuthUser> => {
    const response = await api.post<AuthUser>('/auth/login', request)
    return response.data
  },

  logout: async (): Promise<void> => {
    await api.post('/auth/logout')
  },

  me: async (): Promise<AuthUser> => {
    const response = await api.get<AuthUser>('/auth/me')
    return response.data
  },
}

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

export const areasApi = {
  list: async (): Promise<Area[]> => {
    const response = await api.get<Area[]>('/areas')
    return response.data
  },

  get: async (areaId: number): Promise<Area> => {
    const response = await api.get<Area>(`/areas/${areaId}`)
    return response.data
  },

  create: async (request: AreaCreateRequest): Promise<Area> => {
    const response = await api.post<Area>('/areas', request)
    return response.data
  },

  update: async (areaId: number, request: AreaUpdateRequest): Promise<Area> => {
    const response = await api.patch<Area>(`/areas/${areaId}`, request)
    return response.data
  },

  remove: async (areaId: number): Promise<void> => {
    await api.delete(`/areas/${areaId}`)
  },
}

export const analysesApi = {
  list: async (limit = 20, offset = 0): Promise<AnalysisSummary[]> => {
    const response = await api.get<AnalysisSummary[]>('/analyses', {
      params: { limit, offset },
    })
    return response.data
  },

  get: async (jobId: string): Promise<AnalysisDetail> => {
    const response = await api.get<AnalysisDetail>(`/analyses/${jobId}`)
    return response.data
  },

  report: async (jobId: string): Promise<Blob> => {
    const response = await api.get<Blob>(`/analyses/${jobId}/report.pdf`, {
      responseType: 'blob',
    })
    return response.data
  },

  retry: async (jobId: string): Promise<ClassificationPending> => {
    const response = await api.post<ClassificationPending>(
      `/analyses/${jobId}/retry`
    )
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