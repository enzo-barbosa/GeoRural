import axios from 'axios'
import type {
  AuthUser,
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