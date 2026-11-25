import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import {
  analysesApi,
  areasApi,
  classifyApi,
  geocodeApi,
  healthApi,
} from '@/services/api'
import type {
  AreaCreateRequest,
  AreaUpdateRequest,
  ClassifyRequest,
} from '@/types/api'

export function useHealth() {
  return useQuery({
    queryKey: ['health'],
    queryFn: healthApi.check,
    refetchInterval: 30000,
    retry: 3,
  })
}

export function useModelsStatus() {
  return useQuery({
    queryKey: ['modelsStatus'],
    queryFn: classifyApi.getModelsStatus,
    refetchInterval: 60000,
  })
}

export function useClassify() {
  const queryClient = useQueryClient()

  return useMutation({
    mutationFn: (request: ClassifyRequest) => classifyApi.classify(request),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['modelsStatus'] })
      queryClient.invalidateQueries({ queryKey: ['analyses'] })
    },
  })
}

export function useAnalysis(jobId: string | null) {
  return useQuery({
    queryKey: ['analysis', jobId],
    queryFn: () => analysesApi.get(jobId as string),
    enabled: !!jobId,
    refetchInterval: (query) => {
      const status = query.state.data?.status
      return status === 'queued' || status === 'running' ? 2000 : false
    },
  })
}

export function useAnalyses(limit = 20) {
  return useQuery({
    queryKey: ['analyses', limit],
    queryFn: () => analysesApi.list(limit),
    refetchInterval: (query) => {
      const analyses = query.state.data
      const hasActive = analyses?.some(
        (analysis) =>
          analysis.status === 'queued' || analysis.status === 'running'
      )
      return hasActive ? 5000 : false
    },
  })
}

export function useRetryAnalysis() {
  const queryClient = useQueryClient()

  return useMutation({
    mutationFn: (jobId: string) => analysesApi.retry(jobId),
    onSuccess: (data) => {
      queryClient.invalidateQueries({ queryKey: ['analyses'] })
      queryClient.invalidateQueries({ queryKey: ['analysis', data.job_id] })
    },
  })
}

export function useAreas() {
  return useQuery({
    queryKey: ['areas'],
    queryFn: areasApi.list,
  })
}

export function useArea(areaId: number | null) {
  return useQuery({
    queryKey: ['area', areaId],
    queryFn: () => areasApi.get(areaId as number),
    enabled: areaId != null,
  })
}

export function useCreateArea() {
  const queryClient = useQueryClient()

  return useMutation({
    mutationFn: (request: AreaCreateRequest) => areasApi.create(request),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['areas'] })
    },
  })
}

export function useUpdateArea() {
  const queryClient = useQueryClient()

  return useMutation({
    mutationFn: ({
      areaId,
      request,
    }: {
      areaId: number
      request: AreaUpdateRequest
    }) => areasApi.update(areaId, request),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['areas'] })
    },
  })
}

export function useDeleteArea() {
  const queryClient = useQueryClient()

  return useMutation({
    mutationFn: (areaId: number) => areasApi.remove(areaId),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['areas'] })
    },
  })
}

export function useGeocode(query: string) {
  const normalized = query.trim()
  return useQuery({
    queryKey: ['geocode', normalized],
    queryFn: () => geocodeApi.search(normalized),
    enabled: normalized.length >= 3,
    staleTime: 5 * 60 * 1000,
    retry: 0,
  })
}
