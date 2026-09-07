import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { classifyApi, healthApi } from '@/services/api'
import type { ClassifyRequest, ClassifyResponse, ModelsStatusResponse } from '@/types/api'

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
    },
  })
}

export function useClassificationResult(jobId: string | null) {
  return useQuery({
    queryKey: ['classificationResult', jobId],
    queryFn: async () => {
      if (!jobId) return null
      // Poll for result - the classify endpoint returns immediately with result
      // In a production app, you'd have a separate endpoint to poll for async jobs
      return null
    },
    enabled: !!jobId,
    refetchInterval: 2000,
  })
}