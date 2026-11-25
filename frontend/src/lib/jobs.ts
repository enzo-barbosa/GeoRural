import type { AnalysisStatus } from '@/types/api'

export const ACTIVE_STATUSES: AnalysisStatus[] = ['queued', 'running']

export const STAGE_LABELS: Record<string, string> = {
  queued: 'Na fila',
  starting: 'Iniciando',
  validating: 'Validando área',
  coverage: 'Detectando cobertura',
  searching_images: 'Buscando imagens',
  downloading: 'Baixando imagens',
  preprocessing: 'Pré-processando',
  features: 'Extraindo atributos',
  inference: 'Classificando pixels',
  postprocessing: 'Pós-processando',
  saving_layers: 'Gerando camadas',
  topography: 'Analisando relevo',
  crop: 'Estimando cultivos',
  finalizing: 'Finalizando',
  done: 'Concluída',
  failed: 'Falhou',
  interrupted: 'Interrompida',
}

export function jobStageLabel(stage: string | null): string {
  if (!stage) return 'Aguardando'
  return STAGE_LABELS[stage] ?? stage
}

export function isActiveStatus(status: AnalysisStatus): boolean {
  return ACTIVE_STATUSES.includes(status)
}
