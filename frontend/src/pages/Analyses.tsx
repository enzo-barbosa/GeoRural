import { useState } from 'react'
import { Link, useNavigate } from 'react-router-dom'
import { AlertCircle, History, MapPin, RotateCcw } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardTitle } from '@/components/ui/card'
import { UserMenu } from '@/components/UserMenu'
import { useAnalyses, useRetryAnalysis } from '@/hooks/useApi'
import { formatArea, formatDateTime } from '@/lib/format'
import { isActiveStatus, jobStageLabel } from '@/lib/jobs'
import type { AnalysisStatus } from '@/types/api'

const STATUS_LABEL: Record<AnalysisStatus, string> = {
  queued: 'Na fila',
  running: 'Processando',
  completed: 'Concluída',
  failed: 'Falhou',
}

const STATUS_CLASS: Record<AnalysisStatus, string> = {
  queued: 'bg-gray-100 text-gray-700',
  running: 'bg-blue-100 text-blue-700',
  completed: 'bg-emerald-100 text-emerald-700',
  failed: 'bg-red-100 text-red-700',
}

export function Analyses() {
  const navigate = useNavigate()
  const { data: analyses = [], isLoading, isError } = useAnalyses()
  const retryMutation = useRetryAnalysis()
  const [actionError, setActionError] = useState<string | null>(null)

  const handleRetry = async (jobId: string) => {
    setActionError(null)
    try {
      const pending = await retryMutation.mutateAsync(jobId)
      navigate(`/results?job_id=${pending.job_id}`)
    } catch {
      setActionError('Não foi possível reprocessar a análise.')
    }
  }

  return (
    <div className="min-h-screen bg-gray-50">
      <header className="bg-white border-b border-gray-200">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center h-16">
            <Link to="/" className="flex items-center gap-2">
              <MapPin className="h-8 w-8 text-primary" />
              <span className="text-xl font-bold text-gray-900">GeoRural</span>
            </Link>
            <nav className="flex items-center gap-4">
              <Link to="/classify" className="text-gray-600 hover:text-gray-900">
                Classificar
              </Link>
              <Link to="/areas" className="text-gray-600 hover:text-gray-900">
                Minhas áreas
              </Link>
              <UserMenu />
            </nav>
          </div>
        </div>
      </header>

      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        <div className="mb-8">
          <h1 className="text-3xl font-bold text-gray-900">Histórico</h1>
          <p className="mt-2 text-gray-600">
            Todas as classificações feitas na sua conta, mais recentes primeiro.
          </p>
          {actionError && (
            <p className="mt-3 text-sm text-red-600">{actionError}</p>
          )}
        </div>

        {isLoading && (
          <div className="text-center py-16">
            <div className="animate-spin rounded-full h-10 w-10 border-4 border-primary border-t-transparent mx-auto" />
            <p className="mt-4 text-gray-600">Carregando histórico...</p>
          </div>
        )}

        {isError && (
          <Card>
            <CardContent className="py-12 text-center text-gray-600">
              Não foi possível carregar o histórico.
            </CardContent>
          </Card>
        )}

        {!isLoading && !isError && analyses.length === 0 && (
          <Card>
            <CardContent className="py-12 text-center">
              <History className="h-12 w-12 text-gray-300 mx-auto mb-4" />
              <h2 className="text-lg font-semibold text-gray-900">
                Nenhuma análise ainda
              </h2>
              <p className="mt-2 text-gray-600">
                Faça sua primeira classificação para vê-la aqui.
              </p>
              <Button onClick={() => navigate('/classify')} className="mt-6">
                Ir para Classificar
              </Button>
            </CardContent>
          </Card>
        )}

        <div className="space-y-4">
          {analyses.map((analysis) => {
            const title =
              analysis.name ?? analysis.area_name ?? `Análise ${analysis.id}`
            return (
              <Card key={analysis.id}>
                <CardContent className="py-4">
                  <div className="flex flex-col sm:flex-row gap-4 sm:items-center">
                    <div className="w-full sm:w-40 h-24 rounded-lg bg-gray-100 overflow-hidden flex-shrink-0">
                      {analysis.image_url ? (
                        <img
                          src={analysis.image_url}
                          alt={`Mapa da análise ${title}`}
                          className="w-full h-full object-cover"
                        />
                      ) : (
                        <div className="w-full h-full flex items-center justify-center text-gray-400 text-xs">
                          Sem imagem
                        </div>
                      )}
                    </div>

                    <div className="flex-1">
                      <div className="flex flex-wrap items-center gap-2">
                        <CardTitle className="text-lg">{title}</CardTitle>
                        <span
                          className={`text-xs px-2 py-0.5 rounded-full font-medium ${STATUS_CLASS[analysis.status]}`}
                        >
                          {STATUS_LABEL[analysis.status]}
                        </span>
                      </div>
                      <p className="mt-1 text-sm text-gray-500">
                        {formatDateTime(analysis.created_at)} ·{' '}
                        {analysis.satellite.toUpperCase()} ·{' '}
                        {analysis.classifier.replace('_', ' ')} ·{' '}
                        {formatArea(analysis.area_ha)}
                      </p>
                      {analysis.area_name && (
                        <p className="text-sm text-gray-600">
                          Área: {analysis.area_name}
                        </p>
                      )}
                      {isActiveStatus(analysis.status) && (
                        <div className="mt-2 max-w-sm">
                          <div className="flex justify-between text-xs text-gray-500">
                            <span>{jobStageLabel(analysis.stage)}</span>
                            <span>{analysis.progress}%</span>
                          </div>
                          <div className="mt-1 h-1.5 w-full overflow-hidden rounded-full bg-gray-200">
                            <div
                              className="h-full rounded-full bg-primary transition-all"
                              style={{
                                width: `${Math.max(0, Math.min(100, analysis.progress))}%`,
                              }}
                            />
                          </div>
                        </div>
                      )}
                      {analysis.status === 'failed' && analysis.error && (
                        <p className="mt-1 flex items-center gap-1 text-sm text-red-600">
                          <AlertCircle className="h-4 w-4" />
                          {analysis.error}
                        </p>
                      )}
                    </div>

                    <div className="flex flex-col gap-2">
                      <Button
                        variant="outline"
                        onClick={() =>
                          navigate(`/results?job_id=${analysis.id}`)
                        }
                      >
                        {isActiveStatus(analysis.status)
                          ? 'Acompanhar'
                          : 'Ver resultado'}
                      </Button>
                      {analysis.status === 'failed' && (
                        <Button
                          onClick={() => handleRetry(analysis.id)}
                          disabled={retryMutation.isPending}
                          className="gap-2"
                        >
                          <RotateCcw className="h-4 w-4" />
                          Reprocessar
                        </Button>
                      )}
                    </div>
                  </div>
                </CardContent>
              </Card>
            )
          })}
        </div>
      </main>
    </div>
  )
}
