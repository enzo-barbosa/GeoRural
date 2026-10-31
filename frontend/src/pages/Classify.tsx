import { useCallback, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import {
  MapPin,
  AlertCircle,
  CheckCircle,
  Loader2,
  AlertTriangle,
  Satellite,
  Globe2,
} from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Label } from '@/components/ui/label'
import { Select, type SelectOption } from '@/components/ui/select'
import { MapView } from '@/components/map/MapView'
import { useClassify, useModelsStatus } from '@/hooks/useApi'
import { classifyApi } from '@/services/api'
import type { ClassifierName, ClassifyRequest } from '@/types/api'

const CLASSIFIER_OPTIONS = [
  { value: 'auto', label: 'Automático (recomendado)' },
  { value: 'ensemble', label: 'Ensemble (RF + SVM + U-Net)' },
  { value: 'random_forest', label: 'Random Forest (domínio)' },
  { value: 'svm', label: 'SVM (domínio)' },
  { value: 'unet_cnn', label: 'U-Net (Brasil, Sentinel-2)' },
] as const

const CLASS_LEGEND = [
  { name: 'Vegetação Nativa', color: '#1a5276' },
  { name: 'Pastagem', color: '#27ae60' },
  { name: 'Agricultura', color: '#2ecc71' },
  { name: 'Solo Exposto', color: '#f39c12' },
  { name: 'Área Construída', color: '#e74c3c' },
  { name: 'Água', color: '#2e86de' },
] as const

export function Classify() {
  const navigate = useNavigate()
  const [polygon, setPolygon] = useState<number[][][] | null>(null)
  const [satellite, setSatellite] = useState<'sentinel2' | 'landsat8'>('sentinel2')
  const [classifier, setClassifier] = useState<ClassifierName>('auto')
  const [coverage, setCoverage] = useState<'brazil' | 'global' | null>(null)
  const [startDate, setStartDate] = useState('2023-01-01')
  const [endDate, setEndDate] = useState('2023-12-31')
  const [maxCloud, setMaxCloud] = useState(80)
  const [includeTopography, setIncludeTopography] = useState(true)
  const [includeCrop, setIncludeCrop] = useState(true)
  const [error, setError] = useState<string | null>(null)

  const { data: modelsStatus } = useModelsStatus()
  const classifyMutation = useClassify()

  const handlePolygonDraw = useCallback(async (coordinates: number[][][]) => {
    setPolygon(coordinates)
    setCoverage(null)
    setError(null)

    try {
      const result = await classifyApi.getCoverage({
        polygon: { type: 'Polygon', coordinates },
      })
      setCoverage(result.coverage)
      if (result.coverage === 'global') {
        setIncludeCrop(false)
      }
    } catch {
      // A cobertura é apenas informativa; a classificação continua disponível.
    }
  }, [])

  const handlePolygonRemove = useCallback(() => {
    setPolygon(null)
    setCoverage(null)
    setIncludeCrop(true)
  }, [])

  const hasModels =
    modelsStatus?.brazil_rf ||
    modelsStatus?.brazil_svm ||
    modelsStatus?.brazil_unet ||
    modelsStatus?.global_rf ||
    modelsStatus?.global_svm

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    setError(null)

    if (!polygon) {
      setError('Desenhe uma área no mapa')
      return
    }

    const request: ClassifyRequest = {
      polygon: { type: 'Polygon', coordinates: polygon },
      satellite,
      classifier,
      start_date: startDate,
      end_date: endDate,
      max_cloud: maxCloud,
      include_topography: includeTopography,
      include_crop: coverage === 'global' ? false : includeCrop,
    }

    try {
      const result = await classifyMutation.mutateAsync(request)

      if (result.status === 'failed') {
        setError(result.error)
        return
      }

      localStorage.setItem(`result_${result.job_id}`, JSON.stringify(result))
      navigate(`/results?job_id=${result.job_id}`)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Erro ao classificar')
    }
  }

  return (
    <div className="min-h-screen bg-gray-50">
      <header className="bg-white border-b border-gray-200">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center h-16">
            <div className="flex items-center gap-2">
              <MapPin className="h-8 w-8 text-primary" />
              <span className="text-xl font-bold text-gray-900">GeoRural</span>
            </div>
            <nav className="flex items-center gap-4">
              <a href="/" className="text-gray-600 hover:text-gray-900">Início</a>
              <span className="text-gray-900 font-medium">Classificar</span>
              <a href="/results" className="text-gray-600 hover:text-gray-900">Resultados</a>
            </nav>
          </div>
        </div>
      </header>

      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        <div className="mb-8">
          <h1 className="text-3xl font-bold text-gray-900">Classificação de Uso do Solo</h1>
          <p className="mt-2 text-gray-600">Desenhe uma área no mapa e selecione as configurações para classificar</p>
        </div>

        {error && (
          <div className="mb-6 flex items-center gap-2 p-4 bg-red-50 border border-red-200 rounded-lg text-red-700">
            <AlertCircle className="h-5 w-5 flex-shrink-0" />
            <p>{error}</p>
          </div>
        )}

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
          <div className="lg:col-span-1">
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <MapPin className="h-5 w-5" />
                  1. Defina a Área de Interesse
                </CardTitle>
              </CardHeader>
              <CardContent>
                <MapView
                  onDraw={handlePolygonDraw}
                  onRemove={handlePolygonRemove}
                  initialPolygon={polygon}
                  disabled={classifyMutation.isPending}
                />
                {polygon ? (
                  <div className="mt-4 flex items-center gap-2 text-green-700 bg-green-50 p-3 rounded-lg">
                    <CheckCircle className="h-5 w-5" />
                    <span>Área definida! Ajuste as configurações ao lado.</span>
                  </div>
                ) : (
                  <div className="mt-4 flex items-center gap-2 text-yellow-700 bg-yellow-50 p-3 rounded-lg">
                    <AlertTriangle className="h-5 w-5" />
                    <span>Desenhe uma área no mapa usando a ferramenta de polígono</span>
                  </div>
                )}
                {coverage && (
                  <div className="mt-3 flex items-center gap-2 text-sm text-blue-700 bg-blue-50 p-3 rounded-lg">
                    <Globe2 className="h-4 w-4" />
                    <span>
                      Cobertura: {coverage === 'brazil'
                        ? 'Brasil (MapBiomas, detalhado)'
                        : 'Global (ESA WorldCover)'}
                    </span>
                  </div>
                )}
              </CardContent>
            </Card>
          </div>

          <div className="lg:col-span-1">
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Satellite className="h-5 w-5" />
                  2. Configurações
                </CardTitle>
              </CardHeader>
              <CardContent>
                <form onSubmit={handleSubmit} className="space-y-6">
                  <div>
                    <Label htmlFor="satellite">Satélite</Label>
                    <Select
                      value={satellite}
                      onValueChange={(v: string) => setSatellite(v as 'sentinel2' | 'landsat8')}
                      options={[
                        { value: 'sentinel2', label: 'Sentinel-2 (10m)' },
                        { value: 'landsat8', label: 'Landsat 8 (30m)' },
                      ] as SelectOption[]}
                      placeholder="Selecione o satélite"
                      className="mt-2"
                    />
                  </div>

                  <details className="rounded-lg border border-gray-200 p-3">
                    <summary className="cursor-pointer text-sm font-medium text-gray-700">
                      Avançado — escolher modelo
                    </summary>
                    <p className="mt-1 text-xs text-gray-500">
                      O padrão escolhe automaticamente o especialista do Brasil
                      ou a base global.
                    </p>
                    <div className="mt-3">
                      <Label htmlFor="classifier">Classificador</Label>
                      <Select
                        value={classifier}
                        onValueChange={(v: string) =>
                          setClassifier(v as ClassifierName)
                        }
                        options={CLASSIFIER_OPTIONS.map((opt) => {
                          const unetDisabled =
                            opt.value === 'unet_cnn' &&
                            (!modelsStatus?.brazil_unet ||
                              satellite !== 'sentinel2')
                          return {
                            value: opt.value,
                            label: unetDisabled
                              ? `${opt.label} (indisponível)`
                              : opt.label,
                            disabled: unetDisabled,
                          }
                        })}
                        placeholder="Selecione o classificador"
                        className="mt-2"
                      />
                    </div>
                    {!hasModels && (
                      <p className="mt-2 text-sm text-yellow-600">
                        Nenhum modelo treinado encontrado. Treine antes de
                        classificar.
                      </p>
                    )}
                    {coverage === 'global' && (
                      <p className="mt-2 text-xs text-gray-500">
                        Fora do Brasil não há tipo de cultivo nem U-Net; usa-se
                        RF/SVM globais.
                      </p>
                    )}
                  </details>

                  <div className="grid grid-cols-2 gap-4">
                    <div>
                      <Label htmlFor="start_date">Data Inicial</Label>
                      <input
                        id="start_date"
                        type="date"
                        value={startDate}
                        onChange={(e) => setStartDate(e.target.value)}
                        className="mt-2 w-full rounded-md border border-gray-300 px-3 py-2 text-sm focus:border-primary focus:outline-none focus:ring-1 focus:ring-primary"
                      />
                    </div>
                    <div>
                      <Label htmlFor="end_date">Data Final</Label>
                      <input
                        id="end_date"
                        type="date"
                        value={endDate}
                        onChange={(e) => setEndDate(e.target.value)}
                        className="mt-2 w-full rounded-md border border-gray-300 px-3 py-2 text-sm focus:border-primary focus:outline-none focus:ring-1 focus:ring-primary"
                      />
                    </div>
                  </div>

<div>
                      <Label htmlFor="max_cloud">Máximo de Cobertura de Nuvens (%): {maxCloud}</Label>
                      <input
                        id="max_cloud"
                        type="range"
                        min={0}
                        max={100}
                        step={5}
                        value={maxCloud}
                        onChange={(e) => setMaxCloud(Number(e.target.value))}
                        className="mt-2 w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer accent-primary"
                      />
                    </div>

                  <div className="border-t pt-4 space-y-3">
                    <Label className="font-medium">Análises adicionais</Label>
                    <label className="flex items-center gap-2 text-sm text-gray-700">
                      <input
                        type="checkbox"
                        checked={includeTopography}
                        onChange={(e) => setIncludeTopography(e.target.checked)}
                        className="h-4 w-4 accent-primary"
                      />
                      Análise topográfica (elevação, declividade)
                    </label>
                    <label
                      className={`flex items-center gap-2 text-sm ${
                        coverage === 'global'
                          ? 'text-gray-400'
                          : 'text-gray-700'
                      }`}
                    >
                      <input
                        type="checkbox"
                        checked={includeCrop && coverage !== 'global'}
                        onChange={(e) => setIncludeCrop(e.target.checked)}
                        disabled={coverage === 'global'}
                        className="h-4 w-4 accent-primary"
                      />
                      Tipo de cultivo (série NDVI/EVI)
                    </label>
                    {coverage === 'global' && (
                      <p className="text-xs text-gray-500">
                        Disponível apenas para áreas no Brasil.
                      </p>
                    )}
                  </div>

                  <div className="border-t pt-6">
                    <Label className="font-medium">Legenda das Classes</Label>
                    <div className="mt-3 space-y-2">
                      {CLASS_LEGEND.map((cls) => (
                        <div key={cls.name} className="flex items-center gap-3">
                          <div
                            className="w-4 h-4 rounded"
                            style={{ backgroundColor: cls.color }}
                          />
                          <span className="text-sm text-gray-700">{cls.name}</span>
                        </div>
                      ))}
                    </div>
                  </div>

                  <Button
                    type="submit"
                    className="w-full"
                    size="lg"
                    disabled={!polygon || classifyMutation.isPending}
                  >
                    {classifyMutation.isPending ? (
                      <>
                        <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                        Classificando...
                      </>
                    ) : (
                      'Classificar'
                    )}
                  </Button>
                </form>
              </CardContent>
            </Card>
          </div>
        </div>
      </main>
    </div>
  )
}