import { useCallback, useEffect, useState } from 'react'
import { useNavigate, useSearchParams } from 'react-router-dom'
import {
  MapPin,
  AlertCircle,
  CheckCircle,
  Loader2,
  AlertTriangle,
  Satellite,
  Globe2,
  Save,
} from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Label } from '@/components/ui/label'
import { Select, type SelectOption } from '@/components/ui/select'
import { UserMenu } from '@/components/UserMenu'
import { MapView, type MapFocus } from '@/components/map/MapView'
import {
  LocationSearch,
  type LocationSelection,
} from '@/components/map/LocationSearch'
import {
  useArea,
  useAreas,
  useClassify,
  useCreateArea,
  useModelsStatus,
} from '@/hooks/useApi'
import { classifyApi } from '@/services/api'
import { apiErrorMessage } from '@/lib/errors'
import type { ClassifierName, ClassifyRequest } from '@/types/api'

const CLASSIFIER_OPTIONS = [
  { value: 'auto', label: 'Automático (recomendado)' },
  { value: 'ensemble', label: 'Ensemble (RF + SVM + U-Net)' },
  { value: 'random_forest', label: 'Random Forest (domínio)' },
  { value: 'svm', label: 'SVM (domínio)' },
  { value: 'unet_cnn', label: 'U-Net (Brasil, Sentinel-2)' },
  { value: 'sp_ensemble', label: 'Regional SP (experimental)' },
] as const

const CLASS_LEGEND = [
  { name: 'Vegetação Nativa', color: '#1a5276' },
  { name: 'Pastagem', color: '#0E6B3D' },
  { name: 'Agricultura', color: '#2ecc71' },
  { name: 'Solo Exposto', color: '#f39c12' },
  { name: 'Área Construída', color: '#e74c3c' },
  { name: 'Água', color: '#2e86de' },
] as const

export function Classify() {
  const navigate = useNavigate()
  const [searchParams, setSearchParams] = useSearchParams()
  const areaParam = searchParams.get('area_id')
  const requestedAreaId = areaParam && Number.isFinite(Number(areaParam))
    ? Number(areaParam)
    : null

  const [polygon, setPolygon] = useState<number[][][] | null>(null)
  const [focus, setFocus] = useState<MapFocus | null>(null)
  const [marker, setMarker] = useState<[number, number] | null>(null)
  const [satellite, setSatellite] = useState<'sentinel2' | 'landsat8'>('sentinel2')
  const [classifier, setClassifier] = useState<ClassifierName>('auto')
  const [coverage, setCoverage] = useState<'brazil' | 'global' | null>(null)
  const [startDate, setStartDate] = useState('2023-01-01')
  const [endDate, setEndDate] = useState('2023-12-31')
  const [maxCloud, setMaxCloud] = useState(80)
  const [includeTopography, setIncludeTopography] = useState(true)
  const [includeCrop, setIncludeCrop] = useState(true)
  const [postprocess, setPostprocess] = useState(true)
  const [error, setError] = useState<string | null>(null)

  const [activeAreaId, setActiveAreaId] = useState<number | null>(null)
  const [activeAreaName, setActiveAreaName] = useState<string | null>(null)
  const [areaNameInput, setAreaNameInput] = useState('')
  const [areaMessage, setAreaMessage] = useState<string | null>(null)
  const [analysisName, setAnalysisName] = useState('')
  const [mapKey, setMapKey] = useState(0)

  const { data: modelsStatus } = useModelsStatus()
  const { data: areas = [] } = useAreas()
  const { data: requestedArea } = useArea(requestedAreaId)
  const classifyMutation = useClassify()
  const createArea = useCreateArea()

  useEffect(() => {
    if (!requestedArea) return
    // oxlint-disable-next-line react/set-state-in-effect
    setPolygon(requestedArea.geojson.coordinates)
    // oxlint-disable-next-line react/set-state-in-effect
    setActiveAreaId(requestedArea.id)
    // oxlint-disable-next-line react/set-state-in-effect
    setActiveAreaName(requestedArea.name)
    // oxlint-disable-next-line react/set-state-in-effect
    setMapKey((key) => key + 1)
  }, [requestedArea])

  const updateCoverage = useCallback(async (coordinates: number[][][]) => {
    setCoverage(null)
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

  const handlePolygonDraw = useCallback(
    async (coordinates: number[][][]) => {
      setPolygon(coordinates)
      setActiveAreaId(null)
      setActiveAreaName(null)
      setAreaMessage(null)
      setError(null)
      setSearchParams({}, { replace: true })
      await updateCoverage(coordinates)
    },
    [setSearchParams, updateCoverage]
  )

  const handlePolygonRemove = useCallback(() => {
    setPolygon(null)
    setCoverage(null)
    setIncludeCrop(true)
    setActiveAreaId(null)
    setActiveAreaName(null)
    setSearchParams({}, { replace: true })
  }, [setSearchParams])

  const handleLoadArea = useCallback(
    (value: string) => {
      const area = areas.find((item) => String(item.id) === value)
      if (!area) return

      setPolygon(area.geojson.coordinates)
      setActiveAreaId(area.id)
      setActiveAreaName(area.name)
      setAreaMessage(null)
      setError(null)
      setMapKey((key) => key + 1)
      void updateCoverage(area.geojson.coordinates)
    },
    [areas, updateCoverage]
  )

  const handleUnlinkArea = useCallback(() => {
    setActiveAreaId(null)
    setActiveAreaName(null)
    setAreaMessage(null)
    setSearchParams({}, { replace: true })
  }, [setSearchParams])

  const handleSaveArea = async () => {
    if (!polygon) return
    const name = areaNameInput.trim()
    if (!name) {
      setAreaMessage('Informe um nome para salvar a área.')
      return
    }

    setAreaMessage(null)
    try {
      const area = await createArea.mutateAsync({
        name,
        polygon: { type: 'Polygon', coordinates: polygon },
      })
      setActiveAreaId(area.id)
      setActiveAreaName(area.name)
      setAreaNameInput('')
      setAreaMessage('Área salva em Minhas áreas.')
      setSearchParams({ area_id: String(area.id) }, { replace: true })
    } catch (err) {
      setAreaMessage(
        apiErrorMessage(err, 'Não foi possível salvar a área.')
      )
    }
  }

  const handleLocationSelect = useCallback(
    (selection: LocationSelection) => {
      setFocus({
        lat: selection.lat,
        lon: selection.lon,
        bbox: selection.bbox ?? null,
        zoom: selection.bbox ? undefined : 14,
        id: Date.now(),
      })
      setMarker([selection.lat, selection.lon])
      // Ao buscar um novo local, começa uma área nova.
      handlePolygonRemove()
    },
    [handlePolygonRemove]
  )

  const hasModels =
    modelsStatus?.brazil_rf ||
    modelsStatus?.brazil_svm ||
    modelsStatus?.brazil_unet ||
    modelsStatus?.global_rf ||
    modelsStatus?.global_svm ||
    modelsStatus?.sp_rf ||
    modelsStatus?.sp_svm

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
      postprocess,
      area_id: activeAreaId,
      name: analysisName.trim() || null,
    }

    try {
      const result = await classifyMutation.mutateAsync(request)

      if (result.status === 'failed') {
        setError(result.error)
        return
      }

      navigate(`/results?job_id=${result.job_id}`)
    } catch (err) {
      setError(apiErrorMessage(err, 'Erro ao classificar'))
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
              <a href="/areas" className="text-gray-600 hover:text-gray-900">Minhas áreas</a>
              <a href="/analyses" className="text-gray-600 hover:text-gray-900">Histórico</a>
              <UserMenu />
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
                <LocationSearch onSelect={handleLocationSelect} />
                <MapView
                  key={mapKey}
                  onDraw={handlePolygonDraw}
                  onRemove={handlePolygonRemove}
                  initialPolygon={polygon}
                  disabled={classifyMutation.isPending}
                  focus={focus}
                  marker={marker}
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

                <div className="mt-4 border-t pt-4 space-y-3">
                  <div>
                    <Label>Área salva</Label>
                    <Select
                      value={activeAreaId ? String(activeAreaId) : ''}
                      onValueChange={handleLoadArea}
                      options={areas.map((area) => ({
                        value: String(area.id),
                        label: area.area_ha
                          ? `${area.name} (${area.area_ha.toLocaleString('pt-BR', { maximumFractionDigits: 1 })} ha)`
                          : area.name,
                      }))}
                      placeholder={
                        areas.length > 0
                          ? 'Carregar uma área salva'
                          : 'Nenhuma área salva ainda'
                      }
                      disabled={areas.length === 0}
                      className="mt-2"
                    />
                  </div>

                  {activeAreaName ? (
                    <div className="flex items-center justify-between gap-2 text-sm bg-emerald-50 text-emerald-800 p-3 rounded-lg">
                      <span>
                        Área selecionada: <strong>{activeAreaName}</strong>
                      </span>
                      <Button
                        type="button"
                        variant="ghost"
                        size="sm"
                        onClick={handleUnlinkArea}
                      >
                        Desvincular
                      </Button>
                    </div>
                  ) : polygon ? (
                    <div className="flex gap-2">
                      <input
                        value={areaNameInput}
                        onChange={(event) => setAreaNameInput(event.target.value)}
                        placeholder="Nome da área (ex.: Fazenda Boa Vista)"
                        className="w-full h-10 rounded-md border border-gray-300 px-3 text-sm focus:outline-none focus:ring-2 focus:ring-primary/40"
                      />
                      <Button
                        type="button"
                        variant="outline"
                        onClick={() => {
                          void handleSaveArea()
                        }}
                        disabled={createArea.isPending}
                        className="gap-1.5 whitespace-nowrap"
                      >
                        <Save className="h-4 w-4" />
                        {createArea.isPending ? 'Salvando...' : 'Salvar área'}
                      </Button>
                    </div>
                  ) : null}

                  {areaMessage && (
                    <p className="text-xs text-gray-600">{areaMessage}</p>
                  )}
                </div>
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
                          const spDisabled =
                            opt.value === 'sp_ensemble' &&
                            !(modelsStatus?.sp_rf && modelsStatus?.sp_svm)
                          const disabled = unetDisabled || spDisabled
                          return {
                            value: opt.value,
                            label: disabled
                              ? `${opt.label} (indisponível)`
                              : opt.label,
                            disabled,
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
                    <label className="flex items-center gap-2 text-sm text-gray-700">
                      <input
                        type="checkbox"
                        checked={postprocess}
                        onChange={(e) => setPostprocess(e.target.checked)}
                        className="h-4 w-4 accent-primary"
                      />
                      Suavizar classificação (maioria 3×3)
                    </label>
                    <p className="text-xs text-gray-500">
                      Reduz ruído pixel a pixel e melhora a leitura visual.
                    </p>
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

                  <div className="border-t pt-6">
                    <Label htmlFor="analysis_name">Nome da análise (opcional)</Label>
                    <input
                      id="analysis_name"
                      type="text"
                      value={analysisName}
                      onChange={(event) => setAnalysisName(event.target.value)}
                      placeholder="Ex.: Safra 2023 — Fazenda Boa Vista"
                      className="mt-2 w-full rounded-md border border-gray-300 px-3 py-2 text-sm focus:border-primary focus:outline-none focus:ring-1 focus:ring-primary"
                    />
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