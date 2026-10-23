import { useEffect, useState, type ComponentType } from 'react'
import { useSearchParams, useNavigate } from 'react-router-dom'
import {
  ArrowLeft,
  Download,
  Image as ImageIcon,
  Layers,
  BarChart2,
  AlertCircle,
  Brain,
  Ruler,
  Mountain,
  Sprout,
} from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Tabs, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { ResultMap } from '@/components/map/ResultMap'
import type { ClassificationResult } from '@/types/api'
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Cell,
} from 'recharts'

const CLASS_COLORS: Record<string, string> = {
  'Vegetação Nativa': '#1a5276',
  'Pastagem': '#27ae60',
  'Agricultura': '#2ecc71',
  'Solo Exposto': '#f39c12',
  'Área Construída': '#e74c3c',
}

const CROP_COLORS: Record<string, string> = {
  'Cana-de-açúcar': '#8e44ad',
  'Citros': '#f1c40f',
  'Pastagem': '#27ae60',
  'Soja': '#16a085',
  'Milho': '#e67e22',
}

function MeasurementCard({
  icon: Icon,
  label,
  value,
  accent,
}: {
  icon: ComponentType<{ className?: string }>
  label: string
  value: string
  accent: string
}) {
  return (
    <Card>
      <CardContent className="py-4">
        <div className="flex items-center gap-3">
          <div className={`p-2 rounded-lg ${accent}`}>
            <Icon className="h-5 w-5" />
          </div>
          <div>
            <p className="text-sm text-gray-500">{label}</p>
            <p className="font-semibold text-gray-900">{value}</p>
          </div>
        </div>
      </CardContent>
    </Card>
  )
}

export function Results() {
  const [searchParams] = useSearchParams()
  const navigate = useNavigate()
  const jobId = searchParams.get('job_id')
  const [result, setResult] = useState<ClassificationResult | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [activeTab, setActiveTab] = useState<'real' | 'classification' | 'overlay'>('classification')
  const [overlayAlpha, setOverlayAlpha] = useState(0.5)

  useEffect(() => {
    if (!jobId) {
      setError('Nenhum job_id fornecido')
      setLoading(false)
      return
    }

    const stored = localStorage.getItem(`result_${jobId}`)
    if (stored) {
      try {
        const parsed = JSON.parse(stored) as ClassificationResult
        if (!parsed || !parsed.stats || !parsed.classification_image_url) {
          throw new Error('Formato de resultado incompatível')
        }
        setResult(parsed)
      } catch {
        localStorage.removeItem(`result_${jobId}`)
        setError('Resultado antigo/incompatível. Faça a classificação novamente.')
      }
      setLoading(false)
      return
    }

    setError('Resultado não encontrado no armazenamento local')
    setLoading(false)
  }, [jobId])

  if (loading) {
    return (
      <div className="min-h-screen bg-gray-50 flex items-center justify-center">
        <div className="text-center">
          <div className="animate-spin rounded-full h-12 w-12 border-4 border-primary border-t-transparent mx-auto" />
          <p className="mt-4 text-gray-600">Carregando resultados...</p>
        </div>
      </div>
    )
  }

  if (error || !result) {
    return (
      <div className="min-h-screen bg-gray-50 flex items-center justify-center">
        <Card className="w-full max-w-md">
          <CardContent className="py-12 text-center">
            <AlertCircle className="h-12 w-12 text-red-500 mx-auto mb-4" />
            <h2 className="text-xl font-bold text-gray-900">Resultado não encontrado</h2>
            <p className="mt-2 text-gray-600">{error || 'Faça uma classificação primeiro'}</p>
            <Button onClick={() => navigate('/classify')} className="mt-6">
              <ArrowLeft className="mr-2 h-4 w-4" />
              Voltar para Classificar
            </Button>
          </CardContent>
        </Card>
      </div>
    )
  }

  const classAreas = result.class_areas ?? []
  const layers = result.layers ?? []
  const stats = result.stats ?? []

  const handleDownloadCSV = () => {
    const headers = ['Classe', 'Area (ha)', 'Pixels', 'Proporcao (%)', 'Talhoes']
    const rows = classAreas.map((s) => [
      s.class_name,
      s.area_ha.toFixed(4),
      s.pixels.toString(),
      s.percent.toFixed(2),
      s.parcels.toString(),
    ])
    const csv = [headers, ...rows].map((r) => r.join(',')).join('\n')
    const blob = new Blob([csv], { type: 'text/csv' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = `resultado_classificacao_${result.job_id}.csv`
    a.click()
    URL.revokeObjectURL(url)
  }

  const handleDownloadPNG = (imageUrl: string, filename: string) => {
    const a = document.createElement('a')
    a.href = imageUrl
    a.download = filename
    a.click()
  }

  const barColors = stats.map((s) => CLASS_COLORS[s.class_name] || '#888888')

  return (
    <div className="min-h-screen bg-gray-50">
      <header className="bg-white border-b border-gray-200">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center h-16">
            <div className="flex items-center gap-4">
              <Button variant="ghost" size="icon" onClick={() => navigate('/classify')}>
                <ArrowLeft className="h-5 w-5" />
              </Button>
              <div>
                <h1 className="text-xl font-bold text-gray-900">Resultados da Classificação</h1>
                <p className="text-sm text-gray-500">Job: {result.job_id}</p>
              </div>
            </div>
          </div>
        </div>
      </header>

      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-8">
          <Card>
            <CardContent className="py-4">
              <div className="flex items-center gap-3">
                <div className="p-2 bg-blue-100 rounded-lg">
                  <ImageIcon className="h-5 w-5 text-blue-600" />
                </div>
                <div>
                  <p className="text-sm text-gray-500">Satélite</p>
                  <p className="font-semibold text-gray-900">{result.satellite.toUpperCase()}</p>
                </div>
              </div>
            </CardContent>
          </Card>
          <Card>
            <CardContent className="py-4">
              <div className="flex items-center gap-3">
                <div className="p-2 bg-green-100 rounded-lg">
                  <Brain className="h-5 w-5 text-green-600" />
                </div>
                <div>
                  <p className="text-sm text-gray-500">Classificador</p>
                  <p className="font-semibold text-gray-900">
                    {result.classifier.replace('_', ' ').toUpperCase()}
                  </p>
                </div>
              </div>
            </CardContent>
          </Card>
          <Card>
            <CardContent className="py-4">
              <div className="flex items-center gap-3">
                <div className="p-2 bg-purple-100 rounded-lg">
                  <Layers className="h-5 w-5 text-purple-600" />
                </div>
                <div>
                  <p className="text-sm text-gray-500">Imagens</p>
                  <p className="font-semibold text-gray-900">{result.num_images}</p>
                </div>
              </div>
            </CardContent>
          </Card>
        </div>

        {result.measurements && (
          <div className="grid grid-cols-2 md:grid-cols-4 gap-6 mb-8">
            <MeasurementCard
              icon={Ruler}
              label="Área total"
              value={`${result.measurements.total_area_ha.toLocaleString('pt-BR')} ha`}
              accent="bg-emerald-100 text-emerald-600"
            />
            <MeasurementCard
              icon={Ruler}
              label="Perímetro"
              value={`${result.measurements.perimeter_m.toLocaleString('pt-BR')} m`}
              accent="bg-sky-100 text-sky-600"
            />
            <MeasurementCard
              icon={Layers}
              label="Talhões (por classe)"
              value={result.measurements.parcels.toLocaleString('pt-BR')}
              accent="bg-amber-100 text-amber-600"
            />
            <MeasurementCard
              icon={ImageIcon}
              label="Pixels analisados"
              value={result.measurements.total_pixels.toLocaleString('pt-BR')}
              accent="bg-slate-100 text-slate-600"
            />
          </div>
        )}

        {result.bounds && layers.length > 0 && (
          <Card className="mb-8">
            <CardHeader>
              <CardTitle>Sobreposição no Mapa</CardTitle>
            </CardHeader>
            <CardContent>
              <ResultMap
                bounds={result.bounds}
                rgbImageUrl={result.rgb_image_url}
                layers={layers}
              />
              <div className="mt-4 space-y-3">
                {layers.map((layer) => (
                  <div key={layer.name}>
                    <p className="text-sm font-medium text-gray-700 mb-1">{layer.label}</p>
                    <div className="flex flex-wrap gap-x-4 gap-y-1 text-xs text-gray-600">
                      {layer.legend.map((item) => (
                        <span key={item.label} className="flex items-center gap-1">
                          <span
                            className="inline-block w-3 h-3 rounded"
                            style={{ backgroundColor: item.color }}
                          />
                          {item.label}
                        </span>
                      ))}
                    </div>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        )}

        <Card className="mb-8">
          <CardHeader>
            <CardTitle>Mapa de Classificação</CardTitle>
          </CardHeader>
          <CardContent>
            <Tabs value={activeTab} onValueChange={(value: string) => setActiveTab(value as 'real' | 'classification' | 'overlay')} className="w-full">
              <TabsList className="grid w-full grid-cols-3 mb-4">
                <TabsTrigger value="real" disabled={!result.rgb_image_url}>
                  <ImageIcon className="mr-2 h-4 w-4" />
                  Imagem Real
                </TabsTrigger>
                <TabsTrigger value="classification">
                  <Layers className="mr-2 h-4 w-4" />
                  Classificação
                </TabsTrigger>
                <TabsTrigger value="overlay" disabled={!result.rgb_image_url}>
                  <BarChart2 className="mr-2 h-4 w-4" />
                  Sobreposta
                </TabsTrigger>
              </TabsList>

              <div className="relative aspect-square max-h-[600px] overflow-hidden rounded-lg border border-gray-200 bg-gray-100">
                {activeTab === 'real' && result.rgb_image_url && (
                  <img
                    src={result.rgb_image_url}
                    alt="Imagem real do satélite"
                    className="w-full h-full object-contain"
                  />
                )}
                {activeTab === 'classification' && (
                  <img
                    src={result.classification_image_url}
                    alt="Mapa temático de classificação"
                    className="w-full h-full object-contain"
                  />
                )}
                {activeTab === 'overlay' && result.overlay_image_url && result.rgb_image_url && (
                  <>
                    <img
                      src={result.rgb_image_url}
                      alt="Imagem real"
                      className="w-full h-full object-contain"
                    />
                    <img
                      src={result.overlay_image_url}
                      alt="Classificação sobreposta"
                      className="absolute inset-0 w-full h-full object-contain"
                      style={{ opacity: overlayAlpha }}
                    />
                    <div className="absolute bottom-4 left-4 right-4 bg-white/90 p-2 rounded-lg shadow-lg">
                      <label className="flex items-center gap-2 text-sm">
                        <span>Transparência:</span>
                        <input
                          type="range"
                          min="0"
                          max="1"
                          step="0.1"
                          value={overlayAlpha}
                          onChange={(e) => setOverlayAlpha(parseFloat(e.target.value))}
                          className="flex-1"
                        />
                        <span className="w-10 text-right">{Math.round(overlayAlpha * 100)}%</span>
                      </label>
                    </div>
                  </>
                )}
              </div>
            </Tabs>
          </CardContent>
        </Card>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
          <Card>
            <CardHeader>
              <CardTitle>Distribuição das Classes</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="h-[300px]">
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={stats} layout="vertical">
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis type="number" domain={[0, 'dataMax + 10']} />
                    <YAxis dataKey="class_name" type="category" width={120} />
                    <Tooltip
                      formatter={(value: unknown) => [typeof value === 'number' ? value.toFixed(2) : '0.00', 'Proporção (%)']}
                      labelFormatter={(label) => label}
                    />
                    <Bar dataKey="percent" radius={[0, 4, 4, 0]}>
                      {stats.map((_, index) => (
                        <Cell key={`cell-${index}`} fill={barColors[index]} />
                      ))}
                    </Bar>
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle>Detalhamento</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="overflow-x-auto">
                <table className="w-full text-sm">
                  <thead>
                    <tr className="border-b border-gray-200">
                      <th className="text-left py-2 font-medium text-gray-500">Classe</th>
                      <th className="text-right py-2 font-medium text-gray-500">Área (ha)</th>
                      <th className="text-right py-2 font-medium text-gray-500">Talhões</th>
                      <th className="text-right py-2 font-medium text-gray-500">%</th>
                    </tr>
                  </thead>
                  <tbody>
                    {classAreas.map((stat) => (
                      <tr key={stat.class_name} className="border-b border-gray-100">
                        <td className="py-3 flex items-center gap-2">
                          <div
                            className="w-3 h-3 rounded"
                            style={{ backgroundColor: CLASS_COLORS[stat.class_name] || '#888' }}
                          />
                          <span className="text-gray-900">{stat.class_name}</span>
                        </td>
                        <td className="py-3 text-right text-gray-600">
                          {stat.area_ha.toLocaleString('pt-BR', { maximumFractionDigits: 2 })}
                        </td>
                        <td className="py-3 text-right text-gray-600">{stat.parcels}</td>
                        <td className="py-3 text-right font-medium text-gray-900">
                          {stat.percent.toFixed(2)}%
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </CardContent>
          </Card>
        </div>

        {result.topography && (
          <Card className="mt-8">
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Mountain className="h-5 w-5 text-gray-700" />
                Análise Topográfica
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6">
                <div className="p-4 bg-gray-50 rounded-lg">
                  <p className="text-sm text-gray-500">Elevação mín.</p>
                  <p className="text-xl font-bold text-gray-900">
                    {result.topography.elevation_min?.toFixed(1) ?? '—'} m
                  </p>
                </div>
                <div className="p-4 bg-gray-50 rounded-lg">
                  <p className="text-sm text-gray-500">Elevação média</p>
                  <p className="text-xl font-bold text-gray-900">
                    {result.topography.elevation_mean?.toFixed(1) ?? '—'} m
                  </p>
                </div>
                <div className="p-4 bg-gray-50 rounded-lg">
                  <p className="text-sm text-gray-500">Elevação máx.</p>
                  <p className="text-xl font-bold text-gray-900">
                    {result.topography.elevation_max?.toFixed(1) ?? '—'} m
                  </p>
                </div>
                <div className="p-4 bg-gray-50 rounded-lg">
                  <p className="text-sm text-gray-500">Desnível</p>
                  <p className="text-xl font-bold text-gray-900">
                    {result.topography.relief_m?.toFixed(1) ?? '—'} m
                  </p>
                </div>
              </div>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <div>
                  <p className="text-sm text-gray-500 mb-2">
                    Declividade média: <strong>{result.topography.slope_mean?.toFixed(1) ?? '—'}°</strong>
                    {' · '}máx.: <strong>{result.topography.slope_max?.toFixed(1) ?? '—'}°</strong>
                  </p>
                  <p className="text-sm text-gray-500">
                    Relevo dominante: <strong>{result.topography.dominant_class ?? '—'}</strong>
                  </p>
                </div>
                <div className="overflow-x-auto">
                  <table className="w-full text-sm">
                    <thead>
                      <tr className="border-b border-gray-200">
                        <th className="text-left py-2 font-medium text-gray-500">Relevo</th>
                        <th className="text-right py-2 font-medium text-gray-500">Área (ha)</th>
                        <th className="text-right py-2 font-medium text-gray-500">%</th>
                      </tr>
                    </thead>
                    <tbody>
                      {result.topography.classes.map((item) => (
                        <tr key={item.class_name} className="border-b border-gray-100">
                          <td className="py-2 text-gray-900">{item.class_name}</td>
                          <td className="py-2 text-right text-gray-600">
                            {item.area_ha.toLocaleString('pt-BR', { maximumFractionDigits: 2 })}
                          </td>
                          <td className="py-2 text-right text-gray-600">{item.percent.toFixed(2)}%</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            </CardContent>
          </Card>
        )}

        {result.crop_stats && result.crop_stats.length > 0 && (
          <Card className="mt-8">
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Sprout className="h-5 w-5 text-green-600" />
                Tipo de Cultivo (estimado)
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
                <div className="h-[280px]">
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart data={result.crop_stats} layout="vertical">
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis type="number" domain={[0, 'dataMax + 10']} />
                      <YAxis dataKey="class_name" type="category" width={110} />
                      <Tooltip
                        formatter={(value: unknown) => [typeof value === 'number' ? value.toFixed(2) : '0.00', 'Proporção (%)']}
                      />
                      <Bar dataKey="percent" radius={[0, 4, 4, 0]}>
                        {result.crop_stats.map((item, index) => (
                          <Cell key={`crop-${index}`} fill={CROP_COLORS[item.class_name] || '#888'} />
                        ))}
                      </Bar>
                    </BarChart>
                  </ResponsiveContainer>
                </div>
                <div className="overflow-x-auto">
                  <table className="w-full text-sm">
                    <thead>
                      <tr className="border-b border-gray-200">
                        <th className="text-left py-2 font-medium text-gray-500">Cultura</th>
                        <th className="text-right py-2 font-medium text-gray-500">Área (ha)</th>
                        <th className="text-right py-2 font-medium text-gray-500">%</th>
                      </tr>
                    </thead>
                    <tbody>
                      {result.crop_stats.map((item) => (
                        <tr key={item.class_name} className="border-b border-gray-100">
                          <td className="py-2 flex items-center gap-2">
                            <span
                              className="inline-block w-3 h-3 rounded"
                              style={{ backgroundColor: CROP_COLORS[item.class_name] || '#888' }}
                            />
                            <span className="text-gray-900">{item.class_name}</span>
                          </td>
                          <td className="py-2 text-right text-gray-600">
                            {item.area_ha.toLocaleString('pt-BR', { maximumFractionDigits: 2 })}
                          </td>
                          <td className="py-2 text-right text-gray-600">{item.percent.toFixed(2)}%</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
              <p className="mt-4 text-xs text-gray-500">
                Estimativa baseada na série temporal NDVI/EVI (perfis fenológicos). Use como apoio à decisão.
              </p>
            </CardContent>
          </Card>
        )}

        <Card className="mt-8">
          <CardHeader>
            <CardTitle>Análise de Áreas Ociosas</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-3 text-gray-700">
              <p><strong>Áreas identificadas como potenciais para arrendamento:</strong></p>
              <ul className="list-disc list-inside space-y-1 ml-4">
                <li><strong>Solo Exposto:</strong> Áreas sem cobertura vegetal, podem indicar terras disponíveis</li>
                <li><strong>Pastagem degradada:</strong> Áreas com NDVI baixo dentro de pastagens</li>
              </ul>
              <div className="grid grid-cols-2 gap-4 mt-4">
                {stats
                  .filter((s) => ['Solo Exposto', 'Pastagem'].includes(s.class_name))
                  .map((stat) => (
                    <div key={stat.class_name} className="p-4 bg-gray-50 rounded-lg">
                      <p className="font-medium">{stat.class_name}</p>
                      <p className="text-2xl font-bold text-primary">{stat.percent.toFixed(1)}%</p>
                      <p className="text-sm text-gray-500">{stat.pixels.toLocaleString()} pixels</p>
                    </div>
                  ))}
              </div>
            </div>
          </CardContent>
        </Card>

        <Card className="mt-8">
          <CardHeader>
            <CardTitle>Exportar Resultados</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="flex flex-wrap gap-4">
              <Button onClick={handleDownloadCSV} className="gap-2">
                <Download className="h-4 w-4" />
                Baixar Relatório (CSV)
              </Button>
              <Button
                variant="outline"
                onClick={() => handleDownloadPNG(result.classification_image_url, `mapa_classificacao_${result.job_id}.png`)}
                className="gap-2"
              >
                <Download className="h-4 w-4" />
                Baixar Mapa (PNG)
              </Button>
              {result.rgb_image_url && (
                <Button
                  variant="outline"
                  onClick={() => handleDownloadPNG(result.rgb_image_url!, `imagem_real_${result.job_id}.png`)}
                  className="gap-2"
                >
                  <Download className="h-4 w-4" />
                  Baixar Imagem Real (PNG)
                </Button>
              )}
              {result.overlay_image_url && (
                <Button
                  variant="outline"
                  onClick={() => handleDownloadPNG(result.overlay_image_url!, `sobreposta_${result.job_id}.png`)}
                  className="gap-2"
                >
                  <Download className="h-4 w-4" />
                  Baixar Sobreposta (PNG)
                </Button>
              )}
            </div>
          </CardContent>
        </Card>
      </main>
    </div>
  )
}
