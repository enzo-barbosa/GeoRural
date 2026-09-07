import { useEffect, useState } from 'react'
import { useSearchParams, useNavigate } from 'react-router-dom'
import { ArrowLeft, Download, Image as ImageIcon, Layers, BarChart2, AlertCircle, CheckCircle, Brain } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Tabs, TabsList, TabsTrigger } from '@/components/ui/tabs'
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

interface ClassStats {
  class_name: string
  pixels: number
  percent: number
}

interface ClassificationResult {
  job_id: string
  status: 'completed'
  satellite: string
  classifier: string
  start_date: string
  end_date: string
  num_images: number
  stats: ClassStats[]
  classification_image_url: string
  rgb_image_url: string | null
  overlay_image_url: string | null
}

const CLASS_COLORS: Record<string, string> = {
  'Vegetação Nativa': '#1a5276',
  'Pastagem': '#27ae60',
  'Agricultura': '#2ecc71',
  'Solo Exposto': '#f39c12',
  'Área Construída': '#e74c3c',
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

    // In a real app, you'd fetch from an API endpoint
    // For now, we'll simulate loading
    fetch(`/api/classify/outputs/${jobId}/classification.png`)
      .then((res) => {
        if (!res.ok) throw new Error('Resultado não encontrado')
        return res.blob()
      })
      .then(() => {
        // Build result from localStorage or API
        const stored = localStorage.getItem(`result_${jobId}`)
        if (stored) {
          setResult(JSON.parse(stored))
        } else {
          setError('Resultado não encontrado no armazenamento local')
        }
        setLoading(false)
      })
      .catch(() => {
        setError('Erro ao carregar resultado')
        setLoading(false)
      })
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

  const handleDownloadCSV = () => {
    const headers = ['Classe', 'Pixels', 'Proporção (%)']
    const rows = result.stats.map((s) => [s.class_name, s.pixels.toString(), s.percent.toFixed(2)])
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

  const barColors = result.stats.map((s) => CLASS_COLORS[s.class_name] || '#888888')

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
                  <BarChart data={result.stats} layout="vertical">
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis type="number" domain={[0, 'dataMax + 10']} />
                    <YAxis dataKey="class_name" type="category" width={120} />
                    <Tooltip
                      formatter={(value: any) => [typeof value === 'number' ? value.toFixed(2) : '0.00', 'Proporção (%)']}
                      labelFormatter={(label) => label}
                    />
                    <Bar dataKey="percent" radius={[0, 4, 4, 0]}>
                      {result.stats.map((_, index) => (
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
                      <th className="text-right py-2 font-medium text-gray-500">Pixels</th>
                      <th className="text-right py-2 font-medium text-gray-500">Proporção (%)</th>
                    </tr>
                  </thead>
                  <tbody>
                    {result.stats.map((stat) => (
                      <tr key={stat.class_name} className="border-b border-gray-100">
                        <td className="py-3 flex items-center gap-2">
                          <div
                            className="w-3 h-3 rounded"
                            style={{ backgroundColor: CLASS_COLORS[stat.class_name] || '#888' }}
                          />
                          <span className="text-gray-900">{stat.class_name}</span>
                        </td>
                        <td className="py-3 text-right text-gray-600">{stat.pixels.toLocaleString()}</td>
                        <td className="py-3 text-right font-medium text-gray-900">{stat.percent.toFixed(2)}%</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </CardContent>
          </Card>
        </div>

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
                {result.stats
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