import { useState } from 'react'
import { Link, useNavigate } from 'react-router-dom'
import {
  AlertCircle,
  Map as MapIcon,
  MapPin,
  Pencil,
  Plus,
  Trash2,
} from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { UserMenu } from '@/components/UserMenu'
import { useAreas, useDeleteArea, useUpdateArea } from '@/hooks/useApi'
import { apiErrorMessage } from '@/lib/errors'
import { formatArea, formatDateTime } from '@/lib/format'

export function Areas() {
  const navigate = useNavigate()
  const { data: areas = [], isLoading, isError } = useAreas()
  const updateArea = useUpdateArea()
  const deleteArea = useDeleteArea()
  const [error, setError] = useState<string | null>(null)

  const handleRename = (areaId: number, currentName: string) => {
    const name = window.prompt('Novo nome da área', currentName)?.trim()
    if (!name || name === currentName) return
    updateArea.mutate(
      { areaId, request: { name } },
      {
        onError: (err) =>
          setError(apiErrorMessage(err, 'Não foi possível renomear a área.')),
      }
    )
  }

  const handleDelete = (areaId: number, name: string) => {
    const confirmed = window.confirm(
      `Excluir a área "${name}"? O histórico de análises é preservado.`
    )
    if (!confirmed) return
    deleteArea.mutate(areaId, {
      onError: (err) =>
        setError(apiErrorMessage(err, 'Não foi possível excluir a área.')),
    })
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
              <Link to="/analyses" className="text-gray-600 hover:text-gray-900">
                Histórico
              </Link>
              <UserMenu />
            </nav>
          </div>
        </div>
      </header>

      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        <div className="flex flex-wrap items-center justify-between gap-4 mb-8">
          <div>
            <h1 className="text-3xl font-bold text-gray-900">Minhas áreas</h1>
            <p className="mt-2 text-gray-600">
              Salve polígonos para reutilizar nas próximas classificações.
            </p>
          </div>
          <Link to="/classify">
            <Button className="gap-2">
              <Plus className="h-4 w-4" />
              Nova área
            </Button>
          </Link>
        </div>

        {error && (
          <div className="mb-6 flex items-center gap-2 p-4 bg-red-50 border border-red-200 rounded-lg text-red-700">
            <AlertCircle className="h-5 w-5 flex-shrink-0" />
            <p>{error}</p>
          </div>
        )}

        {isLoading && (
          <div className="text-center py-16">
            <div className="animate-spin rounded-full h-10 w-10 border-4 border-primary border-t-transparent mx-auto" />
            <p className="mt-4 text-gray-600">Carregando áreas...</p>
          </div>
        )}

        {isError && (
          <Card>
            <CardContent className="py-12 text-center text-gray-600">
              Não foi possível carregar suas áreas.
            </CardContent>
          </Card>
        )}

        {!isLoading && !isError && areas.length === 0 && (
          <Card>
            <CardContent className="py-12 text-center">
              <MapIcon className="h-12 w-12 text-gray-300 mx-auto mb-4" />
              <h2 className="text-lg font-semibold text-gray-900">
                Nenhuma área salva
              </h2>
              <p className="mt-2 text-gray-600">
                Desenhe um polígono na tela de classificação e clique em
                “Salvar área”.
              </p>
              <Button onClick={() => navigate('/classify')} className="mt-6">
                Ir para Classificar
              </Button>
            </CardContent>
          </Card>
        )}

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          {areas.map((area) => (
            <Card key={area.id} className="h-full flex flex-col">
              <CardHeader>
                <CardTitle className="text-lg">{area.name}</CardTitle>
                <p className="text-sm text-gray-500">
                  {formatArea(area.area_ha)} · criada em{' '}
                  {formatDateTime(area.created_at)}
                </p>
              </CardHeader>
              <CardContent className="flex-1 flex flex-col justify-between gap-4">
                <p className="text-sm text-gray-600">
                  {area.description || 'Sem descrição.'}
                </p>
                <div className="flex flex-wrap gap-2">
                  <Button
                    size="sm"
                    className="gap-1.5"
                    onClick={() => navigate(`/classify?area_id=${area.id}`)}
                  >
                    <MapIcon className="h-4 w-4" />
                    Abrir no mapa
                  </Button>
                  <Button
                    size="sm"
                    variant="outline"
                    className="gap-1.5"
                    onClick={() => handleRename(area.id, area.name)}
                    disabled={updateArea.isPending}
                  >
                    <Pencil className="h-4 w-4" />
                    Renomear
                  </Button>
                  <Button
                    size="sm"
                    variant="ghost"
                    className="gap-1.5 text-red-600 hover:text-red-700"
                    onClick={() => handleDelete(area.id, area.name)}
                    disabled={deleteArea.isPending}
                  >
                    <Trash2 className="h-4 w-4" />
                    Excluir
                  </Button>
                </div>
              </CardContent>
            </Card>
          ))}
        </div>
      </main>
    </div>
  )
}
