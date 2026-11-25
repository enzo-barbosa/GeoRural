import { Link } from 'react-router-dom'
import { MapPin, Cpu, Layers, ArrowRight } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { UserMenu } from '@/components/UserMenu'
import { useAuth } from '@/contexts/AuthContext'

const features = [
  {
    icon: MapPin,
    title: 'Imagens de Satélite',
    description: 'Sentinel-2 e Landsat 8 via Google Earth Engine, com pré-processamento automático.',
  },
  {
    icon: Cpu,
    title: 'Classificadores ML',
    description: 'Random Forest, SVM e U-Net CNN para comparar acurácia e custo computacional.',
  },
  {
    icon: Layers,
    title: 'Interface Interativa',
    description: 'Desenhe a área, visualize resultados e identifique áreas ociosas no mapa.',
  },
]

const steps = [
  { number: '1', text: 'Vá para <strong>Classificar</strong> no menu lateral.' },
  { number: '2', text: '<strong>Desenhe um polígono</strong> no mapa definindo sua área de interesse.' },
  { number: '3', text: 'Selecione o <strong>satélite</strong> e o <strong>classificador</strong>.' },
  { number: '4', text: 'Clique em <strong>Classificar</strong> e aguarde o processamento.' },
  { number: '5', text: 'Visualize os <strong>resultados</strong> na página Resultados.' },
]

const classes = [
  { name: 'Vegetação Nativa', description: 'Mata, cerrado, floresta', color: '#1a5276' },
  { name: 'Pastagem', description: 'Áreas de pasto', color: '#0E6B3D' },
  { name: 'Agricultura', description: 'Lavouras cultivadas', color: '#2ecc71' },
  { name: 'Solo Exposto', description: 'Terra sem cobertura', color: '#f39c12' },
  { name: 'Área Construída', description: 'Edificações, estradas', color: '#e74c3c' },
  { name: 'Água', description: 'Rios, represas e lagos', color: '#2e86de' },
]

export function Home() {
  const { user } = useAuth()

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
              <Link to="/classify">
                <Button variant="ghost">Classificar</Button>
              </Link>
              <Link to="/results">
                <Button variant="ghost">Resultados</Button>
              </Link>
              {user && (
                <>
                  <Link to="/areas">
                    <Button variant="ghost">Minhas áreas</Button>
                  </Link>
                  <Link to="/analyses">
                    <Button variant="ghost">Histórico</Button>
                  </Link>
                </>
              )}
              {user ? (
                <UserMenu />
              ) : (
                <>
                  <Link to="/login">
                    <Button variant="ghost">Entrar</Button>
                  </Link>
                  <Link to="/register">
                    <Button>Criar conta</Button>
                  </Link>
                </>
              )}
            </nav>
          </div>
        </div>
      </header>

      <main>
        <section className="relative overflow-hidden bg-gradient-to-b from-primary/5 to-white py-20 sm:py-32">
          <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 text-center">
            <h1 className="text-4xl sm:text-5xl lg:text-6xl font-bold tracking-tight text-gray-900">
              Geolocalização de{' '}
              <span className="text-primary">Propriedades Rurais</span>
            </h1>
            <p className="mt-6 text-lg sm:text-xl text-gray-600 max-w-3xl mx-auto">
              Classifique o uso e cobertura do solo com imagens de satélite e aprendizado de máquina.
              Identifique áreas ociosas, pastagens, agricultura e vegetação nativa.
            </p>
            <div className="mt-10 flex justify-center gap-4">
              <Link to="/classify">
                <Button size="lg" className="gap-2">
                  Começar Classificação
                  <ArrowRight className="h-4 w-4" />
                </Button>
              </Link>
              <Link to="/results">
                <Button size="lg" variant="outline">
                  Ver Resultados
                </Button>
              </Link>
            </div>
          </div>
        </section>

        <section className="py-20 sm:py-28 bg-white">
          <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
            <div className="text-center mb-16">
              <h2 className="text-3xl sm:text-4xl font-bold text-gray-900">Funcionalidades</h2>
              <p className="mt-4 text-lg text-gray-600">Ferramentas poderosas para análise de uso do solo</p>
            </div>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
              {features.map((feature, index) => (
                <Card key={index} className="h-full">
                  <CardHeader>
                    <feature.icon className="h-10 w-10 text-primary mb-2" />
                    <CardTitle>{feature.title}</CardTitle>
                  </CardHeader>
                  <CardContent>
                    <p className="text-gray-600">{feature.description}</p>
                  </CardContent>
                </Card>
              ))}
            </div>
          </div>
        </section>

        <section className="py-20 sm:py-28 bg-gray-50">
          <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
            <div className="text-center mb-16">
              <h2 className="text-3xl sm:text-4xl font-bold text-gray-900">Como Usar</h2>
              <p className="mt-4 text-lg text-gray-600">Processo simples em 5 passos</p>
            </div>
            <div className="space-y-6 max-w-3xl mx-auto">
              {steps.map((step, index) => (
                <div key={index} className="flex gap-4 text-left">
                  <div className="flex-shrink-0 flex items-center justify-center w-10 h-10 rounded-full bg-primary text-white font-bold">
                    {step.number}
                  </div>
                  <div className="pt-1">
                    <p className="text-gray-700" dangerouslySetInnerHTML={{ __html: step.text }} />
                  </div>
                </div>
              ))}
            </div>
          </div>
        </section>

        <section className="py-20 sm:py-28 bg-white">
          <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
            <div className="text-center mb-16">
              <h2 className="text-3xl sm:text-4xl font-bold text-gray-900">Classes de Cobertura</h2>
              <p className="mt-4 text-lg text-gray-600">6 classes (Brasil: MapBiomas; global: ESA WorldCover)</p>
            </div>
            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-5 gap-4">
              {classes.map((cls, index) => (
                <Card key={index} className="text-center">
                  <CardContent className="py-6">
                    <div
                      className="w-12 h-12 rounded-full mx-auto mb-3"
                      style={{ backgroundColor: cls.color }}
                    />
                    <h3 className="font-semibold text-gray-900">{cls.name}</h3>
                    <p className="text-sm text-gray-500 mt-1">{cls.description}</p>
                  </CardContent>
                </Card>
              ))}
            </div>
          </div>
        </section>

        <section className="py-20 bg-primary text-white text-center">
          <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
            <h2 className="text-3xl sm:text-4xl font-bold">Pronto para começar?</h2>
            <p className="mt-4 text-lg text-primary-100">Classifique sua primeira área agora mesmo</p>
            <Link to="/classify" className="mt-8 inline-block">
              <Button size="lg" variant="secondary" className="gap-2">
                Acessar Classificador
                <ArrowRight className="h-4 w-4" />
              </Button>
            </Link>
          </div>
        </section>
      </main>

      <footer className="bg-gray-900 text-white py-12">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 text-center">
          <p>TCC - Geolocalização de Propriedades Rurais</p>
          <p className="text-gray-400 mt-1">Centro Universitário de Fernandópolis</p>
        </div>
      </footer>
    </div>
  )
}