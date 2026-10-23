import { Component, type ErrorInfo, type ReactNode } from 'react'
import { AlertCircle } from 'lucide-react'
import { Button } from '@/components/ui/button'

interface ErrorBoundaryProps {
  children: ReactNode
}

interface ErrorBoundaryState {
  hasError: boolean
  message: string
}

export class ErrorBoundary extends Component<ErrorBoundaryProps, ErrorBoundaryState> {
  state: ErrorBoundaryState = { hasError: false, message: '' }

  static getDerivedStateFromError(error: Error): ErrorBoundaryState {
    return { hasError: true, message: error.message }
  }

  componentDidCatch(error: Error, info: ErrorInfo) {
    console.error('Erro não tratado na aplicação:', error, info)
  }

  handleReload = () => {
    this.setState({ hasError: false, message: '' })
    window.location.href = '/'
  }

  render() {
    if (this.state.hasError) {
      return (
        <div className="min-h-screen bg-gray-50 flex items-center justify-center p-4">
          <div className="w-full max-w-md rounded-xl border border-gray-200 bg-white p-8 text-center shadow-sm">
            <AlertCircle className="h-12 w-12 text-red-500 mx-auto mb-4" />
            <h1 className="text-xl font-bold text-gray-900">Algo deu errado</h1>
            <p className="mt-2 text-sm text-gray-600">
              Ocorreu um erro inesperado ao renderizar esta página.
            </p>
            {this.state.message && (
              <p className="mt-2 break-words text-xs text-gray-400">{this.state.message}</p>
            )}
            <Button onClick={this.handleReload} className="mt-6">
              Voltar ao início
            </Button>
          </div>
        </div>
      )
    }

    return this.props.children
  }
}
