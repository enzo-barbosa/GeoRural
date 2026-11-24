import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'
import path from 'path'

// Dentro do Docker o backend é alcançado pelo nome do serviço (`backend`).
// Fora dele, o padrão continua sendo localhost.
const apiTarget = process.env.VITE_PROXY_TARGET || 'http://localhost:8000'

export default defineConfig({
  plugins: [react()],
  resolve: {
    alias: {
      '@': path.resolve(__dirname, './src'),
    },
  },
  build: {
    outDir: '../backend/frontend/dist',
    emptyOutDir: true,
    cssMinify: false,
  },
  server: {
    port: 5173,
    proxy: {
      '/api': {
        target: apiTarget,
        changeOrigin: true,
      },
      '/outputs': {
        target: apiTarget,
        changeOrigin: true,
      },
      '/health': {
        target: apiTarget,
        changeOrigin: true,
      },
    },
  },
})
