import { defineConfig, loadEnv } from 'vite'
import react from '@vitejs/plugin-react'

// https://vite.dev/config/
export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, '..', '')
  const backendUrl = env.MAGIFF_API_URL || 'http://127.0.0.1:8000'
  const authorization = env.MAGIFF_API_KEY
    ? { Authorization: `Bearer ${env.MAGIFF_API_KEY}` }
    : undefined

  return {
    plugins: [react()],
    server: {
      proxy: {
        '/api/agent': {
          target: backendUrl,
          changeOrigin: true,
          rewrite: () => '/v1/agent/query',
          headers: authorization,
        },
      },
    },
  }
})
