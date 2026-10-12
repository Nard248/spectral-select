/// <reference types="vitest/config" />
import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// Dev proxy: /api (HTTP + WebSocket) → the Python engine.
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: { '/api': { target: 'http://127.0.0.1:8765', changeOrigin: true, ws: true } },
  },
  build: { chunkSizeWarningLimit: 2600 },
  test: { environment: 'node', include: ['src/**/*.test.ts'] },
})
