import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import wasm from 'vite-plugin-wasm'

// https://vite.dev/config/
export default defineConfig({
  base: '/webraw/',
  plugins: [react(), wasm()],
  build: {
    target: 'esnext',
  },
  worker: {
    format: 'es',
  },
})
