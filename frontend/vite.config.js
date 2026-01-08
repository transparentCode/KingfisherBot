import { defineConfig } from 'vite'
import { svelte } from '@sveltejs/vite-plugin-svelte'

const backendUrl = process.env.BACKEND_URL || 'http://127.0.0.1:8080';

// https://vite.dev/config/
export default defineConfig({
  plugins: [svelte()],
  server: {
    proxy: {
      '/api': {
        target: backendUrl,
        changeOrigin: true,
        secure: false,
      },
      '/socket.io': {
        target: backendUrl,
        changeOrigin: true,
        ws: true,
        secure: false,
      }
    }
  }
})
