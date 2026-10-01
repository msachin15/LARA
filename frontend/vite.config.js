// Vite config for the React frontend.
// Tailwind and the React plugin are both enabled. The dev server defaults to port 5173.
import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'

// https://vite.dev/config/
export default defineConfig({
  plugins: [tailwindcss(),react()],
})
