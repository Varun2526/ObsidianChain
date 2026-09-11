import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// The dashboard talks to the Phase 7 API and nothing else. Proxying /api in
// development keeps the backend untouched: no CORS middleware, no contract
// change, no new infrastructure.
export default defineConfig({
  plugins: [react()],
  build: {
    rollupOptions: {
      // Cytoscape is ~60% of the bundle and only the detail route needs it.
      // Splitting it keeps the queue - the first thing an investigator loads
      // - small, without hiding the cost behind a raised warning threshold.
      output: { manualChunks: { cytoscape: ["cytoscape"] } },
    },
  },
  server: {
    port: 5173,
    proxy: { "/api": { target: "http://127.0.0.1:8000", changeOrigin: true } },
  },
  test: {
    environment: "jsdom",
    setupFiles: ["./src/test/setup.ts"],
    globals: true,
    css: false,
  },
});
