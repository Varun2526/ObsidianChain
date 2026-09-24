import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Read without @types/node: this file is typechecked with the app's DOM lib.
const env = (globalThis as { process?: { env: Record<string, string | undefined> } }).process?.env ?? {};

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
    port: Number(env.OC_WEB_PORT ?? 5173),
    // OC_API_TARGET lets a second stack (an isolated end-to-end run) sit
    // beside the everyday one without touching its database.
    proxy: { "/api": { target: env.OC_API_TARGET ?? "http://127.0.0.1:8000", changeOrigin: true } },
  },
  test: {
    environment: "jsdom",
    setupFiles: ["./src/test/setup.ts"],
    globals: true,
    css: false,
  },
});
