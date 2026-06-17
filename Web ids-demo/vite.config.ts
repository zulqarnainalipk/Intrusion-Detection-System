import path from "path"
import react from "@vitejs/plugin-react"
import { defineConfig } from "vite"
import sourceIdentifierPlugin from 'vite-plugin-source-identifier'

export default defineConfig(({ command, mode }) => {
  const isProductionBuild = command === "build" || mode === "production" || process.env.BUILD_MODE === "prod";

  return {
    plugins: [
      react(),
      sourceIdentifierPlugin({
        enabled: !isProductionBuild,
        attributePrefix: "data-matrix",
        includeProps: true,
      }),
    ],
    resolve: {
      alias: {
        "@": path.resolve(__dirname, "./src"),
      },
    },
  };
})
