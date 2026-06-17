import { createReadStream, existsSync, statSync } from "node:fs";
import { extname, join, normalize, resolve } from "node:path";
import http from "node:http";
import process from "node:process";

const args = process.argv.slice(2);
const portFlagIndex = args.findIndex((arg) => arg === "--port");
const port = portFlagIndex >= 0 ? Number.parseInt(args[portFlagIndex + 1] ?? "", 10) : 4173;

if (!Number.isInteger(port) || port <= 0) {
  console.error("Invalid port. Use --port <number>.");
  process.exit(1);
}

const distDir = resolve(process.cwd(), "dist");
const mimeTypes = {
  ".css": "text/css; charset=utf-8",
  ".html": "text/html; charset=utf-8",
  ".js": "text/javascript; charset=utf-8",
  ".json": "application/json; charset=utf-8",
  ".png": "image/png",
  ".jpg": "image/jpeg",
  ".jpeg": "image/jpeg",
  ".svg": "image/svg+xml",
  ".ico": "image/x-icon",
  ".txt": "text/plain; charset=utf-8",
  ".woff": "font/woff",
  ".woff2": "font/woff2",
};

function safeResolve(requestPath) {
  const decodedPath = decodeURIComponent(requestPath.split("?")[0]);
  const normalizedPath = normalize(decodedPath).replace(/^(\.\.[/\\])+/, "");
  const targetPath = resolve(distDir, `.${normalizedPath}`);
  return targetPath.startsWith(distDir) ? targetPath : distDir;
}

function sendFile(filePath, response) {
  const ext = extname(filePath).toLowerCase();
  const contentType = mimeTypes[ext] ?? "application/octet-stream";
  response.writeHead(200, { "Content-Type": contentType });
  createReadStream(filePath).pipe(response);
}

const server = http.createServer((request, response) => {
  const requestPath = request.url ?? "/";
  let filePath = requestPath === "/" ? join(distDir, "index.html") : safeResolve(requestPath);

  if (existsSync(filePath) && statSync(filePath).isDirectory()) {
    filePath = join(filePath, "index.html");
  }

  if (existsSync(filePath) && statSync(filePath).isFile()) {
    sendFile(filePath, response);
    return;
  }

  const spaFallback = join(distDir, "index.html");
  if (existsSync(spaFallback)) {
    sendFile(spaFallback, response);
    return;
  }

  response.writeHead(404, { "Content-Type": "text/plain; charset=utf-8" });
  response.end("dist/index.html not found. Run npm install and npm run build first.");
});

server.listen(port, "127.0.0.1", () => {
  console.log(`IDS demo running at http://127.0.0.1:${port}`);
});

function shutdown() {
  server.close(() => process.exit(0));
}

process.on("SIGINT", shutdown);
process.on("SIGTERM", shutdown);
