"""isotope_zero CLI — Browser dashboard served at localhost.

``izero serve [--port]`` runs a tiny read-only HTTP server (stdlib
``http.server`` — zero new dependencies) that serves the built React
dashboard at ``http://localhost:<port>`` and pushes live state to it
via Server-Sent Events.

Design: deep void black / white / ocean-blue palette, shadcn/21st.dev-inspired
layered surfaces, glassmorphism KPI cards, dock footer, animated
search bar, tag chips, alert banner, and progressive loading.

Endpoints:
    GET /            → the React dashboard (served from web/dist/)
    GET /api/state   → one JSON snapshot of the store (polling / initial load)
    GET /api/events  → SSE stream; pushes a fresh state JSON every interval
    GET /assets/*    → static assets (JS/CSS) from web/dist/assets/
"""

from __future__ import annotations

import json
import mimetypes
import os
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable

from .data import collect_state

# Refresh cadence for the SSE stream (seconds).
DEFAULT_INTERVAL = 2.0
DEFAULT_PORT = 8930

# Path to the built React app (web/dist relative to repo root)
_REPO_ROOT = Path(__file__).resolve().parents[4]  # src/isotope_zero/cli/dash/ -> repo
_DIST_DIR = _REPO_ROOT / "web" / "dist"
_ASSETS_DIR = _DIST_DIR / "assets"

# Fallback inline HTML (for environments without built React app)
# Palette: deep void black, white, ocean-blue accents (#90CAF9 light, #2196F3 material blue)
_FALLBACK_HTML = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>isotope_zero · memory dashboard</title>
<style>
  :root {
    --void: #000000;
    --white: #ffffff;
    --sage: #90CAF9;
    --moss: #7B93B0;
    --forest: #0D47A1;
    --charcoal: #1A1A1A;
    --surface: #0A0A0A;
    --surface-2: #121212;
    --border: rgba(144, 202, 249, 0.12);
    --border-active: rgba(144, 202, 249, 0.35);
    --glass-bg: rgba(26, 26, 26, 0.55);
    --glass-border: rgba(144, 202, 249, 0.15);
    --success: #90CAF9; --warning: #FFB74D; --error: #EF5350; --info: #4FC3F7;
    --mono: ui-monospace, "SF Mono", "Cascadia Code", Menlo, Consolas, monospace;
    --sans: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    --radius-sm: 6px; --radius: 10px; --radius-lg: 14px; --radius-xl: 18px;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  html, body { height: 100%; background: var(--void); color: var(--white); font-family: var(--sans); }
  body { display: flex; flex-direction: column; align-items: center; justify-content: center; gap: 24px; padding: 24px; }
  h1 { font-size: 28px; font-weight: 600; letter-spacing: -0.02em; }
  .sub { color: var(--moss); font-size: 14px; }
  .spinner { width: 40px; height: 40px; border: 3px solid var(--border); border-top-color: var(--sage); border-radius: 50%; animation: spin 1s linear infinite; }
  @keyframes spin { to { transform: rotate(360deg); } }
  .hint { font-family: var(--mono); font-size: 12px; color: var(--charcoal); }
</style>
</head>
<body>
  <div class="spinner" aria-hidden="true"></div>
  <h1>isotope_zero</h1>
  <p class="sub">memory dashboard</p>
  <p class="hint">React build not found at <code>web/dist/</code>. Run <code>cd web && npm run build</code> or use the inline fallback.</p>
</body>
</html>
"""


class _DashboardHandler(BaseHTTPRequestHandler):
    """Serves the page + JSON/SSE endpoints. Read-only."""

    _supplier: Callable[[], dict[str, Any]] | None = None
    _interval: float = DEFAULT_INTERVAL
    _heartbeat: float = 15.0
    server_version = "isotope-zero-dash/1.3"
    protocol_version = "HTTP/1.1"

    def _send_json(self, obj: dict[str, Any], status: int = 200) -> None:
        body = json.dumps(obj).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_html(self, html: str, status: int = 200) -> None:
        body = html.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_file(self, filepath: Path, status: int = 200) -> None:
        """Send a file with appropriate MIME type."""
        try:
            data = filepath.read_bytes()
        except OSError:
            self._send_html("not found", 404)
            return
        mime, _ = mimetypes.guess_type(str(filepath))
        if mime is None:
            mime = "application/octet-stream"
        self.send_response(status)
        self.send_header("Content-Type", mime)
        self.send_header("Cache-Control", "public, max-age=31536000, immutable")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self) -> None:
        supplier = self._supplier
        path = self.path.split("?", 1)[0]

        # Serve static assets from web/dist/assets/
        if path.startswith("/assets/"):
            asset_path = _DIST_DIR / path.lstrip("/")
            if asset_path.is_file():
                self._send_file(asset_path)
                return
            self._send_html("not found", 404)
            return

        # Serve React index.html for root and any non-API route (SPA fallback)
        if path in ("/", "/index.html", ""):
            index_path = _DIST_DIR / "index.html"
            if index_path.is_file():
                self._send_file(index_path)
                return
            # Fallback to inline HTML if build not found
            self._send_html(_FALLBACK_HTML)
            return

        # API endpoints
        if path == "/api/state":
            if supplier is None:
                self._send_json({"error": "no store attached"}, 500)
                return
            try:
                self._send_json(supplier())
            except Exception as e:
                self._send_json({"error": str(e)}, 500)
            return

        if path == "/api/events":
            self._stream_events()
            return

        # Any other route: SPA fallback to index.html (if exists) or 404
        index_path = _DIST_DIR / "index.html"
        if index_path.is_file():
            self._send_file(index_path)
        else:
            self._send_html(_FALLBACK_HTML)

    def _stream_events(self) -> None:
        """Server-Sent Events with heartbeat pings and immediate first frame."""
        supplier = self._supplier
        if supplier is None:
            self._send_json({"error": "no store attached"}, 500)
            return
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Connection", "keep-alive")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(b"retry: 3000\n\n")
        self.wfile.flush()
        next_data = time.monotonic()
        next_ping = time.monotonic() + self._heartbeat
        try:
            while True:
                now = time.monotonic()
                if now >= next_data:
                    try:
                        data = json.dumps(supplier()).encode("utf-8")
                    except Exception:
                        data = b'{"error":"state unavailable"}'
                    self.wfile.write(b"data: " + data + b"\n\n")
                    self.wfile.flush()
                    next_data = now + self._interval
                elif now >= next_ping:
                    self.wfile.write(b": ping\n\n")
                    self.wfile.flush()
                    next_ping = now + self._heartbeat
                else:
                    time.sleep(min(0.25, next_data - now, next_ping - now))
        except (BrokenPipeError, ConnectionResetError, OSError):
            pass

    def log_message(self, fmt: str, *args: Any) -> None:
        pass


class DashboardServer:
    """Thin wrapper bundling the store supplier + HTTP server lifecycle."""

    def __init__(
        self,
        store,
        db_path: str,
        *,
        host: str = "127.0.0.1",
        port: int = DEFAULT_PORT,
        interval: float = DEFAULT_INTERVAL,
    ) -> None:
        self.store = store
        self.db_path = db_path
        self.host = host
        self.port = port
        self.interval = interval

        _DashboardHandler._supplier = staticmethod(
            lambda: collect_state(self.store, self.db_path)
        )
        _DashboardHandler._interval = interval
        self.httpd = ThreadingHTTPServer((host, port), _DashboardHandler)

    @property
    def url(self) -> str:
        return f"http://{self.host}:{self.port}"

    def serve_forever(self) -> None:
        try:
            self.httpd.serve_forever()
        except KeyboardInterrupt:
            pass
        finally:
            self.httpd.server_close()

    def close(self) -> None:
        self.httpd.server_close()


def run_server(
    store,
    db_path: str,
    *,
    host: str = "127.0.0.1",
    port: int = DEFAULT_PORT,
    interval: float = DEFAULT_INTERVAL,
    open_browser: bool = False,
) -> int:
    """Run the dashboard server until interrupted. Returns 0 on clean exit."""
    server = DashboardServer(store, db_path, host=host, port=port, interval=interval)
    url = server.url
    print(f"{'●' if True else ''} isotope_zero memory dashboard", flush=True)
    print(f"  open: {url}", flush=True)
    print(f"  db:   {db_path}  ·  refresh {interval:g}s", flush=True)
    print("  Ctrl-C to stop", flush=True)
    if open_browser:
        import webbrowser

        webbrowser.open(url)
    server.serve_forever()
    server.close()
    return 0


__all__ = [
    "DEFAULT_INTERVAL",
    "DEFAULT_PORT",
    "DashboardServer",
    "run_server",
    "_PAGE_HTML",
]
