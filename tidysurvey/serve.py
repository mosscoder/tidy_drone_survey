"""`tidysurvey serve` — view the products folder locally, byte ranges included.

PMTiles clients read archives with HTTP Range requests. Python's stock
`http.server` silently ignores Range (full-file 200s), so the obvious local
server breaks the map viewer. This one implements single-range byte serving
(RFC 7233) on top of the stdlib handler — no dependencies, threaded, good
enough to pan around a 4 GB archive on localhost. Real hosting is any static
server or bucket that supports byte serving (S3/GCS do).
"""
from __future__ import annotations

import os
import re
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

_RANGE = re.compile(r"bytes=(\d*)-(\d*)$")


class _Limited:
    """File-like that stops after `remaining` bytes — copyfile() reads until
    empty, so this bounds the base handler's send loop to the range."""

    def __init__(self, f, remaining):
        self.f = f
        self.remaining = remaining

    def read(self, n=-1):
        if self.remaining <= 0:
            return b""
        n = self.remaining if n < 0 else min(n, self.remaining)
        data = self.f.read(n)
        self.remaining -= len(data)
        return data

    def close(self):
        self.f.close()


class RangeRequestHandler(SimpleHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def end_headers(self):
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Access-Control-Allow-Origin", "*")
        super().end_headers()

    def send_head(self):
        path = self.translate_path(self.path)
        rng = self.headers.get("Range")
        if not rng or os.path.isdir(path):
            # advertise ranges on plain responses so clients ask for them
            resp = super().send_head()
            return resp
        m = _RANGE.match(rng.strip())
        try:
            f = open(path, "rb")
        except OSError:
            self.send_error(404, "File not found")
            return None
        size = os.fstat(f.fileno()).st_size
        if not m or (not m.group(1) and not m.group(2)):
            f.close()
            self.send_error(416, "Unsatisfiable Range")
            return None
        if not m.group(1):                       # suffix form: last N bytes
            length = min(int(m.group(2)), size)
            start, end = size - length, size - 1
        else:
            start = int(m.group(1))
            end = min(int(m.group(2)), size - 1) if m.group(2) else size - 1
        if start >= size or start > end:
            f.close()
            self.send_response(416)
            self.send_header("Content-Range", f"bytes */{size}")
            self.send_header("Content-Length", "0")   # HTTP/1.1 keep-alive
            self.end_headers()
            return None
        self.send_response(206)
        self.send_header("Content-Type", self.guess_type(path))
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.send_header("Content-Length", str(end - start + 1))
        self.end_headers()
        f.seek(start)
        return _Limited(f, end - start + 1)

    def log_message(self, fmt, *args):          # one quiet line per request
        pass


def serve_dir(root, port=8080, open_url=None, log=print):
    """Serve `root` (blocking) with byte-range support until Ctrl-C.
    If open_url is set, open it in the default browser once we're listening."""
    handler = partial(RangeRequestHandler, directory=str(root))
    httpd = ThreadingHTTPServer(("", port), handler)
    if open_url:
        import threading
        import webbrowser

        def _open():
            try:
                webbrowser.open(open_url)
            except Exception:
                pass                     # a missing browser must not kill the server
        # fire just after serve_forever starts so the first request is handled
        threading.Timer(0.4, _open).start()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        log("  serve: stopped")
    finally:
        httpd.server_close()
