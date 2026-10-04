"""Self-contained localhost CAPTCHA-style benchmark server.

The visual design is intentionally neutral and uses only generated shapes/text.
No external CAPTCHA provider, asset, API, or branding is contacted.
"""

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Optional
from urllib.parse import urlencode


_PAGE = r'''<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width,initial-scale=1" />
<title>Visual Verification Lab</title>
<style>
  * { box-sizing: border-box; }
  body { margin:0; min-height:100vh; font-family:Arial,sans-serif; background:#f3f5f7; color:#202124; display:flex; align-items:center; justify-content:center; }
  .card { width:520px; background:white; border:1px solid #d9dde3; border-radius:10px; box-shadow:0 6px 24px rgba(0,0,0,.08); overflow:hidden; }
  .head { padding:20px 22px 16px; border-bottom:1px solid #e5e8ec; }
  .brand { font-size:13px; letter-spacing:.08em; text-transform:uppercase; color:#68707d; }
  h1 { font-size:22px; margin:6px 0 0; }
  #prompt { font-size:16px; margin:0 0 14px; line-height:1.4; }
  .body { padding:22px; }
  .grid { display:grid; grid-template-columns:repeat(3, 1fr); gap:8px; }
  .tile { height:108px; border:2px solid #d5dae1; background:#f8fafc; border-radius:8px; font-size:42px; cursor:pointer; user-select:none; }
  .tile.selected { border-color:#3b6ea8; background:#e8f1fb; box-shadow:inset 0 0 0 2px #3b6ea8; }
  .order-grid { display:grid; grid-template-columns:repeat(3,1fr); gap:10px; }
  .order-btn { height:86px; border:2px solid #d5dae1; background:#fbfcfe; border-radius:8px; font-size:30px; cursor:pointer; }
  .order-btn.chosen { border-color:#6d7c91; background:#edf1f5; }
  .track-wrap { padding:34px 12px 24px; }
  .track { position:relative; height:54px; border-radius:10px; background:#e6e9ee; border:1px solid #cbd1d9; overflow:hidden; }
  .target { position:absolute; top:4px; width:34px; height:44px; border:2px dashed #68798d; border-radius:7px; background:rgba(255,255,255,.45); }
  .handle { position:absolute; top:5px; left:8px; width:42px; height:42px; border-radius:8px; background:#426a94; box-shadow:0 2px 6px rgba(0,0,0,.2); cursor:grab; }
  .handle:active { cursor:grabbing; }
  .footer { display:flex; align-items:center; justify-content:space-between; gap:12px; margin-top:18px; }
  #status { min-height:22px; font-size:14px; color:#596273; }
  #verify { min-width:108px; height:42px; border:0; border-radius:7px; background:#315f8d; color:white; font-weight:700; cursor:pointer; }
  #verify:hover { background:#264c71; }
  .meta { padding:12px 22px; background:#fafbfc; border-top:1px solid #e5e8ec; font-size:12px; color:#7a8390; }
</style>
</head>
<body>
<div class="card">
  <div class="head"><div class="brand">Local benchmark</div><h1>Visual Verification Lab</h1></div>
  <div class="body">
    <p id="prompt"></p>
    <div id="challenge"></div>
    <div class="footer"><div id="status"></div><button id="verify">Verify</button></div>
  </div>
  <div class="meta">Synthetic fixture · loopback only · seed <span id="seed"></span></div>
</div>
<script>
(() => {
  const params = new URLSearchParams(location.search);
  const seed = (parseInt(params.get('seed') || '0', 10) >>> 0);
  const forcedType = (params.get('type') || '').toLowerCase();
  document.getElementById('seed').textContent = String(seed);

  let state = seed || 1;
  const rnd = () => {
    state = (1664525 * state + 1013904223) >>> 0;
    return state / 4294967296;
  };
  const randint = (n) => Math.floor(rnd() * n);
  const shuffle = (arr) => {
    const out = arr.slice();
    for (let i = out.length - 1; i > 0; i--) {
      const j = randint(i + 1); [out[i], out[j]] = [out[j], out[i]];
    }
    return out;
  };

  const types = ['grid', 'order', 'slider'];
  const type = types.includes(forcedType) ? forcedType : types[seed % types.length];
  const prompt = document.getElementById('prompt');
  const challenge = document.getElementById('challenge');
  const status = document.getElementById('status');
  const verify = document.getElementById('verify');

  window.__motionTrace = [];
  const trace = (kind, e) => {
    window.__motionTrace.push({kind, x:e.clientX, y:e.clientY, t:performance.now()});
  };
  document.addEventListener('mousemove', e => trace('move', e));
  document.addEventListener('mousedown', e => trace('down', e));
  document.addEventListener('mouseup', e => trace('up', e));

  window.__labResult = {done:false, reward:0, type, seed};
  const finish = (ok) => {
    window.__labResult = {
      done:true,
      reward:ok ? 1 : 0,
      type,
      seed,
      motion_events:window.__motionTrace.length
    };
    status.textContent = ok ? 'Passed' : 'Try again';
    status.style.color = ok ? '#23723a' : '#a13b33';
  };

  let verifyFn = () => false;

  if (type === 'grid') {
    const symbols = ['▲', '●', '■', '◆'];
    const target = symbols[randint(symbols.length)];
    const labels = Array.from({length:9}, () => symbols[randint(symbols.length)]);
    const forced = shuffle([0,1,2,3,4,5,6,7,8]).slice(0, 2 + randint(2));
    forced.forEach(i => labels[i] = target);
    const correct = new Set(labels.map((v,i) => v === target ? i : -1).filter(i => i >= 0));
    const selected = new Set();
    prompt.textContent = `Select every tile containing ${target}.`;
    const grid = document.createElement('div'); grid.className = 'grid';
    labels.forEach((label, i) => {
      const b = document.createElement('button');
      b.className = 'tile'; b.textContent = label;
      b.addEventListener('click', () => {
        if (selected.has(i)) { selected.delete(i); b.classList.remove('selected'); }
        else { selected.add(i); b.classList.add('selected'); }
      });
      grid.appendChild(b);
    });
    challenge.appendChild(grid);
    verifyFn = () => selected.size === correct.size && [...selected].every(i => correct.has(i));
  }

  if (type === 'order') {
    const symbols = shuffle(['★','▲','●','◆','■','✚']);
    const target = shuffle(symbols).slice(0, 3);
    const chosen = [];
    prompt.textContent = `Click in this order: ${target.join('  →  ')}`;
    const grid = document.createElement('div'); grid.className = 'order-grid';
    shuffle(symbols).forEach(label => {
      const b = document.createElement('button');
      b.className = 'order-btn'; b.textContent = label;
      b.addEventListener('click', () => {
        if (chosen.length < target.length) {
          chosen.push(label); b.classList.add('chosen');
          b.dataset.rank = String(chosen.length);
        }
      });
      grid.appendChild(b);
    });
    challenge.appendChild(grid);
    verifyFn = () => chosen.length === target.length && chosen.every((v,i) => v === target[i]);
  }

  if (type === 'slider') {
    prompt.textContent = 'Drag the blue handle so its center is inside the dashed target.';
    const wrap = document.createElement('div'); wrap.className = 'track-wrap';
    const track = document.createElement('div'); track.className = 'track';
    const target = document.createElement('div'); target.className = 'target';
    const handle = document.createElement('div'); handle.className = 'handle';
    track.appendChild(target); track.appendChild(handle); wrap.appendChild(track); challenge.appendChild(wrap);

    const trackWidth = 452;
    const handleWidth = 42;
    const targetWidth = 34;
    const targetLeft = 125 + randint(250);
    target.style.left = `${targetLeft}px`;
    let left = 8;
    let dragging = false;
    let grabDx = 0;

    handle.addEventListener('mousedown', e => {
      dragging = true;
      const r = handle.getBoundingClientRect();
      grabDx = e.clientX - r.left;
      e.preventDefault();
    });
    document.addEventListener('mousemove', e => {
      if (!dragging) return;
      const r = track.getBoundingClientRect();
      left = Math.max(0, Math.min(trackWidth - handleWidth, e.clientX - r.left - grabDx));
      handle.style.left = `${left}px`;
    });
    document.addEventListener('mouseup', () => { dragging = false; });
    verifyFn = () => {
      const center = left + handleWidth / 2;
      const targetCenter = targetLeft + targetWidth / 2;
      return Math.abs(center - targetCenter) <= 15;
    };
  }

  verify.addEventListener('click', () => finish(Boolean(verifyFn())));
  window.__labGetState = () => ({...window.__labResult, motion_events:window.__motionTrace.length});
})();
</script>
</body>
</html>'''


class _Handler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):  # pragma: no cover - keep tests quiet
        return

    def do_GET(self):
        path = self.path.split("?", 1)[0]
        if path == "/health":
            payload = json.dumps({"ok": True}).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
            return
        if path != "/":
            self.send_error(404)
            return
        payload = _PAGE.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


class LocalBenchmarkServer:
    """Background HTTP server bound to 127.0.0.1 only."""

    def __init__(self, port: int = 0) -> None:
        self._requested_port = int(port)
        self._server: Optional[ThreadingHTTPServer] = None
        self._thread: Optional[threading.Thread] = None

    @property
    def port(self) -> int:
        if self._server is None:
            raise RuntimeError("server is not started")
        return int(self._server.server_address[1])

    def url(self, seed: int = 0, challenge_type: Optional[str] = None) -> str:
        query = {"seed": int(seed)}
        if challenge_type:
            query["type"] = challenge_type
        return "http://127.0.0.1:{}/?{}".format(self.port, urlencode(query))

    def start(self) -> "LocalBenchmarkServer":
        if self._server is not None:
            return self
        self._server = ThreadingHTTPServer(("127.0.0.1", self._requested_port), _Handler)
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        if self._server is None:
            return
        self._server.shutdown()
        self._server.server_close()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        self._server = None
        self._thread = None

    def __enter__(self) -> "LocalBenchmarkServer":
        return self.start()

    def __exit__(self, exc_type, exc, tb) -> None:
        self.stop()
