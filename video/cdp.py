"""Minimal Chrome DevTools Protocol client (stdlib only) + screencast recorder.

record(url, steps, out_dir): opens url in headless Chrome, runs steps
[(js or None, seconds), ...] while saving every screencast frame as JPEG with its
timestamp, then writes frames.txt (ffmpeg concat list) next to them.
"""
import base64, json, os, socket, struct, subprocess, threading, time, urllib.request

CHROME = r"C:\Program Files\Google\Chrome\Application\chrome.exe"


class WS:
    def __init__(self, url):
        host, rest = url[len("ws://"):].split("/", 1)
        h, p = host.split(":")
        self.s = socket.create_connection((h, int(p)))
        key = base64.b64encode(os.urandom(16)).decode()
        self.s.sendall((f"GET /{rest} HTTP/1.1\r\nHost: {host}\r\nUpgrade: websocket\r\nConnection: Upgrade\r\n"
                        f"Sec-WebSocket-Key: {key}\r\nSec-WebSocket-Version: 13\r\n\r\n").encode())
        buf = b""
        while b"\r\n\r\n" not in buf:
            buf += self.s.recv(4096)
        self.buf = buf.split(b"\r\n\r\n", 1)[1]
        self.lock = threading.Lock()

    def _read(self, n):
        while len(self.buf) < n:
            chunk = self.s.recv(1 << 20)
            if not chunk:
                raise ConnectionError("closed")
            self.buf += chunk
        out, self.buf = self.buf[:n], self.buf[n:]
        return out

    def send(self, text):
        data = text.encode()
        mask = os.urandom(4)
        n = len(data)
        head = bytes([0x81]) + (bytes([0x80 | n]) if n < 126 else
                                bytes([0x80 | 126]) + struct.pack(">H", n) if n < 65536 else
                                bytes([0x80 | 127]) + struct.pack(">Q", n))
        with self.lock:
            self.s.sendall(head + mask + bytes(b ^ mask[i % 4] for i, b in enumerate(data)))

    def recv(self):
        parts = []
        while True:
            b1, b2 = self._read(2)
            n = b2 & 0x7F
            if n == 126:
                n = struct.unpack(">H", self._read(2))[0]
            elif n == 127:
                n = struct.unpack(">Q", self._read(8))[0]
            payload = self._read(n)
            op = b1 & 0x0F
            if op in (1, 0):
                parts.append(payload)
                if b1 & 0x80:
                    return b"".join(parts).decode()
            elif op == 8:
                raise ConnectionError("closed")


class Browser:
    def __init__(self, width=1280, height=720, scale=1.5, port=9333, profile=None, extra=()):
        self.proc = subprocess.Popen([CHROME, "--headless=new", f"--remote-debugging-port={port}", "--no-first-run",
                                      "--hide-scrollbars", "--mute-audio", "--autoplay-policy=no-user-gesture-required",
                                      f"--user-data-dir={profile}", f"--window-size={width},{height}", *extra, "about:blank"],
                                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        for _ in range(100):
            try:
                tabs = json.load(urllib.request.urlopen(f"http://127.0.0.1:{port}/json/list"))
                ws = next(t["webSocketDebuggerUrl"] for t in tabs if t["type"] == "page")
                break
            except Exception:
                time.sleep(0.2)
        self.ws = WS(ws)
        self.id = 0
        self.pending = {}
        self.events = []
        self.cond = threading.Condition()
        threading.Thread(target=self._loop, daemon=True).start()
        self.call("Page.enable")
        self.call("Runtime.enable")
        self.call("Emulation.setDeviceMetricsOverride", width=width, height=height, deviceScaleFactor=scale, mobile=False)
        self.size = (int(width * scale), int(height * scale))
        self.on_frame = None

    def _loop(self):
        while True:
            try:
                msg = json.loads(self.ws.recv())
            except Exception:
                return
            if msg.get("method") == "Page.screencastFrame" and self.on_frame:
                self.on_frame(msg["params"])
                continue
            with self.cond:
                if "id" in msg:
                    self.pending[msg["id"]] = msg
                else:
                    self.events.append(msg)
                self.cond.notify_all()

    def call(self, method, **params):
        with self.cond:
            self.id += 1
            i = self.id
        self.ws.send(json.dumps({"id": i, "method": method, "params": params}))
        with self.cond:
            self.cond.wait_for(lambda: i in self.pending, timeout=60)
            msg = self.pending.pop(i, {})
        if "error" in msg:
            raise RuntimeError(f"{method}: {msg['error']}")
        return msg.get("result", {})

    def js(self, expr):
        r = self.call("Runtime.evaluate", expression=expr, awaitPromise=True, returnByValue=True)
        if r.get("exceptionDetails"):
            raise RuntimeError(f"JS error: {r['exceptionDetails']}")
        return r.get("result", {}).get("value")

    def goto(self, url, wait=2.0):
        self.call("Page.navigate", url=url)
        time.sleep(wait)

    def shot(self, path):
        data = self.call("Page.captureScreenshot", format="png")["data"]
        open(path, "wb").write(base64.b64decode(data))

    def close(self):
        self.proc.kill()


def record(browser, steps, out_dir, quality=88):
    """Run steps [(js or callable or None, seconds)] while recording; write frames + concat list."""
    os.makedirs(out_dir, exist_ok=True)
    frames = []

    def on_frame(p):
        n = len(frames)
        path = os.path.join(out_dir, f"f{n:05d}.jpg")
        open(path, "wb").write(base64.b64decode(p["data"]))
        frames.append((time.perf_counter(), path))
        threading.Thread(target=browser.call, args=("Page.screencastFrameAck",), kwargs={"sessionId": p["sessionId"]},
                         daemon=True).start()

    browser.on_frame = on_frame
    w, h = browser.size
    browser.call("Page.startScreencast", format="jpeg", quality=quality, maxWidth=w, maxHeight=h, everyNthFrame=1)
    t0 = time.perf_counter()
    for action, secs in steps:
        start = time.perf_counter()
        if callable(action):
            action()
        elif action:
            browser.js(action)
        time.sleep(max(0, secs - (time.perf_counter() - start)))
    total = time.perf_counter() - t0
    browser.call("Page.stopScreencast")
    browser.on_frame = None
    time.sleep(0.3)
    # concat list: each frame lasts until the next one; frames before t0 start at 0
    lines = []
    seq = [(max(0.0, t - t0), p) for t, p in frames]
    for i, (t, p) in enumerate(seq):
        nxt = seq[i + 1][0] if i + 1 < len(seq) else total
        lines.append(f"file '{os.path.abspath(p)}'\nduration {max(0.001, nxt - t):.4f}")
    if seq:
        lines.append(f"file '{os.path.abspath(seq[-1][1])}'")
    open(os.path.join(out_dir, "frames.txt"), "w").write("\n".join(lines) + "\n")
    return total, len(frames)
