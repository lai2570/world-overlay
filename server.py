"""手機 App 的後端：收手機拍的照片 → 跑組裝分析 → 回傳引導畫面資料。API key 只放在這台電腦。

用法：
  .venv/Scripts/python server.py            # 手機連同一個 Wi-Fi，開 http://<電腦IP>:8000
  .venv/Scripts/python server.py --port 9000

分析要 1 分鐘左右，所以用「建立工作 → 輪詢進度」：
  POST /api/analyze   body 為 JPEG → {"job": id}
  GET  /api/job?id=   → {"status": running|done|error, "log": [...], "result": {...}}
  GET  /api/sample    → 不呼叫 API，直接回傳 out/mic_parts_guide.json 的範例結果
設定環境變數 APP_PASSCODE 時，請求要帶 X-Passcode 標頭（放到網路上時避免別人花你的額度）。
"""
import argparse
import io
import json
import os
import socket
import threading
import time
import traceback
import uuid
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from build_guide import STATIC, analyze, page_data
from test_grounding import ROOT, load_api_key, load_image, to_data_url

MAX_UPLOAD = 15 * 1024 * 1024
APP_OUT = ROOT / "out" / "app"
SAMPLE = ROOT / "out" / "mic_parts_guide.json"
SAMPLE_IMG = ROOT / "samples" / "mic_parts.jpg"
CONTENT_TYPES = {".html": "text/html; charset=utf-8", ".css": "text/css; charset=utf-8",
                 ".js": "text/javascript; charset=utf-8"}

jobs: dict[str, dict] = {}
jobs_lock = threading.Lock()


def run_job(job_id: str, photo: bytes, key: str):
    job = jobs[job_id]

    def log(msg: str):
        with jobs_lock:
            job["log"].append(msg)
        print(f"[{job_id}] {msg}")

    try:
        img = load_image(io.BytesIO(photo))
        plan, located, shapes = analyze(key, img, log=log)
        data = page_data(img, plan, located, shapes)
        APP_OUT.mkdir(parents=True, exist_ok=True)
        img.save(APP_OUT / f"{job_id}.jpg", quality=90)
        (APP_OUT / f"{job_id}.json").write_text(
            json.dumps({"plan": plan, "located": located, "shapes": shapes}, ensure_ascii=False), encoding="utf-8")
        with jobs_lock:
            job.update(status="done", result=data, seconds=round(time.time() - job["started"]))
    except Exception as e:  # 回報給手機，不讓工作卡在 running
        traceback.print_exc()
        with jobs_lock:
            job.update(status="error", error=str(e) or e.__class__.__name__)


class Handler(BaseHTTPRequestHandler):
    server_version = "WorldOverlay/0.1"

    def send_json(self, obj, status=HTTPStatus.OK):
        body = json.dumps(obj, ensure_ascii=False).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def send_file(self, path: Path):
        body = path.read_bytes()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", CONTENT_TYPES.get(path.suffix, "application/octet-stream"))
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self.wfile.write(body)

    def authorized(self) -> bool:
        code = os.environ.get("APP_PASSCODE")
        if code and self.headers.get("X-Passcode") != code:
            self.send_json({"error": "通行碼錯誤"}, HTTPStatus.UNAUTHORIZED)
            return False
        return True

    def do_GET(self):
        url = urlparse(self.path)
        if url.path in ("/", "/index.html"):
            return self.send_file(STATIC / "index.html")
        if url.path.startswith("/static/"):
            f = (STATIC / url.path.removeprefix("/static/")).resolve()
            if f.parent == STATIC.resolve() and f.is_file():
                return self.send_file(f)
        if url.path == "/api/job":
            if not self.authorized():
                return
            job_id = parse_qs(url.query).get("id", [""])[0]
            with jobs_lock:
                job = dict(jobs.get(job_id) or {}, log=list((jobs.get(job_id) or {}).get("log", [])))
            if not job:
                return self.send_json({"error": "找不到這個工作"}, HTTPStatus.NOT_FOUND)
            job.pop("started", None)
            return self.send_json(job)
        if url.path == "/api/sample":
            saved = json.loads(SAMPLE.read_text(encoding="utf-8"))
            img = load_image(SAMPLE_IMG)
            return self.send_json({"result": page_data(img, saved["plan"], saved["located"], saved["shapes"]),
                                   "image": to_data_url(img)})
        self.send_json({"error": "not found"}, HTTPStatus.NOT_FOUND)

    def do_POST(self):
        if urlparse(self.path).path != "/api/analyze":
            return self.send_json({"error": "not found"}, HTTPStatus.NOT_FOUND)
        if not self.authorized():
            return
        length = int(self.headers.get("Content-Length") or 0)
        if not 0 < length <= MAX_UPLOAD:
            return self.send_json({"error": "照片大小不對（上限 15 MB）"}, HTTPStatus.BAD_REQUEST)
        photo = self.rfile.read(length)
        job_id = uuid.uuid4().hex[:8]
        with jobs_lock:
            jobs[job_id] = {"status": "running", "log": [], "started": time.time()}
        threading.Thread(target=run_job, args=(job_id, photo, self.server.api_key), daemon=True).start()
        self.send_json({"job": job_id})

    def log_message(self, fmt, *args):
        if "/api/job" not in self.path:  # 輪詢太頻繁，不印
            super().log_message(fmt, *args)


def lan_ip() -> str:
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        try:
            s.connect(("192.0.2.1", 80))  # 不會真的送出封包，只是讓系統選出對外的網卡
            return s.getsockname()[0]
        except OSError:
            return "127.0.0.1"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()
    server = ThreadingHTTPServer(("0.0.0.0", args.port), Handler)
    server.api_key = load_api_key()
    print(f"WORLD OVERLAY 伺服器啟動：手機連同一個 Wi-Fi，打開 http://{lan_ip()}:{args.port}")
    if os.environ.get("APP_PASSCODE"):
        print("已啟用通行碼（APP_PASSCODE）")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
