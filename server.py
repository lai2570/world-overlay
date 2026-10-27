"""手機 App 的後端：收手機拍的照片 → 跑組裝分析 → 回傳引導畫面資料。API key 只放在這台電腦。

用法：
  .venv/Scripts/python server.py            # 手機連同一個 Wi-Fi，開 http://<電腦IP>:8000
  .venv/Scripts/python server.py --port 9000

分析要 1 分鐘左右，所以用「建立工作 → 輪詢進度」：
  POST /api/assemble  JSON {parts: [data URL...], manual: data URL 或 null} → {"job": id}
                      一個零件一張照片或一段轉一圈的影片（data:video/...）→ 組裝動畫（parts_guide.py）；
                      去背完就有初步結果，之後每一步的圖陸續補上
  GET  /api/job?id=&v= → {"status": running|done|error, "log": [...], "version": n, "result": {...}}
                      result 只在 version 比 v 新時才附上（結果裡有圖片，避免每秒重傳）
  GET  /api/parts_sample → 不呼叫 API，回傳 out/parts_sample.json 的範例結果
  舊版「一張照片」流程（static/photo.html）：
  POST /api/analyze   body 為 JPEG，或 JSON {image, manual}（都是 data URL，manual 可省略）→ {"job": id}
  GET  /api/sample    → 不呼叫 API，直接回傳 out/mic_parts_guide.json 的範例結果
設定環境變數 APP_PASSCODE 時，請求要帶 X-Passcode 標頭（放到網路上時避免別人花你的額度）。
"""
import argparse
import base64
import copy
import io
import json
import os
import shutil
import socket
import ssl
import subprocess
import threading
import time
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import parts_guide
from clip import load_clip
from build_guide import MANUAL_MAX_SIDE, STATIC, analyze, page_data
from test_grounding import ROOT, load_api_key, load_image, to_data_url
from tracker import track

MAX_UPLOAD = 150 * 1024 * 1024  # 最多 10 個零件（照片或短片）+ 說明書
MAX_PARTS = 10
APP_OUT = ROOT / "out" / "app"
SAMPLE = ROOT / "out" / "mic_parts_guide.json"
SAMPLE_IMG = ROOT / "samples" / "mic_parts.jpg"
PARTS_SAMPLE = ROOT / "out" / "parts_sample.json"
CONTENT_TYPES = {".html": "text/html; charset=utf-8", ".css": "text/css; charset=utf-8",
                 ".js": "text/javascript; charset=utf-8"}

jobs: dict[str, dict] = {}
jobs_lock = threading.Lock()


def run_job(job_id: str, photo: bytes, key: str, manual: bytes | None = None):
    job = jobs[job_id]

    def log(msg: str):
        with jobs_lock:
            job["log"].append(msg)
        print(f"[{job_id}] {msg}")

    try:
        img = load_image(io.BytesIO(photo))
        manual_img = load_image(io.BytesIO(manual), MANUAL_MAX_SIDE) if manual else None
        plan, located, shapes = analyze(key, img, log=log, manual=manual_img)
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


def run_assemble(job_id: str, photos: list[tuple[bytes, bool]], manual: bytes | None, key: str):
    job = jobs[job_id]

    def log(msg: str):
        with jobs_lock:
            job["log"].append(msg)
        print(f"[{job_id}] {msg}")

    def publish(result: dict):
        snap = copy.deepcopy(result)
        with jobs_lock:
            job["result"] = snap
            job["version"] += 1

    try:
        if any(v for _, v in photos):
            log("處理零件影片（轉檔、上傳）")
        with ThreadPoolExecutor(4) as ex:  # 影片要轉檔和上傳，同時做
            imgs = list(ex.map(lambda p: load_clip(p[0], key) if p[1] else load_image(io.BytesIO(p[0])), photos))
        manual_img = load_image(io.BytesIO(manual), MANUAL_MAX_SIDE) if manual else None
        result = parts_guide.analyze(key, imgs, manual_img, log=log, publish=publish)
        publish(result)
        APP_OUT.mkdir(parents=True, exist_ok=True)
        (APP_OUT / f"{job_id}_parts.json").write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")
        with jobs_lock:
            job.update(status="done", seconds=round(time.time() - job["started"]))
    except Exception as e:  # 回報給手機，不讓工作卡在 running
        traceback.print_exc()
        with jobs_lock:
            job.update(status="error", error=str(e) or e.__class__.__name__)


def decode_data_url(url: str) -> bytes:
    return base64.b64decode(url.split(",", 1)[1])


class Handler(BaseHTTPRequestHandler):
    server_version = "WorldOverlay/0.1"

    def send_json(self, obj, status=HTTPStatus.OK):
        body = json.dumps(obj, ensure_ascii=False).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        try:
            self.wfile.write(body)
        except (ConnectionError, ssl.SSLError):  # 手機已經關掉頁面或切走，不用回了
            pass

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
            q = parse_qs(url.query)
            job_id, since = q.get("id", [""])[0], int(q.get("v", ["0"])[0] or 0)
            with jobs_lock:
                job = dict(jobs.get(job_id) or {}, log=list((jobs.get(job_id) or {}).get("log", [])))
            if not job:
                return self.send_json({"error": "找不到這個工作"}, HTTPStatus.NOT_FOUND)
            job.pop("started", None)
            if job.get("version", 0) <= since and since:
                job.pop("result", None)  # 手機已經有這一版
            return self.send_json(job)
        if url.path == "/api/parts_sample":
            return self.send_json({"result": json.loads(PARTS_SAMPLE.read_text(encoding="utf-8"))})
        if url.path == "/api/sample":
            saved = json.loads(SAMPLE.read_text(encoding="utf-8"))
            img = load_image(SAMPLE_IMG)
            return self.send_json({"result": page_data(img, saved["plan"], saved["located"], saved["shapes"]),
                                   "image": to_data_url(img)})
        self.send_json({"error": "not found"}, HTTPStatus.NOT_FOUND)

    def do_POST(self):
        path = urlparse(self.path).path
        if path not in ("/api/analyze", "/api/track", "/api/assemble"):
            return self.send_json({"error": "not found"}, HTTPStatus.NOT_FOUND)
        if not self.authorized():
            return
        length = int(self.headers.get("Content-Length") or 0)
        if not 0 < length <= MAX_UPLOAD:
            return self.send_json({"error": f"上傳大小不對（上限 {MAX_UPLOAD // 1024 // 1024} MB）"}, HTTPStatus.BAD_REQUEST)
        if path == "/api/track":
            return self.handle_track(self.rfile.read(length))
        if path == "/api/assemble":
            return self.handle_assemble(self.rfile.read(length))
        body = self.rfile.read(length)
        if self.headers.get("Content-Type", "").startswith("application/json"):
            # {image: 零件照 data URL, manual: 說明書照 data URL 或 null}
            try:
                req = json.loads(body)
                photo = base64.b64decode(req["image"].split(",", 1)[1])
                manual = base64.b64decode(req["manual"].split(",", 1)[1]) if req.get("manual") else None
            except (ValueError, KeyError, IndexError):
                return self.send_json({"error": "上傳格式不對"}, HTTPStatus.BAD_REQUEST)
        else:
            photo, manual = body, None
        job_id = uuid.uuid4().hex[:8]
        with jobs_lock:
            jobs[job_id] = {"status": "running", "log": [], "started": time.time()}
        threading.Thread(target=run_job, args=(job_id, photo, self.server.api_key, manual), daemon=True).start()
        self.send_json({"job": job_id})

    def handle_assemble(self, body: bytes):
        try:
            req = json.loads(body)
            photos = [(decode_data_url(u), u.startswith("data:video/")) for u in req["parts"]]
            manual = decode_data_url(req["manual"]) if req.get("manual") else None
        except (ValueError, KeyError, IndexError, TypeError):
            return self.send_json({"error": "上傳格式不對"}, HTTPStatus.BAD_REQUEST)
        if not 2 <= len(photos) <= MAX_PARTS:
            return self.send_json({"error": f"零件照片要 2 到 {MAX_PARTS} 張"}, HTTPStatus.BAD_REQUEST)
        job_id = uuid.uuid4().hex[:8]
        with jobs_lock:
            jobs[job_id] = {"status": "running", "log": [], "started": time.time(), "version": 0}
        threading.Thread(target=run_assemble, args=(job_id, photos, manual, self.server.api_key), daemon=True).start()
        self.send_json({"job": job_id})

    def handle_track(self, body: bytes):
        """同步回應（約 2–3 秒；有零件跟丟時約 8 秒）。body: {image: data URL, parts: [...]}"""
        try:
            req = json.loads(body)
            b64 = req["image"].split(",", 1)[1]
            img = load_image(io.BytesIO(base64.b64decode(b64)))
            result = track(self.server.api_key, img, req["parts"])
        except Exception as e:
            traceback.print_exc()
            return self.send_json({"error": str(e) or e.__class__.__name__}, HTTPStatus.INTERNAL_SERVER_ERROR)
        note = f"，重新找：{'、'.join(result['relocated'])}" if result["relocated"] else ""
        note += f"，沿用上次輪廓：{'、'.join(result['kept'])}" if result["kept"] else ""
        note += f"，找不到：{'、'.join(result['lost'])}" if result["lost"] else ""
        sent = "、".join(f"{p['name']}{'' if p.get('box') else '(無框)'}" for p in req["parts"])
        print(f"[track] {result['seconds']} 秒｜{sent}｜描到 {len(result['shapes'])} 個{note}")
        self.send_json(result)

    def log_message(self, fmt, *args):
        if "/api/track" in self.path:  # handle_track 自己印比較精簡的紀錄
            return
        if "/api/job" not in self.path:  # 輪詢太頻繁，不印
            super().log_message(fmt, *args)


def lan_ip() -> str:
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        try:
            s.connect(("192.0.2.1", 80))  # 不會真的送出封包，只是讓系統選出對外的網卡
            return s.getsockname()[0]
        except OSError:
            return "127.0.0.1"


CERTS = ROOT / "certs"


def ensure_cert(ip: str) -> tuple[Path, Path]:
    """區網用的自簽憑證（手機要 HTTPS 才能在網頁裡開即時相機）。IP 變了就重新產生。"""
    cert, key, stamp = CERTS / "cert.pem", CERTS / "key.pem", CERTS / "ip.txt"
    if cert.exists() and key.exists() and stamp.exists() and stamp.read_text() == ip:
        return cert, key
    openssl = shutil.which("openssl") or r"C:\Program Files\Git\usr\bin\openssl.exe"
    CERTS.mkdir(exist_ok=True)
    subprocess.run([openssl, "req", "-x509", "-newkey", "rsa:2048", "-nodes", "-days", "30",
                    "-keyout", str(key), "-out", str(cert), "-subj", "/CN=world-overlay",
                    "-addext", f"subjectAltName=IP:{ip},IP:127.0.0.1,DNS:localhost"],
                   check=True, capture_output=True)
    stamp.write_text(ip)
    return cert, key


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--https", action="store_true", help="用自簽憑證開 HTTPS（即時相機需要）")
    args = ap.parse_args()
    server = ThreadingHTTPServer(("0.0.0.0", args.port), Handler)
    server.api_key = load_api_key()
    ip = lan_ip()
    scheme = "http"
    if args.https:
        cert, key = ensure_cert(ip)
        ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        ctx.load_cert_chain(cert, key)
        server.socket = ctx.wrap_socket(server.socket, server_side=True)
        scheme = "https"
    print(f"WORLD OVERLAY 伺服器啟動：手機連同一個 Wi-Fi，打開 {scheme}://{ip}:{args.port}")
    if args.https:
        print("第一次打開會出現安全性警告（自簽憑證），點「進階」→「繼續前往」即可")
    if os.environ.get("APP_PASSCODE"):
        print("已啟用通行碼（APP_PASSCODE）")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
