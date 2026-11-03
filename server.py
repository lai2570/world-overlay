"""手機 App 的後端：收手機拍的照片 → 跑組裝分析 → 回傳引導畫面資料。API key 只放在這台電腦。

用法：
  .venv/Scripts/python server.py            # 手機連同一個 Wi-Fi，開 http://<電腦IP>:8000
  .venv/Scripts/python server.py --port 9000

分析要 1 分鐘左右，所以用「建立工作 → 輪詢進度」：
  POST /api/assemble  JSON {parts: [data URL...], manual: [data URL...] 或 null} → {"job": id}
                      manual 是說明書的每一頁（照片，或手機把 PDF 每頁畫成的圖），最多 MAX_MANUAL_PAGES 頁
                      一個零件一張照片或一段轉一圈的影片（data:video/...）→ 組裝動畫（parts_guide.py）；
                      去背完就有初步結果，之後每一步的圖陸續補上
  POST /api/video     body 是影片原檔（不轉 base64，省 1/3）→ {"video": id}；一段影片錄全部零件時用
  POST /api/assemble  JSON {video: id, manual} → 先切成每個零件一段（video_split.py），後面一樣
  GET  /api/job?id=&v= → {"status": running|done|error, "log": [...], "version": n, "result": {...}}
                      result 只在 version 比 v 新時才附上（結果裡有圖片，避免每秒重傳）
  GET  /api/parts_sample?lang=en → 不呼叫 API，回傳 out/parts_sample.json 的範例結果（en：照 sample_en.json 翻成英文，工作 id 是 sample_en）
語言：手機帶 X-Lang: en 標頭時錯誤訊息用英文；/api/assemble 的 JSON 帶 lang: "en" 時 AI 用英文寫步驟和進度。沒帶是中文。
  GET  /api/models?job= → {"on": bool, "parts": {零件: pending|ready|failed}}；每個零件的 3D 模型（model3d.py，
                      去背完就開始在背景做，一個約 60 秒）。舊的工作和範例第一次問到時才做
  GET  /api/model?job=&part= → 那個零件的 glb
  POST /api/identify  JSON {job: id 或 "sample", image: 鏡頭畫面 data URL} → {"top": [[零件, 分數]...], "sure": bool}
                      即時辨識鏡頭對著哪個零件（identify.py，DINOv2 在這台電腦上跑，不花 API 額度）
  POST /api/confirm   JSON 同上 → {"name": 零件或 "", "reason"}；不確定時用 Muse Spark 確認（約 3–5 秒）
  舊版「一張照片」流程（static/photo.html）：
  POST /api/analyze   body 為 JPEG，或 JSON {image, manual}（都是 data URL，manual 可省略）→ {"job": id}
  GET  /api/sample    → 不呼叫 API，直接回傳 out/mic_parts_guide.json 的範例結果
通行碼（放到網路上時避免別人花你的額度）：請求要帶 X-Passcode 標頭。
  一人一組：.venv/Scripts/python server.py --add-user 小明   → 印出他的通行碼，存在 users.json（不進 git）
            .venv/Scripts/python server.py --remove-user 小明
  環境變數 APP_PASSCODE 是自己用的通行碼，不限次數。
  有設定任何通行碼時：每人每天最多 APP_DAILY_RUNS 次分析（預設 10，用量記在 out/usage.json），
  全部人同時最多 MAX_RUNNING 個分析在跑。都沒設定時不檢查（只在家裡 Wi-Fi 用）。
"""
import argparse
import base64
import copy
import hmac
import io
import json
import os
import secrets
import shutil
import socket
import ssl
import subprocess
import threading
import time
import traceback
import urllib.request
import uuid
from concurrent.futures import ThreadPoolExecutor
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import parts_guide
from clip import load_clip
from video_split import split_video
from build_guide import MANUAL_MAX_SIDE, STATIC, analyze, page_data
from test_grounding import ROOT, load_api_key, load_image, to_data_url
from tracker import track
import identify

MAX_UPLOAD = 150 * 1024 * 1024  # 最多 10 個零件（照片或短片）+ 說明書
MAX_PARTS = 10
MAX_MANUAL_PAGES = 20  # 和手機端一致
MAX_VIDEO = 500 * 1024 * 1024  # 一段影片錄全部零件：100 秒 1080p 約 100–200 MB
MAX_VIDEO_PARTS = 16
MAX_PENDING_VIDEOS = 4  # 上傳了還沒分析的影片最多留幾段（記憶體）
videos: dict[str, bytes] = {}  # 上傳好、等 /api/assemble 取用的整段影片
USERS_FILE = ROOT / "users.json"
USAGE_FILE = ROOT / "out" / "usage.json"
OWNER = "owner"  # APP_PASSCODE 的使用者，不限次數
DAILY_RUNS = int(os.environ.get("APP_DAILY_RUNS", "10"))
DEPLOY_TOKEN = os.environ.get("DEPLOY_TOKEN", "")  # autodeploy.py 給的，用來問現在能不能重開
MAX_RUNNING = 2  # Muse Image 每分鐘 150 張、Spark 也有限制，同時太多人會互相拖慢
APP_OUT = ROOT / "out" / "app"
SAMPLE = ROOT / "out" / "mic_parts_guide.json"
SAMPLE_IMG = ROOT / "samples" / "mic_parts.jpg"
PARTS_SAMPLE = ROOT / "out" / "parts_sample.json"
SAMPLE_EN = ROOT / "sample_en.json"  # 範例的英文翻譯（零件名稱、目標、每一步的說明）
SAMPLES = ("sample", "sample_en")
CONTENT_TYPES = {".html": "text/html; charset=utf-8", ".css": "text/css; charset=utf-8",
                 ".js": "text/javascript; charset=utf-8", ".glb": "model/gltf-binary"}

jobs: dict[str, dict] = {}
jobs_lock = threading.Lock()


def load_users() -> dict[str, str]:
    """{名字: 通行碼}"""
    users = json.loads(USERS_FILE.read_text(encoding="utf-8")) if USERS_FILE.exists() else {}
    if os.environ.get("APP_PASSCODE"):
        users[OWNER] = os.environ["APP_PASSCODE"]
    return users


def tr(lang: str, zh: str, en: str) -> str:
    return en if lang == "en" else zh


def take_run(user: str | None, lang: str = "zh") -> str | None:
    """開始一次分析前檢查額度，可以就記一次；不行回傳原因。"""
    if user is None:  # 沒設定通行碼
        return None
    with jobs_lock:
        if sum(j["status"] == "running" for j in jobs.values()) >= MAX_RUNNING:
            return tr(lang, "目前有其他人在分析，請 1–2 分鐘後再試", "Someone else is analyzing right now, try again in 1–2 minutes")
        usage = json.loads(USAGE_FILE.read_text(encoding="utf-8")) if USAGE_FILE.exists() else {}
        today = usage.setdefault(time.strftime("%Y-%m-%d"), {})
        if user != OWNER and today.get(user, 0) >= DAILY_RUNS:
            return tr(lang, f"今天的 {DAILY_RUNS} 次分析已經用完，明天再試",
                      f"You have used all {DAILY_RUNS} analyses for today, try again tomorrow")
        today[user] = today.get(user, 0) + 1
        USAGE_FILE.parent.mkdir(parents=True, exist_ok=True)
        USAGE_FILE.write_text(json.dumps(usage, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[{user}] 今天第 {today[user]} 次分析")
    return None


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


def run_assemble(job_id: str, photos: list[tuple[bytes, bool]], manual: list[bytes], key: str,
                 video: bytes | None = None, lang: str = "zh"):
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
            first = job["version"] == 1
        if first:  # 去背圖一好就先算即時辨識要用的零件特徵，使用者按「對準零件」時不用等
            threading.Thread(target=part_index, args=(job_id,), daemon=True).start()
            if model3d_on():
                threading.Thread(target=make_models, args=(job_id, snap), daemon=True).start()

    try:
        if video:
            imgs = split_video(key, video, log, lang)
            if len(imgs) < 2:
                raise RuntimeError(tr(lang, f"影片裡只找到 {len(imgs)} 個零件，請一次拿一個、每個轉幾秒",
                                      f"Only found {len(imgs)} part(s) in the video; film the parts one at a time"))
            imgs = imgs[:MAX_VIDEO_PARTS]
        else:
            if any(v for _, v in photos):
                log(tr(lang, "處理零件影片（轉檔、上傳）", "Processing the part videos"))
            with ThreadPoolExecutor(4) as ex:  # 影片要轉檔和上傳，同時做
                imgs = list(ex.map(lambda p: load_clip(p[0], key) if p[1] else load_image(io.BytesIO(p[0])), photos))
        # 每個零件用的那一格存下來，去背出錯時可以拿來重跑
        frames_dir = APP_OUT / f"{job_id}_parts"
        frames_dir.mkdir(parents=True, exist_ok=True)
        for n, im in enumerate(imgs, 1):
            getattr(im, "photo", im).save(frames_dir / f"{n}.jpg", quality=92)
        manual_imgs = [load_image(io.BytesIO(m), MANUAL_MAX_SIDE) for m in manual]
        for n, im in enumerate(manual_imgs, 1):
            im.save(frames_dir / f"manual{n}.jpg", quality=92)
        result = parts_guide.analyze(key, imgs, manual_imgs, log=log, publish=publish, lang=lang)
        publish(result)
        APP_OUT.mkdir(parents=True, exist_ok=True)
        (APP_OUT / f"{job_id}_parts.json").write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")
        with jobs_lock:
            job.update(status="done", seconds=round(time.time() - job["started"]))
    except Exception as e:  # 回報給手機，不讓工作卡在 running
        traceback.print_exc()
        with jobs_lock:
            job.update(status="error", error=str(e) or e.__class__.__name__)


CONFIRM_GAP = 2.0  # 每人兩次 Spark 確認之間至少隔幾秒
indexes: dict[str, identify.PartIndex] = {}
index_lock = threading.Lock()
infer_lock = threading.Lock()  # CPU 一次算一張就好，同時算反而每張都變慢
last_confirm: dict[str, float] = {}

CONFIRM_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["name", "reason"],
    "properties": {
        "name": {"type": "string", "description": "畫面中間那個零件的名稱，必須是清單裡的其中一個；看不出來或不在清單裡給空字串"},
        "reason": {"type": "string"},
    },
}


def job_result(job: str) -> dict | None:
    """分析結果：範例、記憶體裡的工作，或伺服器重開過就讀存檔。"""
    if job == "sample":
        return json.loads(PARTS_SAMPLE.read_text(encoding="utf-8"))
    if job == "sample_en":
        return english_sample()
    with jobs_lock:
        r = (jobs.get(job) or {}).get("result")
    if r:
        return r
    f = APP_OUT / f"{job}_parts.json"
    if job.isalnum() and f.exists():
        return json.loads(f.read_text(encoding="utf-8"))
    return None


def english_sample() -> dict:
    """範例換成英文：零件名稱（也是圖片的 key）、目標、每一步的說明和理由。"""
    r = json.loads(PARTS_SAMPLE.read_text(encoding="utf-8"))
    en = json.loads(SAMPLE_EN.read_text(encoding="utf-8"))
    names = en["names"]
    r["goal"] = en["goal"]
    for s, t in zip(r["steps"], en["steps"]):
        s.update(t, part_a=names[s["part_a"]], part_b=names[s["part_b"]])
        for k in ("a_img", "b_img"):
            if (s[k] or "").startswith("part:"):
                s[k] = "part:" + names[s[k][5:]]
    r["images"] = {("part:" + names[k[5:]] if k.startswith("part:") else k): v for k, v in r["images"].items()}
    return r


def part_index(job: str) -> identify.PartIndex | None:
    """第一次用到時建（每個零件算一次特徵，16 個約 5–10 秒），之後記住。"""
    with index_lock:
        if job in indexes:
            return indexes[job]
        r = job_result(job)
        if r is None:
            return None
        with infer_lock:
            indexes[job] = identify.PartIndex.from_result(r)
        return indexes[job]


def decode_data_url(url: str) -> bytes:
    return base64.b64decode(url.split(",", 1)[1])


MODEL3D = os.environ.get("MODEL3D_URL", "http://127.0.0.1:8766")
models: dict[str, dict[str, str]] = {}  # 工作 → {零件: pending|ready|failed}，做的過程中用；做完的看存檔
models_lock = threading.Lock()


def model3d_on() -> bool:
    """3D 模型服務有沒有開。服務一次只處理一個請求，所以只看連不連得上，不等它回應。"""
    u = urlparse(MODEL3D)
    try:
        socket.create_connection((u.hostname, u.port), timeout=1).close()
        return True
    except OSError:
        return False


def models_dir(job: str) -> Path:
    return ROOT / "out" / f"parts_{job}_3d" if job in SAMPLES else APP_OUT / f"{job}_3d"


def saved_models(job: str) -> dict[str, str | None]:
    """{零件: glb 檔名，生成失敗是 None}；還沒做的零件不在裡面"""
    f = models_dir(job) / "models.json"
    return json.loads(f.read_text(encoding="utf-8")) if f.exists() else {}


def model_state(job: str) -> dict[str, str] | None:
    with models_lock:
        if job in models:
            return dict(models[job])
    saved = saved_models(job)
    return {n: "ready" if f else "failed" for n, f in saved.items()} or None


def make_models(job: str, result: dict):
    """每個零件的去背圖送給 model3d.py，一個做完就存一個（手機上先做好的先看得到）。"""
    names = [k[5:] for k in result["images"] if k.startswith("part:")]
    with models_lock:
        if job in models:
            return
        models[job] = {n: "pending" for n in names}
    out = models_dir(job)
    out.mkdir(parents=True, exist_ok=True)
    manifest = {}
    for n, name in enumerate(names, 1):
        req = urllib.request.Request(f"{MODEL3D}/shape", data=decode_data_url(result["images"][f"part:{name}"]))
        t = time.time()
        try:
            with urllib.request.urlopen(req, timeout=900) as r:  # 前面排了別人的零件時要等比較久
                (out / f"{n}.glb").write_bytes(r.read())
            manifest[name] = f"{n}.glb"
        except OSError as e:
            print(f"[{job}] 3D 模型失敗：{name}（{e}）")
            manifest[name] = None
        print(f"[{job}] 3D 模型 {n}/{len(names)} {name} {time.time() - t:.0f} 秒")
        (out / "models.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8")
        with models_lock:
            models[job][name] = "ready" if manifest[name] else "failed"


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

    @property
    def lang(self) -> str:
        return "en" if self.headers.get("X-Lang") == "en" else "zh"

    def t(self, zh: str, en: str) -> str:
        return tr(self.lang, zh, en)

    def authorized(self) -> bool:
        """通行碼對了，self.user 是使用者名字（沒設定通行碼時是 None）。"""
        users = load_users()
        self.user = None
        if not users:
            return True
        sent = self.headers.get("X-Passcode", "").strip().encode()
        for name, code in users.items():
            if hmac.compare_digest(sent, code.encode()):
                self.user = name
                return True
        time.sleep(1)  # 放到網路上時拖慢猜通行碼
        self.send_json({"error": self.t("通行碼錯誤", "Wrong passcode")}, HTTPStatus.UNAUTHORIZED)
        return False

    def do_GET(self):
        url = urlparse(self.path)
        if url.path in ("/", "/index.html"):
            return self.send_file(STATIC / "index.html")
        if url.path.startswith("/static/"):
            f = (STATIC / url.path.removeprefix("/static/")).resolve()
            if f.parent == STATIC.resolve() and f.is_file():
                return self.send_file(f)
        if url.path == "/api/busy" and DEPLOY_TOKEN:  # 只有 autodeploy.py 知道 token（通道進來的也是 127.0.0.1，不能只看 IP）
            if not hmac.compare_digest(self.headers.get("X-Deploy-Token", ""), DEPLOY_TOKEN):
                return self.send_json({"error": "not found"}, HTTPStatus.NOT_FOUND)
            with jobs_lock:
                return self.send_json({"running": sum(j["status"] == "running" for j in jobs.values())})
        if url.path == "/api/ping":  # 上傳大影片前先確認通行碼
            return self.authorized() and self.send_json({"ok": True})
        if url.path == "/api/job":
            if not self.authorized():
                return
            q = parse_qs(url.query)
            job_id, since = q.get("id", [""])[0], int(q.get("v", ["0"])[0] or 0)
            with jobs_lock:
                job = dict(jobs.get(job_id) or {}, log=list((jobs.get(job_id) or {}).get("log", [])))
            if not job:
                return self.send_json({"error": self.t("找不到這個工作", "Job not found")}, HTTPStatus.NOT_FOUND)
            job.pop("started", None)
            if job.get("version", 0) <= since and since:
                job.pop("result", None)  # 手機已經有這一版
            return self.send_json(job)
        if url.path in ("/api/models", "/api/model"):
            if not self.authorized():
                return
            q = parse_qs(url.query)
            job = q.get("job", [""])[0]
            if not (job in SAMPLES or job.isalnum()):
                return self.send_json({"error": self.t("找不到這次分析的結果", "Result not found")}, HTTPStatus.NOT_FOUND)
            if url.path == "/api/model":
                f = saved_models(job).get(q.get("part", [""])[0])
                if not f:
                    return self.send_json({"error": self.t("這個零件還沒有 3D 模型", "No 3D model for this part yet")}, HTTPStatus.NOT_FOUND)
                return self.send_file(models_dir(job) / f)
            state, on = model_state(job), model3d_on()
            if state is None and on and (r := job_result(job)):  # 舊的工作或範例：第一次問到才做
                threading.Thread(target=make_models, args=(job, r), daemon=True).start()
                state = {k[5:]: "pending" for k in r["images"] if k.startswith("part:")}
            return self.send_json({"on": on or bool(state), "parts": state or {}})
        if url.path == "/api/parts_sample":
            en = parse_qs(url.query).get("lang") == ["en"]
            return self.send_json({"result": english_sample() if en else json.loads(PARTS_SAMPLE.read_text(encoding="utf-8")),
                                   "job": "sample_en" if en else "sample"})
        if url.path == "/api/sample":
            saved = json.loads(SAMPLE.read_text(encoding="utf-8"))
            img = load_image(SAMPLE_IMG)
            return self.send_json({"result": page_data(img, saved["plan"], saved["located"], saved["shapes"]),
                                   "image": to_data_url(img)})
        self.send_json({"error": "not found"}, HTTPStatus.NOT_FOUND)

    def do_POST(self):
        path = urlparse(self.path).path
        if path not in ("/api/analyze", "/api/track", "/api/assemble", "/api/video", "/api/identify", "/api/confirm"):
            return self.send_json({"error": "not found"}, HTTPStatus.NOT_FOUND)
        if not self.authorized():
            return
        length = int(self.headers.get("Content-Length") or 0)
        if path == "/api/video":
            if not 0 < length <= MAX_VIDEO:
                return self.send_json({"error": self.t(f"影片大小不對（上限 {MAX_VIDEO // 1024 // 1024} MB）",
                                                       f"Video size not allowed (max {MAX_VIDEO // 1024 // 1024} MB)")},
                                      HTTPStatus.BAD_REQUEST)
            vid = uuid.uuid4().hex[:8]
            data = self.rfile.read(length)
            with jobs_lock:
                videos[vid] = data
                while len(videos) > MAX_PENDING_VIDEOS:  # 傳了不分析的：丟最舊的
                    videos.pop(next(iter(videos)))
            return self.send_json({"video": vid})
        if not 0 < length <= MAX_UPLOAD:
            return self.send_json({"error": self.t(f"上傳大小不對（上限 {MAX_UPLOAD // 1024 // 1024} MB）",
                                                   f"Upload size not allowed (max {MAX_UPLOAD // 1024 // 1024} MB)")},
                                  HTTPStatus.BAD_REQUEST)
        if path == "/api/track":
            return self.handle_track(self.rfile.read(length))
        if path in ("/api/identify", "/api/confirm"):
            return self.handle_identify(self.rfile.read(length), path == "/api/confirm")
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
                return self.send_json({"error": self.t("上傳格式不對", "Bad upload format")}, HTTPStatus.BAD_REQUEST)
        else:
            photo, manual = body, None
        if why := take_run(self.user, self.lang):
            return self.send_json({"error": why}, HTTPStatus.TOO_MANY_REQUESTS)
        job_id = uuid.uuid4().hex[:8]
        with jobs_lock:
            jobs[job_id] = {"status": "running", "log": [], "started": time.time()}
        threading.Thread(target=run_job, args=(job_id, photo, self.server.api_key, manual), daemon=True).start()
        self.send_json({"job": job_id})

    def handle_assemble(self, body: bytes):
        try:
            req = json.loads(body)
            photos = [(decode_data_url(u), u.startswith("data:video/")) for u in req.get("parts") or []]
            pages = req.get("manual") or []
            manual = [decode_data_url(u) for u in ([pages] if isinstance(pages, str) else pages)]  # 舊版手機送一張字串
        except (ValueError, KeyError, IndexError, TypeError, AttributeError):
            return self.send_json({"error": self.t("上傳格式不對", "Bad upload format")}, HTTPStatus.BAD_REQUEST)
        if len(manual) > MAX_MANUAL_PAGES:
            return self.send_json({"error": self.t(f"說明書最多 {MAX_MANUAL_PAGES} 頁",
                                                   f"The manual can have at most {MAX_MANUAL_PAGES} pages")},
                                  HTTPStatus.BAD_REQUEST)
        video = None
        if req.get("video"):
            with jobs_lock:
                video = videos.pop(str(req["video"]), None)
            if video is None:
                return self.send_json({"error": self.t("找不到上傳的影片，請重新上傳", "Uploaded video not found, please upload again")},
                                      HTTPStatus.BAD_REQUEST)
        elif not 2 <= len(photos) <= MAX_PARTS:
            return self.send_json({"error": self.t(f"零件照片要 2 到 {MAX_PARTS} 張", f"Add 2 to {MAX_PARTS} part photos")},
                                  HTTPStatus.BAD_REQUEST)
        if why := take_run(self.user, self.lang):
            if video is not None:  # 留著，等一下再按一次不用重新上傳
                with jobs_lock:
                    videos[str(req["video"])] = video
            return self.send_json({"error": why}, HTTPStatus.TOO_MANY_REQUESTS)
        job_id = uuid.uuid4().hex[:8]
        with jobs_lock:
            jobs[job_id] = {"status": "running", "log": [], "started": time.time(), "version": 0}
        lang = "en" if req.get("lang") == "en" else "zh"
        threading.Thread(target=run_assemble, args=(job_id, photos, manual, self.server.api_key, video, lang),
                         daemon=True).start()
        self.send_json({"job": job_id})

    def handle_identify(self, body: bytes, confirm: bool):
        try:
            req = json.loads(body)
            job = str(req["job"])
            frame = load_image(io.BytesIO(decode_data_url(req["image"])), 640)
        except (ValueError, KeyError, IndexError, TypeError, AttributeError, OSError):
            return self.send_json({"error": self.t("上傳格式不對", "Bad upload format")}, HTTPStatus.BAD_REQUEST)
        if not confirm:
            idx = part_index(job)
            if idx is None:
                return self.send_json({"error": self.t("找不到這次分析的結果", "Result not found")}, HTTPStatus.NOT_FOUND)
            t = time.perf_counter()
            with infer_lock:
                top = idx.match(frame)
            sure = len(top) < 2 or top[0][1] - top[1][1] >= identify.SURE_MARGIN
            return self.send_json({"top": [[n, round(s, 2)] for n, s in top[:3]], "sure": sure,
                                   "ms": round((time.perf_counter() - t) * 1000)})
        result = job_result(job)
        if result is None:
            return self.send_json({"error": self.t("找不到這次分析的結果", "Result not found")}, HTTPStatus.NOT_FOUND)
        who = self.user or "local"
        now = time.time()
        with jobs_lock:
            if now - last_confirm.get(who, 0) < CONFIRM_GAP:
                return self.send_json({"error": self.t("太頻繁", "Too many requests")}, HTTPStatus.TOO_MANY_REQUESTS)
            last_confirm[who] = now
        looks = result.get("looks") or {}
        names = [k[5:] for k in result["images"] if k.startswith("part:")]
        listing = "\n".join(f"- {n}" + (f"（{looks[n]}）" if looks.get(n) else "") for n in names)
        hint = "、".join(str(n) for n in req.get("candidates") or [])
        prompt = ("這是使用者手機鏡頭的畫面，正對著一個準備組裝的零件。從下面的零件清單裡判斷畫面中間那個是哪一個。"
                  + (f"影像比對覺得可能是：{hint}。" if hint else "") + "\n零件清單：\n" + listing)
        try:
            out = parts_guide.spark_json(self.server.api_key, [{"type": "text", "text": prompt},
                                                               parts_guide.image_part(frame)],
                                         CONFIRM_SCHEMA, "WhichPart", "low")
        except RuntimeError as e:
            return self.send_json({"error": str(e)}, HTTPStatus.BAD_GATEWAY)
        name = out["name"] if out["name"] in names else ""
        return self.send_json({"name": name, "reason": out["reason"]})

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
    ap.add_argument("--add-user", metavar="名字", help="產生這個人的通行碼（已經有就換一組）")
    ap.add_argument("--remove-user", metavar="名字")
    args = ap.parse_args()
    if args.add_user or args.remove_user:
        users = json.loads(USERS_FILE.read_text(encoding="utf-8")) if USERS_FILE.exists() else {}
        if args.add_user:
            users[args.add_user] = secrets.token_hex(6)
            print(f"{args.add_user} 的通行碼：{users[args.add_user]}")
        elif users.pop(args.remove_user, None) is None:
            print(f"沒有 {args.remove_user}")
        else:
            print(f"已移除 {args.remove_user}（伺服器不用重開）")
        USERS_FILE.write_text(json.dumps(users, ensure_ascii=False, indent=1), encoding="utf-8")
        return
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
    if users := load_users():
        print(f"已啟用通行碼：{len(users)} 人（{'、'.join(users)}），每人每天 {DAILY_RUNS} 次")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
