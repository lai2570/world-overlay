"""一個零件一張照片 → 組裝動畫。

流程（三個 Meta 模型各做擅長的事）：
  1. Muse Spark：看所有零件照片（和說明書，可省略），排出組裝步驟；每一步把零件 A 裝到零件 B，
     或裝到前面某一步完成的組合件上。
  2. Muse Spark 找零件位置 + 接口 → SAM 3.1 把每個零件剪成去背圖。
  3. Muse Image：每一步把 A 裝到 B 上，生成組好之後的照片；B 是前一步的組合件時，照著前一步的
     生成圖修改，前後步驟看起來才會一致。
  4. Muse Spark 在生成的組合件照片上找接口（下一步要裝到這個組合件上時用）。
手機端（static/assemble.js）讓 A 的去背圖飛到 B 的接口，再淡入 Muse Image 生成的組裝結果。

用法：
  .venv/Scripts/python parts_guide.py samples/mic_each/*.jpg [--manual samples/mic_manual.jpg]
  零件也可以是拿著轉一圈的短片（.mp4/.mov）：Spark 直接看影片，SAM 3.1 整段追蹤，動畫裡零件會轉一圈
"""
import argparse
import asyncio
import base64
import copy
import io
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests
from meta_sam_parser import (FramePrompt, PromptBox, PromptObject, PromptPoint, build_object_prompt,
                             image_segmentation_format, parse_responses_stream, video_segmentation_format)
from PIL import Image, ImageChops, ImageDraw, ImageStat

import sam_fast
from build_guide import MANUAL_MAX_SIDE
from clip import Clip, load_clip_file
from test_grounding import API_BASE, GROUNDING_SCHEMA, ROOT, SAM_MODEL, SPARK_MODEL, load_api_key, load_image, to_data_url
from build_guide import LEAK_FILL, fill_ratio, leak_retry_prompt
from test_pipeline import _sse_events, sam_with_prompts

IMAGE_MODEL = "muse-image-1.0"
CUTOUT_MAX = 640      # 去背圖最長邊（像素），手機顯示夠用，回傳資料也不會太大
RESULT_MAX = 900      # 生成的組裝結果最長邊
BOX_PAD = 0.04        # Spark 框往外擴一點再給 SAM
CROP_PAD = 0.2        # 只把框附近裁下來給 SAM（框再往外 20%）
SAM_MAX = 800         # 給 SAM 的裁切圖最長邊：SAM 一次呼叫的時間主要是上傳，1280 約 3–4 秒、640 約 1.5 秒
GROUND_EFFORT = "low"  # 找零件框和接口：low 約 6 秒、預設約 9 秒，位置差不多
SPIN_FRAMES = 12      # 零件影片轉一圈的動畫取幾格
SPIN_MAX = 360
SPIN_CANDIDATES = 18  # 從幾格裡挑轉一圈的格子（追蹤漏掉的會被濾掉）
MAX_VIDEOS = 5        # Muse Spark 一次請求最多幾段影片
SPIN_MIN_AREA = 0.5  # 轉一圈的格子：面積至少要有最清楚那一格的一半，不然是追蹤漏掉了
VIDEO_EXT = {".mp4", ".mov", ".webm", ".m4v"}
MAX_CANDIDATES = 5    # 給 Spark 挑的去背候選最多幾個
MIN_AREA = 0.02       # 候選遮罩至少佔框的 2%，太小的是雜點

# 找零件時多要一個給 SAM 文字提示用的英文名稱（SAM 只認英文的類別名）
PART_GROUNDING_SCHEMA = copy.deepcopy(GROUNDING_SCHEMA)
_obj = PART_GROUNDING_SCHEMA["properties"]["objects"]["items"]
_obj["required"].append("sam_phrase")
_obj["properties"]["sam_phrase"] = {"type": "string", "description": "1–3 English words naming what the target is, "
                                    "with its color, e.g. 'white cable', 'black clamp'"}

PICK_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["choice", "reason"],
    "properties": {
        "choice": {"type": "integer", "description": "最好的候選編號；都不對給 0"},
        "reason": {"type": "string"},
    },
}

PLAN_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["goal", "parts", "steps"],
    "properties": {
        "goal": {"type": "string", "description": "組好之後是什麼（一句話）"},
        "parts": {"type": "array", "items": {
            "type": "object",
            "additionalProperties": False,
            "required": ["photo", "name", "look"],
            "properties": {
                "photo": {"type": "integer", "description": "這個零件在第幾張零件照片（從 1 開始）"},
                "name": {"type": "string", "description": "短名稱，步驟裡用同一個名稱"},
                "look": {"type": "string", "description": "外觀（顏色、形狀、材質、明顯特徵），讓人能在照片裡認出它"},
            },
        }},
        "steps": {"type": "array", "items": {
            "type": "object",
            "additionalProperties": False,
            "required": ["part_a", "a_where", "onto_step", "part_b", "b_where", "instruction", "why", "confidence",
                         "a_scale", "manual_step"],
            "properties": {
                "part_a": {"type": "string", "description": "這一步拿起來的零件（parts 裡的 name）"},
                "a_where": {"type": "string", "description": "part_a 上要接的部位"},
                "onto_step": {"type": "integer",
                              "description": "要裝到第幾步完成的組合件上；裝到單一零件上時給 0"},
                "part_b": {"type": "string",
                           "description": "onto_step 為 0 時是要裝上去的零件（parts 裡的 name）；否則是組合件上被接的那個零件"},
                "b_where": {"type": "string", "description": "part_b 上對應的接口"},
                "instruction": {"type": "string", "description": "一句話動作指示"},
                "why": {"type": "string", "description": "從照片（和說明書）看到什麼特徵推斷的"},
                "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
                "a_scale": {"type": "number",
                            "description": "part_a 實際長度約是 B（零件或組合件）的幾倍，例如 0.4；動畫用來縮放"},
                "manual_step": {"type": "string", "description": "對應說明書哪一步；沒有說明書時留空字串"},
            },
        }},
    },
}

PLAN_PROMPT = (
    "Each of the {n} part photos shows one part of a product to be assembled (photos are numbered from 1 in the "
    "order given{manual_note}). Some parts are short videos of the part being turned around in a hand instead of "
    "a photo; look at every side in them, since threads and sockets are often on the back. Work out how to physically assemble them. List every part with the photo it "
    "appears in. Then give the mounting steps in order. Each step attaches part_a either onto a single part "
    "(onto_step = 0, part_b = that part) or onto the assembly finished in an earlier step (onto_step = that step "
    "number, part_b = the part of that assembly that part_a connects to). Every part should be used. "
    "Base your reasoning on visible features such as threads, sockets, clamps and sizes. "
    "Name parts in plain words a person would say (e.g. \"shock mount\"), not identifiers like \"shock_mount\". "
    "Answer in {language}."
)
LANGUAGES = {"zh": "Traditional Chinese", "en": "English"}  # 手機選的語言：步驟、名稱都用這個語言寫


def spark_json(key: str, content: list, schema: dict, name: str, effort: str | None = None) -> dict:
    body = {"model": SPARK_MODEL, "messages": [{"role": "user", "content": content}],
            "response_format": {"type": "json_schema", "json_schema": {"name": name, "strict": True, "schema": schema}}}
    if effort:
        body["reasoning_effort"] = effort
    for attempt in range(3):  # 偶爾斷線或回 5xx（影片的請求比較久，比較常遇到），重試
        try:
            r = requests.post(f"{API_BASE}/chat/completions", json=body, timeout=300,
                              headers={"Authorization": f"Bearer {key}"})
        except requests.ConnectionError:
            if attempt == 2:
                raise RuntimeError("Muse Spark 連線中斷，請再試一次")
            time.sleep(2 + 3 * attempt)
            continue
        if r.status_code < 500 and r.status_code != 429:
            break
        time.sleep(2 + 3 * attempt)
    if not r.ok:
        raise RuntimeError(f"Muse Spark 呼叫失敗 HTTP {r.status_code}：{r.text[:300]}")
    return json.loads(r.json()["choices"][0]["message"]["content"])


def image_part(img) -> dict:
    return {"type": "image_url", "image_url": {"url": to_data_url(img)}}


def plan(key: str, photos: list, manual=None, lang: str = "zh") -> dict:
    """manual：說明書一張圖，或每一頁一張圖的 list（照片或 PDF 頁），可省略。"""
    pages = manual if isinstance(manual, list) else [manual] if manual else []
    note = (f"; first come the {len(pages)} page(s) of the instruction manual in order, then the part photos"
            if pages else "")
    content = [{"type": "text", "text": PLAN_PROMPT.format(n=len(photos), manual_note=note, language=LANGUAGES[lang])}]
    for i, page in enumerate(pages, 1):
        content.append({"type": "text", "text": f"Manual page {i}:"})
        content.append(image_part(page))
    # 一次請求最多 5 段影片。從同一段錄全部零件的影片切出來的：送整段一次 + 每個零件的時間段和最清楚那一格；
    # 個別短片超過 5 段：多的只送最清楚那一格
    sources = list(dict.fromkeys(p.source[0] for p in photos if isinstance(p, Clip) and p.source))
    for fid in sources:
        content.append({"type": "text", "text": "Video showing the parts one at a time:"})
        content.append({"type": "file", "file": {"file_id": fid}})
    videos = len(sources)
    for i, p in enumerate(photos, 1):
        if isinstance(p, Clip) and p.source:
            content.append({"type": "text", "text": f"Part photo {i} (shown at {p.source[1]:.0f}-{p.source[2]:.0f} s "
                                                    "in the video above; its clearest frame):"})
            content.append(image_part(p.photo))
        elif isinstance(p, Clip) and videos < MAX_VIDEOS:
            videos += 1
            content.append({"type": "text", "text": f"Part photo {i} (video):"})
            content.append({"type": "file", "file": {"file_id": p.file_id}})
        else:
            content.append({"type": "text", "text": f"Part photo {i}:"})
            content.append(image_part(p.photo if isinstance(p, Clip) else p))
    return spark_json(key, content, PLAN_SCHEMA, "PartsPlan")


def ground(key: str, img, targets: list[str], effort: str | None = GROUND_EFFORT) -> list[dict]:
    prompt = ("Locate each target in the image. Return one entry per target, in the same order, on the normalized "
              "0-1000 grid where (0,0) is the top-left corner. Box the whole object for a part; for a connection "
              "feature give the point on it. Never include a hand holding the part in its box. "
              "If a target is not visible, set found=false and use empty arrays. "
              "Write scene and reason in Traditional Chinese.\n\nTargets:\n"
              + "\n".join(f"{i + 1}. {t}" for i, t in enumerate(targets)))
    return spark_json(key, [{"type": "text", "text": prompt}, image_part(img)], PART_GROUNDING_SCHEMA, "Grounding",
                      effort)["objects"]


def cutout(img, full: Image.Image) -> tuple[Image.Image, tuple[int, int]]:
    """整張照片大小的遮罩 → 裁到零件範圍的去背圖，和它在原照片的左上角位置。"""
    rgba = img.convert("RGBA")
    rgba.putalpha(full)
    bbox = full.getbbox() or (0, 0, *img.size)
    return rgba.crop(bbox), (bbox[0], bbox[1])


def mask_full(photo, rec) -> Image.Image:
    """SAM 遮罩換成和照片一樣大的 L 圖（零件 255、其他 0）。"""
    m = Image.frombytes("L", (rec.mask.width, rec.mask.height), sam_fast.raster_255(rec.mask))
    b = rec.bounds
    m = m.resize((max(1, int(b.right - b.left)), max(1, int(b.bottom - b.top))))
    full = Image.new("L", photo.size, 0)
    full.paste(m, (int(b.left), int(b.top)))
    return full


def contrast(photo, full: Image.Image, box: PromptBox) -> float:
    """遮罩內的平均顏色和框邊緣（多半是背景）的平均顏色差多少。描成地板時差很小。"""
    rgb = photo.convert("RGB")
    inside = ImageStat.Stat(rgb, full).mean if full.getbbox() else [0, 0, 0]
    edge = Image.new("L", photo.size, 0)
    d = ImageDraw.Draw(edge)
    d.rectangle([box.left, box.top, box.right - 1, box.bottom - 1], outline=255, width=max(2, (box.right - box.left) // 30))
    edge = ImageChops.subtract(edge, full)
    if not edge.getbbox():
        return 0.0
    bg = ImageStat.Stat(rgb, edge).mean
    return sum((a - b) ** 2 for a, b in zip(inside, bg)) ** .5


def sam_prompts(box: PromptBox, point, frame: int = 0) -> list[PromptObject]:
    """兩種提示一起送：只給框、框 + 零件上的點，之後留下和背景差最多的那個。
    只給框時深色零件放在淺色地板上常描成地板；C 型夾這種中間空心的零件，點又常落在空隙裡。"""
    objs = [PromptObject(id=1, prompts=[FramePrompt(frame=frame, box=box)])]
    if point:
        objs.append(PromptObject(id=2, prompts=[FramePrompt(frame=frame, box=box, points=[point])]))
    return objs


def sam_text(key: str, img: Image.Image, phrase: str):
    """SAM 3.1 的文字提示：畫面裡所有符合這個類別的物件。"""
    body = {"model": SAM_MODEL, "stream": True, "input": [{"type": "message", "role": "user", "content": [
        {"type": "input_text", "text": phrase}, {"type": "input_image", "image_url": to_data_url(img)}]}]}
    with requests.post(f"{API_BASE}/responses", json=body, stream=True, timeout=120,
                       headers={"Authorization": f"Bearer {key}"}) as r:
        r.raise_for_status()

        async def run():
            return await parse_responses_stream(_sse_events(r), image_segmentation_format()).final_result()

        return asyncio.run(run())


def on_white_masked(img: Image.Image, mask: Image.Image) -> Image.Image:
    """只留遮罩內，其他塗白（同樣構圖，候選之間好比較）。"""
    out = Image.new("RGB", img.size, (255, 255, 255))
    out.paste(img, mask=mask)
    return out


def pick_mask(key: str, crop: Image.Image, box: PromptBox, masks: list[Image.Image], part: dict) -> int | None:
    """讓 Muse Spark 看原圖和每個去背候選，挑出剛好是這個零件的那個；都不對回傳 None。"""
    marked = crop.copy()
    ImageDraw.Draw(marked).rectangle([box.left, box.top, box.right - 1, box.bottom - 1], outline=(255, 0, 0), width=3)
    content = [{"type": "text", "text": (
        f"The first image shows a part in the red box: {part['name']} ({part['look']}). The following images are "
        "candidate cutouts of it (everything else painted white). Pick the candidate that shows exactly this part: "
        "the whole part, nothing missing, and no table, floor, hand or other object included. Thin parts like cables "
        "count as complete when the whole cable is kept. If no candidate is right, answer 0. "
        "Write the reason in Traditional Chinese.")}, image_part(marked)]
    for i, m in enumerate(masks, 1):
        content += [{"type": "text", "text": f"Candidate {i}:"}, image_part(on_white_masked(crop, m))]
    n = spark_json(key, content, PICK_SCHEMA, "PickCutout", "low")["choice"]
    return n - 1 if 1 <= n <= len(masks) else None


def segment_one(key: str, photo, box: PromptBox, point: list[int], part: dict | None = None,
                phrase: str = "") -> Image.Image | None:
    """SAM 描出一個零件，回傳整張照片大小的遮罩；沒有像樣的結果回傳 None。
    只送框附近、縮到 SAM_MAX 的裁切圖（SAM 一次呼叫的時間大多花在上傳）。
    同時送幾種提示拿候選：框、框＋點、框內縮＋角落負向點、文字（英文類別名；白色線材放在米色桌上
    只給框常描成桌面），再讓 Muse Spark 看候選挑對的那個（part 有給時）。"""
    W, H = photo.size
    bw, bh = box.right - box.left, box.bottom - box.top
    crop_box = (max(0, int(box.left - bw * CROP_PAD)), max(0, int(box.top - bh * CROP_PAD)),
                min(W, int(box.right + bw * CROP_PAD)), min(H, int(box.bottom + bh * CROP_PAD)))
    crop = photo.crop(crop_box)
    k = min(1.0, SAM_MAX / max(crop.size))
    small = crop.resize((max(1, round(crop.width * k)), max(1, round(crop.height * k)))) if k < 1 else crop
    sw, sh = small.size
    sbox = PromptBox(left=max(0, int((box.left - crop_box[0]) * k)), top=max(0, int((box.top - crop_box[1]) * k)),
                     right=min(sw, max(1, int((box.right - crop_box[0]) * k))),
                     bottom=min(sh, max(1, int((box.bottom - crop_box[1]) * k))))
    pt = None
    if len(point) == 2:
        pt = PromptPoint(x=min(sw - 1, max(0, int((point[0] * W / 1000 - crop_box[0]) * k))),
                         y=min(sh - 1, max(0, int((point[1] * H / 1000 - crop_box[1]) * k))))

    calls = [lambda: sam_with_prompts(key, small, sam_prompts(sbox, pt))[0],
             lambda: sam_with_prompts(key, small, [leak_retry_prompt(0, sbox)])[0]]
    if phrase:
        calls.append(lambda: sam_text(key, small, phrase))

    def safe(f):
        try:
            return f()
        except (Exception, SystemExit):  # 某一種提示失敗，其他的照用
            return None

    with ThreadPoolExecutor(len(calls)) as ex:
        results = list(ex.map(safe, calls))
    box_area = max(1, (sbox.right - sbox.left) * (sbox.bottom - sbox.top))
    area = lambda m: ImageStat.Stat(m).mean[0] / 255 * m.width * m.height
    cands = []
    for n, res in enumerate(results):
        for r in (res.records if res else []):
            if r.kind != "mask":
                continue
            m = mask_full(small, r)
            bb = m.getbbox()
            if not bb or area(m) < box_area * MIN_AREA:
                continue
            if n == 2:  # 文字提示會描到畫面裡所有同類的東西：只留大部分在框裡的
                inter = (max(0, min(bb[2], sbox.right) - max(bb[0], sbox.left)) *
                         max(0, min(bb[3], sbox.bottom) - max(bb[1], sbox.top)))
                if inter < 0.6 * (bb[2] - bb[0]) * (bb[3] - bb[1]):
                    continue
            elif part is None and fill_ratio(r, sbox) > LEAK_FILL:  # 沒有 Spark 把關時，先濾掉描成地板的
                continue
            # 和已經有的候選幾乎一樣就不重複
            if any(area(ImageChops.logical_xor(m.convert("1"), c.convert("1")).convert("L")) < 0.1 * area(m)
                   for c in cands):
                continue
            cands.append(m)
    if not cands:
        return None
    cands.sort(key=lambda m: -contrast(small, m, sbox))
    cands = cands[:MAX_CANDIDATES]
    best = 0
    if part is not None:
        try:
            best = pick_mask(key, small, sbox, cands, part)
        except RuntimeError:
            best = None
        if best is None:  # Spark 說都不對或呼叫失敗：退回沒描到地板、和背景差最多的那個（比整塊方框好）
            ok = [n for n, m in enumerate(cands) if area(m) <= box_area * LEAK_FILL]
            if not ok:
                return None
            best = ok[0]
    full = Image.new("L", photo.size, 0)
    full.paste(cands[best].resize(crop.size, Image.BILINEAR), crop_box[:2])
    return full


def clip_masks(key: str, clip: Clip, box: PromptBox, point: list[int]) -> dict[int, Image.Image]:
    """SAM 3.1 影片追蹤：在最清楚的那一格給框（和點），整段影片一次請求追蹤完。
    回傳 {格: SAM 遮罩紀錄}；用最清楚那一格上和背景差最多的那個提示的追蹤結果。"""
    photo = clip.photo
    W, H = photo.size
    pt = None
    if len(point) == 2:
        pt = PromptPoint(x=min(W - 1, int(point[0] * W / 1000)), y=min(H - 1, int(point[1] * H / 1000)))
    objs = sam_prompts(box, pt, clip.key)
    body = {"model": SAM_MODEL, "stream": True, "metadata": {"propagation_direction": "both"},
            "input": [{"type": "message", "role": "user", "content": [
                {"type": "input_text", "text": build_object_prompt(size=photo.size, objects=objs)},
                {"type": "input_video", "file_id": clip.file_id}]}]}
    with requests.post(f"{API_BASE}/responses", json=body, stream=True, timeout=300,
                       headers={"Authorization": f"Bearer {key}"}) as r:
        if not r.ok:
            raise RuntimeError(f"SAM 影片追蹤失敗 HTTP {r.status_code}：{r.text[:300]}")

        async def run():
            return await parse_responses_stream(_sse_events(r), video_segmentation_format()).final_result()

        result = asyncio.run(run())
    tracks: dict[str, dict[int, object]] = {}
    for rec in result.records:
        if rec.kind == "mask" and rec.frame is not None:
            tracks.setdefault(rec.object_id, {})[rec.frame.frame_index] = rec
    ok = {oid: t for oid, t in tracks.items() if clip.key in t and fill_ratio(t[clip.key], box) <= LEAK_FILL}
    if not ok:
        return {}
    best = max(ok.values(), key=lambda t: contrast(photo, mask_full(photo, t[clip.key]), box))
    return {f: rec for f, rec in best.items() if f < len(clip.frames)}


def prepare_part(key: str, src, part: dict, points: list[str]) -> dict:
    """找零件框和接口點 → SAM 去背。src 是照片或零件影片（Clip）。
    回傳 {image: 去背圖, points: {描述: [x, y] 相對去背圖 0–1}, spin: [轉一圈的去背圖], views: [其他角度]}。"""
    clip = src if isinstance(src, Clip) and src.file_id else None
    photo = src.photo if isinstance(src, Clip) else src
    objs = ground(key, photo, [f"{part['name']}（{part['look']}）"] + points)
    W, H = photo.size
    box = objs[0]["box"] if objs[0]["found"] and len(objs[0]["box"]) == 4 else [0, 0, 1000, 1000]
    x1, y1, x2, y2 = box[0] * W / 1000, box[1] * H / 1000, box[2] * W / 1000, box[3] * H / 1000
    px, py = (x2 - x1) * BOX_PAD, (y2 - y1) * BOX_PAD
    prompt = PromptBox(left=max(0, int(x1 - px)), top=max(0, int(y1 - py)),
                       right=min(W, int(x2 + px)), bottom=min(H, int(y2 + py)))
    point = objs[0]["point"] if objs[0]["found"] else []
    spin, views, full = [], [], None
    if clip:
        recs = clip_masks(key, clip, prompt, point)
        full = mask_full(photo, recs[clip.key]) if clip.key in recs else None
        # 只解要用的格子（解一格遮罩約 0.5 秒）：平均挑 SPIN_CANDIDATES 格當候選
        order = sorted(recs)
        cand = {order[round(n * (len(order) - 1) / max(1, SPIN_CANDIDATES - 1))] for n in range(min(SPIN_CANDIDATES, len(order)))}
        masks = {f: mask_full(photo, recs[f]) for f in sorted(cand)}  # 每一格大小都一樣，用 photo 只是取尺寸
        if full is not None:
            # 追蹤偶爾只抓到零件的一小塊（例如只剩螺絲）：面積不到最清楚那一格一半的格子不用
            area = lambda m: ImageStat.Stat(m).mean[0] * m.width * m.height
            least = area(full) * SPIN_MIN_AREA
            masks = {f: m for f, m in masks.items() if area(m) >= least}
        if masks:
            order = sorted(masks)
            picks = [order[round(n * (len(order) - 1) / max(1, SPIN_FRAMES - 1))] for n in range(min(SPIN_FRAMES, len(order)))]
            for f in dict.fromkeys(picks):
                cut, _ = cutout(clip.frames[f], masks[f])
                cut.thumbnail((SPIN_MAX, SPIN_MAX))
                spin.append(cut)
            # 給 Muse Image 的另一個角度：離最清楚那一格最遠（約轉了半圈）的那一格
            far = max(order, key=lambda f: abs(f - clip.key))
            if abs(far - clip.key) >= len(clip.frames) // 4:
                views.append(cutout(clip.frames[far], masks[far])[0])
    if full is None:
        full = segment_one(key, photo, prompt, point, part, objs[0]["sam_phrase"] if objs[0]["found"] else "")
    if full is not None and full.getbbox():
        cut, (left, top) = cutout(photo, full)
    else:  # SAM 沒描到：用框裁切，不去背
        cut = photo.crop((int(x1), int(y1), int(x2), int(y2))).convert("RGBA")
        left, top = int(x1), int(y1)
    rel = {}
    for desc, o in zip(points, objs[1:]):
        if o["found"] and len(o["point"]) == 2:
            rel[desc] = [(o["point"][0] * W / 1000 - left) / cut.width, (o["point"][1] * H / 1000 - top) / cut.height]
    return {"image": cut, "points": rel, "spin": spin, "views": views}


def on_white(img: Image.Image) -> Image.Image:
    bg = Image.new("RGB", img.size, (245, 245, 242))
    bg.paste(img, mask=img.split()[-1] if img.mode == "RGBA" else None)
    return bg


def edit_image(key: str, images: list[Image.Image], prompt: str) -> Image.Image:
    files = []
    for i, im in enumerate(images):
        buf = io.BytesIO()
        on_white(im).save(buf, "PNG")
        files.append((f"image[{i}]", (f"{i}.png", buf.getvalue(), "image/png")))
    for attempt in range(3):  # 偶爾會回 500；後面的步驟都要接著這張圖畫，重試比整串失敗好
        r = requests.post(f"{API_BASE}/images/edits", headers={"Authorization": f"Bearer {key}"},
                          data={"model": IMAGE_MODEL, "prompt": prompt}, files=files, timeout=300)
        if r.status_code < 500 and r.status_code != 429:
            break
        time.sleep(2 + 3 * attempt)
    if not r.ok:
        raise RuntimeError(f"Muse Image 呼叫失敗 HTTP {r.status_code}：{r.text[:300]}")
    item = r.json()["data"][0]
    raw = base64.b64decode(item["b64_json"]) if item.get("b64_json") else requests.get(item["url"], timeout=60).content
    out = Image.open(io.BytesIO(raw)).convert("RGB")
    out.thumbnail((RESULT_MAX, RESULT_MAX))
    return out


def render_step(key: str, s: dict, looks: dict, a_cut, b_img, onto: bool, a_views=()) -> Image.Image:
    """Muse Image：把 A 裝到 B 上。B 是前一步生成的組合件時，照著那張圖改，前後步驟才會一致。
    a_views：A 從其他角度看的去背圖（零件是影片時才有），一起當參考圖，背面的細節才不會亂畫。"""
    a_desc = f"{s['part_a']} ({looks.get(s['part_a'], '')})"
    which = "Image 3 shows" if len(a_views) == 1 else f"Images 3 to {2 + len(a_views)} show"
    extra = (f"\n{which} the same part as image 2 from another angle; "
             "use it for the details hidden in image 2.") if a_views else ""
    if onto:
        prompt = (f"Image 1 is a partly built assembly. Image 2 is a real part: {a_desc}.\n"
                  f"Attach the part from image 2 to the assembly: {s['instruction']} "
                  f"(its {s['a_where']} goes onto the {s['part_b']}'s {s['b_where']}).\n"
                  "Keep the assembly exactly as in image 1: same pose, framing, scale, lighting and plain background. "
                  "Only add the new part, with the same shape, colors and details as in image 2.")
    else:
        prompt = (f"Image 1 is a real part: {s['part_b']} ({looks.get(s['part_b'], '')}). "
                  f"Image 2 is a real part: {a_desc}.\n"
                  f"Show them assembled: {s['instruction']} "
                  f"(the {s['a_where']} of image 2 goes onto the {s['b_where']} of image 1).\n"
                  "Use exactly these parts with the same shapes, colors and details; do not add anything else. "
                  "Product photo on a plain light background, the whole assembly visible, the connection clearly shown.")
    return edit_image(key, [b_img, a_cut, *a_views], prompt + extra)


def data_url(img: Image.Image, fmt: str) -> str:
    buf = io.BytesIO()
    if fmt == "png":
        img.save(buf, "PNG", optimize=True)
    elif fmt == "webp":  # 轉一圈的動畫格：有透明背景又小
        img.save(buf, "WEBP", quality=80)
    else:
        img.save(buf, "JPEG", quality=88)
    return f"data:image/{fmt};base64," + base64.b64encode(buf.getvalue()).decode()


def analyze(key: str, photos: list, manual=None, log=print, publish=None, lang: str = "zh") -> dict:
    """publish(result) 在有新進度時呼叫：去背完就先給步驟和零件圖，之後每生成一步就補上那一步的結果圖。
    lang："zh" 或 "en"，AI 寫的步驟和進度訊息用這個語言。"""
    def t(zh: str, en: str) -> str:
        return en if lang == "en" else zh

    t0 = time.perf_counter()
    log(t("對照說明書推理組裝步驟", "Working out the steps from the manual") if manual else t("推理組裝步驟", "Working out the assembly steps"))
    p = plan(key, photos, manual, lang)
    steps, looks = p["steps"], {q["name"]: q["look"] for q in p["parts"]}
    dt = time.perf_counter() - t0
    log(t(f"推理完成：{len(p['parts'])} 個零件、{len(steps)} 個步驟（{dt:.0f} 秒）",
          f"Planned: {len(p['parts'])} parts, {len(steps)} steps ({dt:.0f} s)"))

    # 每個零件要找哪些接口：A 端都在 A 的照片上；B 端只有裝到單一零件時在 B 的照片上
    wanted: dict[str, list[str]] = {q["name"]: [] for q in p["parts"]}
    for k, s in enumerate(steps, 1):
        wanted.setdefault(s["part_a"], []).append(s["a_where"])
        if not 0 < s["onto_step"] < k:
            wanted.setdefault(s["part_b"], []).append(s["b_where"])
    by_name = {q["name"]: q for q in p["parts"]}

    t1 = time.perf_counter()
    log(t("剪下每個零件、找接口", "Cutting out each part and finding the joints"))
    with ThreadPoolExecutor(8) as ex:
        futs = {n: ex.submit(prepare_part, key, photos[by_name[n]["photo"] - 1], by_name[n], pts)
                for n, pts in wanted.items() if n in by_name and 1 <= by_name[n]["photo"] <= len(photos)}
        parts = {n: f.result() for n, f in futs.items()}
    dt = time.perf_counter() - t1
    log(t(f"去背完成：{len(parts)} 個（{dt:.0f} 秒）", f"Cut out {len(parts)} parts ({dt:.0f} s)"))

    out_steps = []
    for k, s in enumerate(steps, 1):
        onto = 0 < s["onto_step"] < k
        why = t(f"〔說明書{s['manual_step']}〕", f"[Manual {s['manual_step']}] ") + s["why"] if s.get("manual_step") else s["why"]
        out_steps.append({
            "part_a": s["part_a"], "part_b": s["part_b"], "instruction": s["instruction"], "why": why,
            "confidence": s["confidence"], "a_scale": s["a_scale"], "onto_step": s["onto_step"] if onto else 0,
            "a_img": f"part:{s['part_a']}" if s["part_a"] in parts else None,
            "a_pt": parts.get(s["part_a"], {}).get("points", {}).get(s["a_where"]),
            "b_img": f"result{s['onto_step']}" if onto else (f"part:{s['part_b']}" if s["part_b"] in parts else None),
            "b_pt": None if onto else parts.get(s["part_b"], {}).get("points", {}).get(s["b_where"]),
            "result": None,
        })
    result = {"goal": p["goal"], "steps": out_steps, "pending": len(steps),
              "images": {f"part:{n}": data_url(_shrink(v["image"]), "png") for n, v in parts.items()},
              "spins": {n: [data_url(f, "webp") for f in v["spin"]] for n, v in parts.items() if v["spin"]},
              "looks": {n: looks.get(n, "") for n in parts}}
    if publish:
        publish(result)

    # 依相依順序生成：裝到組合件上的步驟要等那一步生成完；彼此獨立的分支同時跑
    t2 = time.perf_counter()
    log(t(f"Muse Image 依序生成 {len(steps)} 張組裝結果", f"Muse Image is drawing {len(steps)} assembled results"))
    lock = threading.Lock()
    futures: dict[int, object] = {}
    finished = [0]

    def run(k: int):
        s, o = steps[k - 1], out_steps[k - 1]
        try:
            if s["part_a"] not in parts:
                raise RuntimeError(t(f"沒有 {s['part_a']} 的照片", f"no photo of {s['part_a']}"))
            a_cut, a_views = parts[s["part_a"]]["image"], parts[s["part_a"]]["views"]
            if o["onto_step"]:
                base = futures[o["onto_step"]].result()
                if base is None:
                    raise RuntimeError(t(f"第 {o['onto_step']} 步沒有結果圖", f"step {o['onto_step']} has no result image"))
                # 同時做：在組合件照片上找 B 的接口、生成這一步
                with ThreadPoolExecutor(1) as ex:
                    pt = ex.submit(ground, key, base, [t(f"{s['part_b']}上的{s['b_where']}", f"the {s['b_where']} on the {s['part_b']}")])
                    img = render_step(key, s, looks, a_cut, base, True, a_views)
                    obj = pt.result()[0]
                if obj["found"] and len(obj["point"]) == 2:
                    o["b_pt"] = [obj["point"][0] / 1000, obj["point"][1] / 1000]
            else:
                if s["part_b"] not in parts:
                    raise RuntimeError(t(f"沒有 {s['part_b']} 的照片", f"no photo of {s['part_b']}"))
                img = render_step(key, s, looks, a_cut, parts[s["part_b"]]["image"], False, a_views)
        except Exception as e:  # 這一步沒有結果圖：動畫只播零件飛過去，不淡入結果
            log(t(f"步驟 {k} 生成失敗：{e}", f"Step {k} could not be drawn: {e}"))
            img = None
        with lock:
            finished[0] += 1
            if img is not None:
                result["images"][f"result{k}"] = data_url(img, "jpeg")
                o["result"] = f"result{k}"
            result["pending"] -= 1
            dt = time.perf_counter() - t2
            log(t(f"步驟 {k} 生成完成（{finished[0]}/{len(steps)}，{dt:.0f} 秒）",
                  f"Step {k} drawn ({finished[0]}/{len(steps)}, {dt:.0f} s)"))
            if publish:
                publish(result)
        return img

    with ThreadPoolExecutor(max(1, len(steps))) as ex:
        for k in range(1, len(steps) + 1):
            futures[k] = ex.submit(run, k)
        for f in list(futures.values()):
            f.result()
    dt = time.perf_counter() - t0
    log(t(f"全部完成（{dt:.0f} 秒）", f"All done ({dt:.0f} s)"))
    return result


def _shrink(img: Image.Image) -> Image.Image:
    img = img.copy()
    img.thumbnail((CUTOUT_MAX, CUTOUT_MAX))
    return img


def main():
    import sys
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("photos", nargs="+", help="零件照片或轉一圈的影片（.mp4/.mov/.webm），一個零件一個")
    ap.add_argument("--manual", nargs="+", help="說明書照片，可以多頁（可省略）")
    ap.add_argument("--out", default=str(ROOT / "out" / "parts_guide.json"))
    args = ap.parse_args()
    key = load_api_key()
    photos = [load_clip_file(p, key) if Path(p).suffix.lower() in VIDEO_EXT else load_image(p) for p in args.photos]
    manual = [load_image(m, MANUAL_MAX_SIDE) for m in args.manual or []]
    result = analyze(key, photos, manual, log=lambda m: print(f"  {m}"))
    Path(args.out).write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")
    for n, s in enumerate(result["steps"], 1):
        print(f"  步驟 {n}：{s['part_a']} → {s['part_b']}{'（組合件 #' + str(s['onto_step']) + '）' if s['onto_step'] else ''}"
              f"｜{s['instruction']}｜A點 {s['a_pt']} B點 {s['b_pt']}")
    print(f"結果：{args.out}")


if __name__ == "__main__":
    main()
