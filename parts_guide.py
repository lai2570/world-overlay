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
"""
import argparse
import base64
import io
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests
from meta_sam_parser import FramePrompt, PromptBox, PromptObject, PromptPoint, decode_mask_to_raster
from PIL import Image, ImageChops, ImageDraw, ImageStat

from build_guide import MANUAL_MAX_SIDE
from test_grounding import API_BASE, GROUNDING_SCHEMA, ROOT, SPARK_MODEL, load_api_key, load_image, to_data_url
from build_guide import LEAK_FILL, fill_ratio, leak_retry_prompt
from test_pipeline import sam_with_prompts

IMAGE_MODEL = "muse-image-1.0"
CUTOUT_MAX = 640      # 去背圖最長邊（像素），手機顯示夠用，回傳資料也不會太大
RESULT_MAX = 900      # 生成的組裝結果最長邊
BOX_PAD = 0.04        # Spark 框往外擴一點再給 SAM

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
    "order given{manual_note}). Work out how to physically assemble them. List every part with the photo it "
    "appears in. Then give the mounting steps in order. Each step attaches part_a either onto a single part "
    "(onto_step = 0, part_b = that part) or onto the assembly finished in an earlier step (onto_step = that step "
    "number, part_b = the part of that assembly that part_a connects to). Every part should be used. "
    "Base your reasoning on visible features such as threads, sockets, clamps and sizes. "
    "Answer in Traditional Chinese."
)


def spark_json(key: str, content: list, schema: dict, name: str, effort: str | None = None) -> dict:
    body = {"model": SPARK_MODEL, "messages": [{"role": "user", "content": content}],
            "response_format": {"type": "json_schema", "json_schema": {"name": name, "strict": True, "schema": schema}}}
    if effort:
        body["reasoning_effort"] = effort
    r = requests.post(f"{API_BASE}/chat/completions", json=body, timeout=300,
                      headers={"Authorization": f"Bearer {key}"})
    if not r.ok:
        raise RuntimeError(f"Muse Spark 呼叫失敗 HTTP {r.status_code}：{r.text[:300]}")
    return json.loads(r.json()["choices"][0]["message"]["content"])


def image_part(img) -> dict:
    return {"type": "image_url", "image_url": {"url": to_data_url(img)}}


def plan(key: str, photos: list, manual=None) -> dict:
    note = "; the first image is the instruction manual, the part photos follow it" if manual else ""
    content = [{"type": "text", "text": PLAN_PROMPT.format(n=len(photos), manual_note=note)}]
    if manual:
        content.append(image_part(manual))
    content += [image_part(p) for p in photos]
    return spark_json(key, content, PLAN_SCHEMA, "PartsPlan")


def ground(key: str, img, targets: list[str]) -> list[dict]:
    prompt = ("Locate each target in the image. Return one entry per target, in the same order, on the normalized "
              "0-1000 grid where (0,0) is the top-left corner. Box the whole object for a part; for a connection "
              "feature give the point on it. If a target is not visible, set found=false and use empty arrays. "
              "Write scene and reason in Traditional Chinese.\n\nTargets:\n"
              + "\n".join(f"{i + 1}. {t}" for i, t in enumerate(targets)))
    return spark_json(key, [{"type": "text", "text": prompt}, image_part(img)], GROUNDING_SCHEMA, "Grounding")["objects"]


def cutout(img, rec) -> tuple[Image.Image, tuple[int, int]]:
    """SAM 遮罩 → 裁到零件範圍的去背圖，和它在原照片的左上角位置。"""
    full = mask_full(img, rec)
    rgba = img.convert("RGBA")
    rgba.putalpha(full)
    bbox = full.getbbox() or (0, 0, *img.size)
    return rgba.crop(bbox), (bbox[0], bbox[1])


def mask_full(photo, rec) -> Image.Image:
    """SAM 遮罩換成和照片一樣大的 L 圖（零件 255、其他 0）。"""
    raster = decode_mask_to_raster(rec.mask)
    m = Image.frombytes("L", (rec.mask.width, rec.mask.height), bytes(v * 255 for v in raster))
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


def segment_one(key: str, photo, box: PromptBox, point: list[int]):
    """一次 SAM 呼叫試兩種提示：只給框、框 + 零件上的點，留下和背景差最多的那個。
    只給框時深色零件放在淺色地板上常描成地板；C 型夾這種中間空心的零件，點又常落在空隙裡。"""
    W, H = photo.size
    objs = [PromptObject(id=1, prompts=[FramePrompt(frame=0, box=box)])]
    if len(point) == 2:
        pt = PromptPoint(x=int(point[0] * W / 1000), y=int(point[1] * H / 1000))
        objs.append(PromptObject(id=2, prompts=[FramePrompt(frame=0, box=box, points=[pt])]))
    result, _ = sam_with_prompts(key, photo, objs)
    recs = [r for r in result.records if r.kind == "mask" and fill_ratio(r, box) <= LEAK_FILL]
    if not recs:  # 都連背景一起描進去：框內縮 + 角落負向點重描
        retry, _ = sam_with_prompts(key, photo, [leak_retry_prompt(0, box)])
        recs = [r for r in retry.records if r.kind == "mask"]
    if not recs:
        return None
    return max(recs, key=lambda r: contrast(photo, mask_full(photo, r), box))


def prepare_part(key: str, photo, part: dict, points: list[str]) -> dict:
    """找零件框和接口點 → SAM 去背。回傳 {image: 去背圖, points: {描述: [x, y] 相對去背圖 0–1}}。"""
    objs = ground(key, photo, [f"{part['name']}（{part['look']}）"] + points)
    W, H = photo.size
    box = objs[0]["box"] if objs[0]["found"] and len(objs[0]["box"]) == 4 else [0, 0, 1000, 1000]
    x1, y1, x2, y2 = box[0] * W / 1000, box[1] * H / 1000, box[2] * W / 1000, box[3] * H / 1000
    px, py = (x2 - x1) * BOX_PAD, (y2 - y1) * BOX_PAD
    prompt = PromptBox(left=max(0, int(x1 - px)), top=max(0, int(y1 - py)),
                       right=min(W, int(x2 + px)), bottom=min(H, int(y2 + py)))
    rec = segment_one(key, photo, prompt, objs[0]["point"] if objs[0]["found"] else [])
    if rec is not None:
        cut, (left, top) = cutout(photo, rec)
    else:  # SAM 沒描到：用框裁切，不去背
        cut = photo.crop((int(x1), int(y1), int(x2), int(y2))).convert("RGBA")
        left, top = int(x1), int(y1)
    rel = {}
    for desc, o in zip(points, objs[1:]):
        if o["found"] and len(o["point"]) == 2:
            rel[desc] = [(o["point"][0] * W / 1000 - left) / cut.width, (o["point"][1] * H / 1000 - top) / cut.height]
    return {"image": cut, "points": rel}


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
    r = requests.post(f"{API_BASE}/images/edits", headers={"Authorization": f"Bearer {key}"},
                      data={"model": IMAGE_MODEL, "prompt": prompt}, files=files, timeout=300)
    if not r.ok:
        raise RuntimeError(f"Muse Image 呼叫失敗 HTTP {r.status_code}：{r.text[:300]}")
    item = r.json()["data"][0]
    raw = base64.b64decode(item["b64_json"]) if item.get("b64_json") else requests.get(item["url"], timeout=60).content
    out = Image.open(io.BytesIO(raw)).convert("RGB")
    out.thumbnail((RESULT_MAX, RESULT_MAX))
    return out


def render_step(key: str, s: dict, looks: dict, a_cut, b_img, onto: bool) -> Image.Image:
    """Muse Image：把 A 裝到 B 上。B 是前一步生成的組合件時，照著那張圖改，前後步驟才會一致。"""
    a_desc = f"{s['part_a']} ({looks.get(s['part_a'], '')})"
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
    return edit_image(key, [b_img, a_cut], prompt)


def data_url(img: Image.Image, fmt: str) -> str:
    buf = io.BytesIO()
    if fmt == "png":
        img.save(buf, "PNG", optimize=True)
    else:
        img.save(buf, "JPEG", quality=88)
    return f"data:image/{fmt};base64," + base64.b64encode(buf.getvalue()).decode()


def analyze(key: str, photos: list, manual=None, log=print, publish=None) -> dict:
    """publish(result) 在有新進度時呼叫：去背完就先給步驟和零件圖，之後每生成一步就補上那一步的結果圖。"""
    t0 = time.perf_counter()
    log("對照說明書推理組裝步驟" if manual else "推理組裝步驟")
    p = plan(key, photos, manual)
    steps, looks = p["steps"], {q["name"]: q["look"] for q in p["parts"]}
    log(f"推理完成：{len(p['parts'])} 個零件、{len(steps)} 個步驟（{time.perf_counter() - t0:.0f} 秒）")

    # 每個零件要找哪些接口：A 端都在 A 的照片上；B 端只有裝到單一零件時在 B 的照片上
    wanted: dict[str, list[str]] = {q["name"]: [] for q in p["parts"]}
    for k, s in enumerate(steps, 1):
        wanted.setdefault(s["part_a"], []).append(s["a_where"])
        if not 0 < s["onto_step"] < k:
            wanted.setdefault(s["part_b"], []).append(s["b_where"])
    by_name = {q["name"]: q for q in p["parts"]}

    t1 = time.perf_counter()
    log("剪下每個零件、找接口")
    with ThreadPoolExecutor(8) as ex:
        futs = {n: ex.submit(prepare_part, key, photos[by_name[n]["photo"] - 1], by_name[n], pts)
                for n, pts in wanted.items() if n in by_name and 1 <= by_name[n]["photo"] <= len(photos)}
        parts = {n: f.result() for n, f in futs.items()}
    log(f"去背完成：{len(parts)} 個（{time.perf_counter() - t1:.0f} 秒）")

    out_steps = []
    for k, s in enumerate(steps, 1):
        onto = 0 < s["onto_step"] < k
        why = f"〔說明書{s['manual_step']}〕{s['why']}" if s.get("manual_step") else s["why"]
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
              "images": {f"part:{n}": data_url(_shrink(v["image"]), "png") for n, v in parts.items()}}
    if publish:
        publish(result)

    # 依相依順序生成：裝到組合件上的步驟要等那一步生成完；彼此獨立的分支同時跑
    t2 = time.perf_counter()
    log(f"Muse Image 依序生成 {len(steps)} 張組裝結果")
    lock = threading.Lock()
    futures: dict[int, object] = {}
    finished = [0]

    def run(k: int):
        s, o = steps[k - 1], out_steps[k - 1]
        try:
            if s["part_a"] not in parts:
                raise RuntimeError(f"沒有 {s['part_a']} 的照片")
            a_cut = parts[s["part_a"]]["image"]
            if o["onto_step"]:
                base = futures[o["onto_step"]].result()
                if base is None:
                    raise RuntimeError(f"第 {o['onto_step']} 步沒有結果圖")
                # 同時做：在組合件照片上找 B 的接口、生成這一步
                with ThreadPoolExecutor(1) as ex:
                    pt = ex.submit(ground, key, base, [f"{s['part_b']}上的{s['b_where']}"])
                    img = render_step(key, s, looks, a_cut, base, True)
                    obj = pt.result()[0]
                if obj["found"] and len(obj["point"]) == 2:
                    o["b_pt"] = [obj["point"][0] / 1000, obj["point"][1] / 1000]
            else:
                if s["part_b"] not in parts:
                    raise RuntimeError(f"沒有 {s['part_b']} 的照片")
                img = render_step(key, s, looks, a_cut, parts[s["part_b"]]["image"], False)
        except Exception as e:  # 這一步沒有結果圖：動畫只播零件飛過去，不淡入結果
            log(f"步驟 {k} 生成失敗：{e}")
            img = None
        with lock:
            finished[0] += 1
            if img is not None:
                result["images"][f"result{k}"] = data_url(img, "jpeg")
                o["result"] = f"result{k}"
            result["pending"] -= 1
            log(f"步驟 {k} 生成完成（{finished[0]}/{len(steps)}，{time.perf_counter() - t2:.0f} 秒）")
            if publish:
                publish(result)
        return img

    with ThreadPoolExecutor(max(1, len(steps))) as ex:
        for k in range(1, len(steps) + 1):
            futures[k] = ex.submit(run, k)
        for f in list(futures.values()):
            f.result()
    log(f"全部完成（{time.perf_counter() - t0:.0f} 秒）")
    return result


def _shrink(img: Image.Image) -> Image.Image:
    img = img.copy()
    img.thumbnail((CUTOUT_MAX, CUTOUT_MAX))
    return img


def main():
    import sys
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("photos", nargs="+", help="零件照片，一個零件一張")
    ap.add_argument("--manual", help="說明書照片（可省略）")
    ap.add_argument("--out", default=str(ROOT / "out" / "parts_guide.json"))
    args = ap.parse_args()
    photos = [load_image(p) for p in args.photos]
    manual = load_image(args.manual, MANUAL_MAX_SIDE) if args.manual else None
    result = analyze(load_api_key(), photos, manual, log=lambda m: print(f"  {m}"))
    Path(args.out).write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")
    for n, s in enumerate(result["steps"], 1):
        print(f"  步驟 {n}：{s['part_a']} → {s['part_b']}{'（組合件 #' + str(s['onto_step']) + '）' if s['onto_step'] else ''}"
              f"｜{s['instruction']}｜A點 {s['a_pt']} B點 {s['b_pt']}")
    print(f"結果：{args.out}")


if __name__ == "__main__":
    main()
