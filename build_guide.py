"""一張零件照片 → AI 推理組裝步驟 → SAM 描出每步兩個零件的輪廓 → 產生互動組裝引導網頁。

用法：
  .venv/Scripts/python build_guide.py samples/mic_parts.jpg
  .venv/Scripts/python build_guide.py samples/mic_parts.jpg --reuse   # 不呼叫 API，只用上次結果重畫網頁

輸出：out/<圖名>_guide.html（單一檔案，圖片內嵌，可直接傳到手機打開）
      out/<圖名>_guide.json（AI 結果快取）
"""
import argparse
import html
import json
import re
import time
from pathlib import Path

import requests
from meta_sam_parser import FramePrompt, PromptBox, PromptObject, PromptPoint, decode_mask_to_raster, decode_mask_to_svg_path

from test_grounding import API_BASE, ROOT, SPARK_MODEL, load_api_key, load_image, spark_ground, to_data_url
from test_pipeline import sam_with_boxes, sam_with_prompts, spark_boxes_to_pixels

# 推理和定位分兩次呼叫：同一次又推理又標座標時，座標明顯變不準（框會偏掉，SAM 跟著描到地板）。
SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["goal", "parts", "steps"],
    "properties": {
        "goal": {"type": "string", "description": "組好之後是什麼樣子（一句話）"},
        "parts": {"type": "array", "items": {
            "type": "object",
            "additionalProperties": False,
            "required": ["name", "look"],
            "properties": {
                "name": {"type": "string", "description": "短名稱，步驟裡用同一個名稱"},
                "look": {"type": "string", "description": "外觀與在照片中的位置，讓別人能在照片裡找到它，例如「左上方黑色 C 型金屬夾，帶 T 型螺桿」"},
            },
        }},
        "steps": {"type": "array", "items": {
            "type": "object",
            "additionalProperties": False,
            "required": ["part_a", "a_where", "part_b", "b_where", "instruction", "why", "confidence"],
            "properties": {
                "part_a": {"type": "string", "description": "這一步拿起來的零件（parts 裡的 name）"},
                "a_where": {"type": "string", "description": "part_a 上要接的部位，例如「底部的圓柱插銷」"},
                "part_b": {"type": "string", "description": "要接上去的零件（parts 裡的 name）"},
                "b_where": {"type": "string", "description": "part_b 上對應的接口，例如「側面的圓形安裝孔」"},
                "instruction": {"type": "string", "description": "一句話動作指示，例如「把螺柱對準圓孔插入，順時針轉緊旋鈕」"},
                "why": {"type": "string", "description": "從照片看到什麼特徵推斷的"},
                "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
            },
        }},
    },
}
PROMPT = (
    "The photo shows the unassembled parts of a product. Without any assembly instructions, work out how to "
    "physically assemble them. List the parts used in mounting (skip cables and electronics) with a short name "
    "and a description precise enough to find each one in the photo. Then give the mounting steps in order: the "
    "part picked up (part_a), the part it attaches to (part_b), and the exact feature on each part where they "
    "connect. Base your reasoning on visible features such as threads, sockets, clamps and sizes. "
    "Answer in Traditional Chinese."
)


def plan_assembly(key: str, img) -> tuple[dict, float]:
    body = {
        "model": SPARK_MODEL,
        "messages": [{"role": "user", "content": [
            {"type": "text", "text": PROMPT},
            {"type": "image_url", "image_url": {"url": to_data_url(img)}},
        ]}],
        "response_format": {"type": "json_schema", "json_schema": {"name": "Guide", "strict": True, "schema": SCHEMA}},
    }
    t0 = time.perf_counter()
    r = requests.post(f"{API_BASE}/chat/completions", json=body, timeout=300,
                      headers={"Authorization": f"Bearer {key}"})
    if not r.ok:
        raise SystemExit(f"Muse Spark 呼叫失敗 HTTP {r.status_code}：{r.text[:800]}")
    return json.loads(r.json()["choices"][0]["message"]["content"]), time.perf_counter() - t0


def locate(key: str, img, plan: dict) -> tuple[dict, float]:
    """第二次呼叫：用找零件的流程標出每個零件的框，和每一步兩端接口的點。"""
    parts = plan["parts"]
    targets = [f"{p['name']}（{p['look']}）" for p in parts]
    for s in plan["steps"]:
        targets += [f"{s['part_a']}上的{s['a_where']}", f"{s['part_b']}上的{s['b_where']}"]
    result, dt = spark_ground(key, img, targets)
    objs = result["objects"]
    boxes = [{"target": p["name"], "found": o["found"], "box": o["box"]} for p, o in zip(parts, objs)]
    rest = objs[len(parts):]
    points = [[rest[2 * k]["point"], rest[2 * k + 1]["point"]] for k in range(len(plan["steps"]))]
    return {"parts": boxes, "points": points}, dt


SUBPATH_RE = re.compile(r"M[^M]*")
NUM_RE = re.compile(r"-?\d+(?:\.\d+)?")


def clean_path(path: str) -> str:
    """去掉 SAM 多描的小碎塊（地上的頭髮、線頭）。

    零件的洞也是子路徑：在主輪廓範圍內、且不是太小的才保留。斜放的長零件外框很大，
    碎塊常落在框內，所以「在框內」之外還要看大小。
    """
    subs = []
    for sp in SUBPATH_RE.findall(path):
        nums = [float(n) for n in NUM_RE.findall(sp)]
        xs, ys = nums[0::2], nums[1::2]
        subs.append((sp, (min(xs), min(ys), max(xs), max(ys))))
    if len(subs) <= 1:
        return path
    area = lambda b: (b[2] - b[0]) * (b[3] - b[1])
    main = max(subs, key=lambda s: area(s[1]))[1]
    inside = lambda b: b[0] >= main[0] and b[1] >= main[1] and b[2] <= main[2] and b[3] <= main[3]
    return "".join(sp for sp, b in subs
                   if b == main or (inside(b) and area(b) > 0.01 * area(main)) or area(b) > 0.15 * area(main))


PAD_RATIO = 0.1  # 重描時框往內縮的比例
LEAK_FILL = 0.8  # 輪廓填滿提示框 80% 以上，多半是連背景（地板）一起描進去了


def fill_ratio(rec, box: PromptBox) -> float:
    ones = sum(decode_mask_to_raster(rec.mask))
    scale = ((rec.bounds.right - rec.bounds.left) / rec.mask.width) * ((rec.bounds.bottom - rec.bounds.top) / rec.mask.height)
    return ones * scale / max(1, (box.right - box.left) * (box.bottom - box.top))


def leak_retry_prompt(i: int, box: PromptBox) -> PromptObject:
    """不外擴的框 + 中心正向點 + 四個角落負向點（角落通常是背景）。"""
    w, h = box.right - box.left, box.bottom - box.top
    inner = PromptBox(left=box.left + int(w * PAD_RATIO), top=box.top + int(h * PAD_RATIO),
                      right=box.right - int(w * PAD_RATIO), bottom=box.bottom - int(h * PAD_RATIO))
    cx, cy = (box.left + box.right) // 2, (box.top + box.bottom) // 2
    corners = [PromptPoint(x=x, y=y, label="negative")
               for x in (inner.left + 2, inner.right - 3) for y in (inner.top + 2, inner.bottom - 3)]
    return PromptObject(id=i + 1, prompts=[FramePrompt(frame=0, box=inner,
                                                       points=[PromptPoint(x=cx, y=cy), *corners])])



def segment_parts(key: str, img, parts: list[dict]) -> tuple[dict, float]:
    boxes = spark_boxes_to_pixels(parts, img.size)
    result, dt = sam_with_boxes(key, img, boxes)
    masks = {rec.object_id: rec for rec in result.records if rec.kind == "mask"}

    leaky = [(i, part, box) for i, part, box in boxes
             if str(i + 1) in masks and fill_ratio(masks[str(i + 1)], box) > LEAK_FILL]
    if leaky:
        print(f"    {'、'.join(p['target'] for _, p, _ in leaky)} 疑似描到背景，用中心點重描…")
        retry, dt2 = sam_with_prompts(key, img, [leak_retry_prompt(i, box) for i, _, box in leaky])
        dt += dt2
        for rec in retry.records:
            if rec.kind == "mask":
                masks[rec.object_id] = rec

    shapes = {}
    for i, part, _ in boxes:
        rec = masks.get(str(i + 1))
        if rec is None:
            continue
        sx = (rec.bounds.right - rec.bounds.left) / rec.mask.width
        sy = (rec.bounds.bottom - rec.bounds.top) / rec.mask.height
        shapes[part["target"]] = {
            "transform": f"translate({rec.bounds.left} {rec.bounds.top}) scale({sx} {sy})",
            "d": decode_mask_to_svg_path(rec.mask),
        }
    return shapes, dt


def render_html(img, plan: dict, located: dict, shapes: dict, out_path: Path):
    W, H = img.size
    pt = lambda p: [p[0] * W / 1000, p[1] * H / 1000] if len(p) == 2 else None
    # 接口在照片裡看不到時（例如海綿套內部），箭頭改從零件框的中心出發
    centers = {p["target"]: [(p["box"][0] + p["box"][2]) / 2, (p["box"][1] + p["box"][3]) / 2]
               for p in located["parts"] if len(p["box"]) == 4}
    steps = []
    for s, (a, b) in zip(plan["steps"], located["points"]):
        a = a if len(a) == 2 else centers.get(s["part_a"], [])
        b = b if len(b) == 2 else centers.get(s["part_b"], [])
        steps.append({k: s[k] for k in ("part_a", "part_b", "instruction", "why", "confidence")}
                     | {"a": pt(a), "b": pt(b)})
    shapes = {name: sh | {"d": clean_path(sh["d"])} for name, sh in shapes.items()}
    data = json.dumps({"w": W, "h": H, "goal": plan["goal"], "steps": steps, "shapes": shapes}, ensure_ascii=False)
    page = TEMPLATE.replace("__IMG__", to_data_url(img)).replace("__DATA__", data.replace("</", "<\\/"))
    page = page.replace("__GOAL__", html.escape(plan["goal"]))
    out_path.write_text(page, encoding="utf-8")


TEMPLATE = """<!doctype html>
<html lang="zh-Hant"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover">
<title>WORLD OVERLAY 組裝引導</title>
<style>
:root{--a:#ffb020;--b:#22d3ee;--bg:#0b0d10;--card:#16191e;--fg:#f2f4f7;--mute:#9aa3ae}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);font:16px/1.5 system-ui,-apple-system,"Noto Sans TC","Microsoft JhengHei",sans-serif}
header{padding:12px 16px 4px;font-size:13px;color:var(--mute)}
header b{color:var(--fg);letter-spacing:.08em}
.stage{position:relative;width:100%;max-width:960px;margin:0 auto}
.stage img,.stage svg{display:block;width:100%;height:auto}
.stage svg{position:absolute;inset:0;height:100%}
.part{fill:none;stroke-width:4;stroke-linejoin:round;animation:pulse 1.4s ease-in-out infinite}
.part.a{stroke:var(--a);filter:drop-shadow(0 0 6px var(--a)) drop-shadow(0 0 16px var(--a))}
.part.b{stroke:var(--b);filter:drop-shadow(0 0 6px var(--b)) drop-shadow(0 0 16px var(--b));animation-delay:.7s}
@keyframes pulse{50%{stroke-width:8;opacity:.6}}
.flow{fill:none;stroke:#fff;stroke-width:5;stroke-linecap:round;stroke-dasharray:14 12;animation:dash 1s linear infinite;
  filter:drop-shadow(0 0 4px #000)}
@keyframes dash{to{stroke-dashoffset:-26}}
.ring{fill:none;stroke:var(--b);stroke-width:4;transform-box:fill-box;transform-origin:center;animation:ring 1.4s ease-out infinite}
@keyframes ring{from{transform:scale(.4);opacity:1}to{transform:scale(1.8);opacity:0}}
.card{max-width:960px;margin:0 auto;padding:14px 16px 20px}
.count{font-size:13px;color:var(--mute)}
.pair{display:flex;flex-wrap:wrap;align-items:center;gap:8px;margin:6px 0 8px;font-weight:700;font-size:18px}
.chip{padding:2px 10px;border-radius:999px;border:2px solid}
.chip.a{border-color:var(--a);color:var(--a)} .chip.b{border-color:var(--b);color:var(--b)}
.inst{font-size:17px;margin:0 0 8px}
details{color:var(--mute);font-size:14px} summary{cursor:pointer}
.conf{display:inline-block;font-size:12px;padding:1px 8px;border-radius:6px;background:#2a2f37;color:var(--mute);margin-left:6px}
.nav{display:flex;gap:10px;margin-top:14px}
button{flex:1;padding:14px;border:0;border-radius:12px;font:inherit;font-weight:700;background:#2a2f37;color:var(--fg)}
button.primary{background:var(--a);color:#111}
button:disabled{opacity:.35}
.done{display:none;text-align:center;padding:24px 0;font-size:18px}
</style></head><body>
<header><b>WORLD OVERLAY</b>　目標：__GOAL__</header>
<div class="stage">
  <img id="photo" src="__IMG__" alt="零件照片">
  <svg id="ov" preserveAspectRatio="none"></svg>
</div>
<div class="card">
  <div class="count" id="count"></div>
  <div class="pair"><span class="chip a" id="pa"></span><span>→</span><span class="chip b" id="pb"></span></div>
  <p class="inst" id="inst"></p>
  <details><summary>AI 為什麼這樣判斷<span class="conf" id="conf"></span></summary><p id="why"></p></details>
  <div class="done" id="done">全部步驟完成 🎉</div>
  <div class="nav"><button id="prev">上一步</button><button class="primary" id="next">下一步</button></div>
</div>
<script>
const D = __DATA__;
const NS = "http://www.w3.org/2000/svg";
const ov = document.getElementById("ov");
ov.setAttribute("viewBox", `0 0 ${D.w} ${D.h}`);
const $ = id => document.getElementById(id);
const CONF = {high: "把握高", medium: "把握中", low: "把握低"};
let i = Math.min(Math.max((parseInt(location.hash.slice(1)) || 1) - 1, 0), D.steps.length - 1);  // #3 直接開第 3 步

function el(tag, attrs, parent) {
  const e = document.createElementNS(NS, tag);
  for (const [k, v] of Object.entries(attrs)) e.setAttribute(k, v);
  (parent || ov).appendChild(e);
  return e;
}

function curve(a, b) {
  // 二次貝茲曲線，往垂直方向彎一點，看起來像「拿起來放過去」
  const mx = (a[0] + b[0]) / 2, my = (a[1] + b[1]) / 2;
  const dx = b[0] - a[0], dy = b[1] - a[1], len = Math.hypot(dx, dy) || 1;
  const k = Math.min(120, len * 0.25);
  return `M${a[0]} ${a[1]} Q${mx - dy / len * k} ${my + dx / len * k} ${b[0]} ${b[1]}`;
}

function draw() {
  const s = D.steps[i];
  ov.replaceChildren();
  // 壓暗其他零件：遮罩挖空 A、B 兩個零件
  const defs = el("defs", {});
  const mask = el("mask", {id: "hole"}, defs);
  el("rect", {width: D.w, height: D.h, fill: "#fff"}, mask);
  for (const name of [s.part_a, s.part_b]) {
    const sh = D.shapes[name];
    if (sh) el("path", {d: sh.d, transform: sh.transform, fill: "#000", "fill-rule": "evenodd"}, mask);
  }
  el("rect", {width: D.w, height: D.h, fill: "rgba(0,0,0,.55)", mask: "url(#hole)"});
  for (const [name, cls] of [[s.part_a, "a"], [s.part_b, "b"]]) {
    const sh = D.shapes[name];
    if (sh) el("path", {d: sh.d, transform: sh.transform, class: "part " + cls, "vector-effect": "non-scaling-stroke"});
  }
  if (s.a && s.b) {
    const d = curve(s.a, s.b);
    el("path", {d, class: "flow", id: "route"});
    el("circle", {cx: s.b[0], cy: s.b[1], r: 22, class: "ring"});
    const dot = el("circle", {r: 11, fill: "var(--a)", stroke: "#fff", "stroke-width": 3});
    const mv = el("animateMotion", {dur: "1.8s", repeatCount: "indefinite", path: d,
      keyPoints: "0;1", keyTimes: "0;1", calcMode: "spline", keySplines: ".45 0 .25 1"}, dot);
    mv.beginElement && mv.beginElement();
  }
  $("count").textContent = `步驟 ${i + 1} / ${D.steps.length}`;
  $("pa").textContent = s.part_a;
  $("pb").textContent = s.part_b;
  $("inst").textContent = s.instruction;
  $("why").textContent = s.why;
  $("conf").textContent = CONF[s.confidence] || "";
  $("prev").disabled = i === 0;
  $("next").textContent = i === D.steps.length - 1 ? "完成" : "下一步";
  $("done").style.display = "none";
}

$("prev").onclick = () => { if (i > 0) { i--; draw(); } };
$("next").onclick = () => {
  if (i < D.steps.length - 1) { i++; draw(); }
  else { ov.replaceChildren(); $("done").style.display = "block"; }
};
draw();
</script>
</body></html>
"""


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("image")
    ap.add_argument("--reuse", action="store_true", help="用上次的 AI 結果重畫網頁，不呼叫 API")
    ap.add_argument("--resam", action="store_true", help="沿用上次的步驟和定位，只重跑 SAM 輪廓")
    args = ap.parse_args()

    img = load_image(args.image)
    stem = Path(args.image).stem
    (ROOT / "out").mkdir(exist_ok=True)
    cache = ROOT / "out" / f"{stem}_guide.json"

    if args.reuse:
        saved = json.loads(cache.read_text(encoding="utf-8"))
        plan, located, shapes = saved["plan"], saved["located"], saved["shapes"]
    elif args.resam:
        saved = json.loads(cache.read_text(encoding="utf-8"))
        plan, located = saved["plan"], saved["located"]
        shapes, t3 = segment_parts(load_api_key(), img, located["parts"])
        print(f"SAM 3.1：{t3:.1f} 秒，成功 {len(shapes)} 個")
        cache.write_text(json.dumps({"plan": plan, "located": located, "shapes": shapes}, ensure_ascii=False),
                         encoding="utf-8")
    else:
        key = load_api_key()
        print("1/3 Muse Spark 推理組裝步驟…")
        plan, t1 = plan_assembly(key, img)
        print(f"    {t1:.1f} 秒，{len(plan['parts'])} 個零件、{len(plan['steps'])} 個步驟")
        print("2/3 Muse Spark 標出零件和接口位置…")
        located, t2 = locate(key, img, plan)
        print(f"    {t2:.1f} 秒")
        parts = located["parts"]
        print(f"3/3 SAM 3.1 描出 {len(parts)} 個零件的輪廓…")
        shapes, t3 = segment_parts(key, img, parts)
        print(f"    {t3:.1f} 秒，成功 {len(shapes)} 個")
        missing = [p["target"] for p in parts if p["target"] not in shapes]
        if missing:
            print(f"    沒描到：{'、'.join(missing)}（網頁上這些零件只會顯示箭頭）")
        cache.write_text(json.dumps({"plan": plan, "located": located, "shapes": shapes}, ensure_ascii=False),
                         encoding="utf-8")
        print(f"總共 {t1 + t2 + t3:.1f} 秒")

    for n, s in enumerate(plan["steps"], 1):
        print(f"  步驟 {n}：{s['part_a']} → {s['part_b']}｜{s['instruction']}")
    out = ROOT / "out" / f"{stem}_guide.html"
    render_html(img, plan, located, shapes, out)
    print(f"引導網頁：{out}")


if __name__ == "__main__":
    main()
