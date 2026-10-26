"""測法 A：不給組裝方法，讓 Muse Spark 只看照片自己推理怎麼組，並畫出「A 零件 → B 接口」箭頭。

用法：
  .venv/Scripts/python test_assembly.py samples/mic_parts.jpg
  .venv/Scripts/python test_assembly.py samples/mic_parts.jpg --manual samples/mic_manual.jpg

輸出：out/<圖名>_assembly.jpg（每一步一個顏色的箭頭）與 out/<圖名>_assembly.json
"""
import argparse
import json
import math
import time
from pathlib import Path

import requests
from PIL import ImageDraw

from test_grounding import API_BASE, COLORS, ROOT, SPARK_MODEL, font, load_api_key, load_image, to_data_url

POINT = {"type": "array", "items": {"type": "integer"}, "description": "[x,y]，0-1000 正規化座標；看不到給 []"}
SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["goal", "steps", "unused_parts"],
    "properties": {
        "goal": {"type": "string", "description": "組好之後是什麼樣子"},
        "steps": {"type": "array", "items": {
            "type": "object",
            "additionalProperties": False,
            "required": ["part_a", "part_b", "instruction", "a_point", "b_point", "confidence", "why"],
            "properties": {
                "part_a": {"type": "string", "description": "這一步拿起來的零件"},
                "part_b": {"type": "string", "description": "要接上去的零件或已組好的部分"},
                "instruction": {"type": "string", "description": "怎麼接（插入、鎖入、夾住、套上…，方向）"},
                "a_point": {**POINT, "description": "part_a 上要接的那一端/接口；0-1000 座標"},
                "b_point": {**POINT, "description": "part_b 上對應的接口；0-1000 座標"},
                "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
                "why": {"type": "string", "description": "從照片中看到什麼特徵推斷的（螺紋、孔徑、形狀…）"},
            },
        }},
        "unused_parts": {"type": "array", "items": {"type": "string"},
                         "description": "照片中有、但組裝實體結構用不到的零件（例如線材、電子設備）"},
    },
}


def ask(key: str, parts_img, manual_img, effort: str | None):
    text = (
        "The photo shows the unassembled parts of a product. Without any assembly instructions, work out "
        "how to physically assemble them into the finished product. Give the steps in order. For each step "
        "name the part being picked up (part_a) and what it attaches to (part_b), how it attaches, and locate "
        "in the PARTS photo the connection point on each part using the normalized 0-1000 grid ((0,0) = top-left). "
        "Base your reasoning on visible features such as threads, sockets, clamps and sizes. "
        "Only include physical mounting steps; list cables and electronics in unused_parts. "
        "Answer in Traditional Chinese."
    )
    content = [{"type": "text", "text": text}, {"type": "image_url", "image_url": {"url": to_data_url(parts_img)}}]
    if manual_img is not None:
        content[0]["text"] += " The second image is a page of the product manual; use it if it helps."
        content.append({"type": "image_url", "image_url": {"url": to_data_url(manual_img)}})
    body = {
        "model": SPARK_MODEL,
        "messages": [{"role": "user", "content": content}],
        "response_format": {"type": "json_schema", "json_schema": {"name": "Assembly", "strict": True, "schema": SCHEMA}},
    }
    if effort:
        body["reasoning_effort"] = effort
    t0 = time.perf_counter()
    r = requests.post(f"{API_BASE}/chat/completions", json=body, timeout=300,
                      headers={"Authorization": f"Bearer {key}"})
    if not r.ok:
        raise SystemExit(f"Muse Spark 呼叫失敗 HTTP {r.status_code}：{r.text[:800]}")
    return json.loads(r.json()["choices"][0]["message"]["content"]), time.perf_counter() - t0


def arrow(d: ImageDraw.ImageDraw, a, b, color, width=5):
    d.line([a, b], fill=color, width=width)
    ang = math.atan2(b[1] - a[1], b[0] - a[0])
    head = [b, (b[0] - 22 * math.cos(ang - 0.4), b[1] - 22 * math.sin(ang - 0.4)),
            (b[0] - 22 * math.cos(ang + 0.4), b[1] - 22 * math.sin(ang + 0.4))]
    d.polygon(head, fill=color)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("image")
    ap.add_argument("--manual", help="說明書照片（選用）")
    ap.add_argument("--effort", choices=["minimal", "low", "medium", "high"])
    args = ap.parse_args()

    key = load_api_key()
    img = load_image(args.image)
    manual = load_image(args.manual) if args.manual else None
    result, dt = ask(key, img, manual, args.effort)
    print(f"Muse Spark 推理：{dt:.1f} 秒（{'有' if manual else '沒有'}給說明書）")
    print(f"目標：{result['goal']}\n")

    canvas = img.copy()
    d = ImageDraw.Draw(canvas)
    W, H = canvas.size
    f = font(max(18, W // 40))
    for i, s in enumerate(result["steps"]):
        c = COLORS[i % len(COLORS)]
        print(f"步驟 {i + 1} [{s['confidence']}]：{s['part_a']} → {s['part_b']}")
        print(f"   怎麼接：{s['instruction']}")
        print(f"   依據：{s['why']}")
        if len(s["a_point"]) == 2 and len(s["b_point"]) == 2:
            a = (s["a_point"][0] * W / 1000, s["a_point"][1] * H / 1000)
            b = (s["b_point"][0] * W / 1000, s["b_point"][1] * H / 1000)
            arrow(d, a, b, c)
            d.ellipse([a[0] - 8, a[1] - 8, a[0] + 8, a[1] + 8], fill=c, outline="white", width=2)
            d.ellipse([b[0] - 14, b[1] - 14, b[0] + 14, b[1] + 14], outline=c, width=4)
            d.text((a[0] + 10, a[1] - 10), str(i + 1), fill=c, font=f, stroke_width=3, stroke_fill="white")
    print(f"\n用不到的零件：{'、'.join(result['unused_parts']) or '無'}")

    stem = Path(args.image).stem + ("_withmanual" if manual else "")
    canvas.save(ROOT / "out" / f"{stem}_assembly.jpg", quality=90)
    (ROOT / "out" / f"{stem}_assembly.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"箭頭圖：out/{stem}_assembly.jpg")


if __name__ == "__main__":
    main()
