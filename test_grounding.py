"""WORLD OVERLAY 風險驗證：Muse Spark / SAM 3.1 能不能在照片裡找到指定的 IKEA 零件？

用法：
  .venv/Scripts/python test_grounding.py samples/parts.jpg --target "最長的木板" --target "木釘"
  .venv/Scripts/python test_grounding.py samples/parts.jpg --target "木釘" --sam "wooden dowel" --runs 3

結果圖會存在 out/，終端機會印出每次呼叫的延遲。
"""
import argparse
import base64
import io
import json
import os
import re
import statistics
import sys
import time
from pathlib import Path

import requests
from PIL import Image, ImageDraw, ImageFont, ImageOps

API_BASE = "https://api.meta.ai/v1"
SPARK_MODEL = "muse-spark-1.3"
SAM_MODEL = "sam-3.1"
MAX_SIDE = 1280  # 約 1,300–1,500 input tokens；更大只會更貴更慢
ROOT = Path(__file__).parent
COLORS = ["#ff3b30", "#34c759", "#007aff", "#ff9500", "#af52de", "#00c7be"]

sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")


def load_api_key() -> str:
    key = os.environ.get("MODEL_API_KEY")
    env_file = ROOT / ".env"
    if not key and env_file.exists():
        for line in env_file.read_text(encoding="utf-8").splitlines():
            if line.strip().startswith("MODEL_API_KEY="):
                key = line.split("=", 1)[1].strip().strip('"')
    if not key:
        sys.exit("找不到 MODEL_API_KEY：請在 .env 寫入 MODEL_API_KEY=你的key（參考 .env.example）")
    return key


def load_image(path: str) -> Image.Image:
    img = ImageOps.exif_transpose(Image.open(path)).convert("RGB")  # 手機照片要依 EXIF 轉正
    img.thumbnail((MAX_SIDE, MAX_SIDE))
    return img


def to_data_url(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def font(size: int):
    for f in ("C:/Windows/Fonts/msjh.ttc", "C:/Windows/Fonts/mingliu.ttc"):
        if Path(f).exists():
            return ImageFont.truetype(f, size)
    return ImageFont.load_default()


# ---------- Muse Spark grounding ----------

GROUNDING_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["scene", "objects"],
    "properties": {
        "scene": {"type": "string", "description": "一句話描述畫面"},
        "objects": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["target", "found", "box", "point", "confidence", "reason"],
                "properties": {
                    "target": {"type": "string"},
                    "found": {"type": "boolean"},
                    "box": {"type": "array", "items": {"type": "integer"},
                            "description": "[x1,y1,x2,y2]，0-1000 正規化座標；找不到給 []"},
                    "point": {"type": "array", "items": {"type": "integer"},
                              "description": "[x,y]，0-1000 正規化座標；找不到給 []"},
                    "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
                    "reason": {"type": "string", "description": "用什麼特徵判斷的（形狀、長度、孔位…）"},
                },
            },
        },
    },
}


def spark_ground(key: str, img: Image.Image, targets: list[str], effort: str | None = None) -> tuple[dict, float]:
    prompt = (
        "You are helping a user assemble IKEA furniture. Locate each target part in the image.\n"
        "Return one entry per target, in the same order. Use the normalized 0-1000 grid "
        "where (0,0) is the top-left corner. If several pieces match, box the single best one. "
        "If a target is not visible, set found=false and use empty arrays. "
        "Write scene and reason in Traditional Chinese.\n\nTargets:\n"
        + "\n".join(f"{i + 1}. {t}" for i, t in enumerate(targets))
    )
    body = {
        "model": SPARK_MODEL,
        "messages": [{"role": "user", "content": [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": to_data_url(img)}},
        ]}],
        "response_format": {"type": "json_schema", "json_schema": {
            "name": "Grounding", "strict": True, "schema": GROUNDING_SCHEMA}},
    }
    if effort:
        body["reasoning_effort"] = effort
    t0 = time.perf_counter()
    r = requests.post(f"{API_BASE}/chat/completions", json=body, timeout=120,
                      headers={"Authorization": f"Bearer {key}"})
    dt = time.perf_counter() - t0
    if not r.ok:
        sys.exit(f"Muse Spark 呼叫失敗 HTTP {r.status_code}：{r.text[:800]}")
    data = r.json()
    return json.loads(data["choices"][0]["message"]["content"]), dt


def draw_spark(img: Image.Image, result: dict, out_path: Path):
    canvas = img.copy()
    d = ImageDraw.Draw(canvas)
    W, H = canvas.size
    f = font(max(16, W // 45))
    sx = lambda v: max(0, min(W, v * W / 1000))
    sy = lambda v: max(0, min(H, v * H / 1000))
    for i, obj in enumerate(result["objects"]):
        if not obj["found"]:
            continue
        c = COLORS[i % len(COLORS)]
        if len(obj["box"]) == 4:
            x1, y1, x2, y2 = obj["box"]
            d.rectangle([sx(x1), sy(y1), sx(x2), sy(y2)], outline=c, width=4)
            d.text((sx(x1) + 4, sy(y1) + 4), f"{i + 1}. {obj['target']}", fill=c, font=f)
        if len(obj["point"]) == 2:
            px, py = sx(obj["point"][0]), sy(obj["point"][1])
            d.ellipse([px - 9, py - 9, px + 9, py + 9], fill=c, outline="white", width=2)
    canvas.save(out_path, quality=90)


# ---------- SAM 3.1 ----------

BOX_RE = re.compile(r"<\|box;x1=(\d+);y1=(\d+);x2=(\d+);y2=(\d+);w=(\d+);h=(\d+)(?:;c=([\d.]+))?\|>")


def response_text(data: dict) -> str:
    if data.get("output_text"):
        return data["output_text"]
    parts = []
    for item in data.get("output", []):
        for c in item.get("content", []) or []:
            if c.get("type") in ("output_text", "text"):
                parts.append(c.get("text", ""))
    return "".join(parts)


def sam_segment(key: str, img: Image.Image, phrase: str) -> tuple[list, str, float]:
    body = {
        "model": SAM_MODEL,
        "input": [{"type": "message", "role": "user", "content": [
            {"type": "input_text", "text": phrase},
            {"type": "input_image", "image_url": to_data_url(img)},
        ]}],
        "metadata": {"include_confidence": "true"},
    }
    t0 = time.perf_counter()
    r = requests.post(f"{API_BASE}/responses", json=body, timeout=120,
                      headers={"Authorization": f"Bearer {key}"})
    dt = time.perf_counter() - t0
    if not r.ok:
        print(f"  SAM 呼叫失敗 HTTP {r.status_code}：{r.text[:800]}")
        return [], "", dt
    text = response_text(r.json())
    boxes = [tuple(int(v) for v in m.groups()[:6]) + (float(m.group(7)) if m.group(7) else None,)
             for m in BOX_RE.finditer(text)]
    return boxes, text, dt


def draw_sam(img: Image.Image, boxes: list, phrase: str, out_path: Path):
    canvas = img.copy()
    d = ImageDraw.Draw(canvas)
    W, H = canvas.size
    f = font(max(16, W // 45))
    for x1, y1, x2, y2, w, h, conf in boxes:
        k_x, k_y = W / w, H / h
        d.rectangle([x1 * k_x, y1 * k_y, x2 * k_x, y2 * k_y], outline="#00c7be", width=4)
        label = phrase if conf is None else f"{phrase} {conf:.2f}"
        d.text((x1 * k_x + 4, y1 * k_y + 4), label, fill="#00c7be", font=f)
    canvas.save(out_path, quality=90)


# ---------- main ----------

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("image")
    ap.add_argument("--target", action="append", default=[], help="要找的零件（中文描述即可，可重複）")
    ap.add_argument("--sam", action="append", default=[], help="SAM 3.1 用的英文名詞片語，例如 'wooden dowel'（可重複）")
    ap.add_argument("--runs", type=int, default=1, help="重複呼叫幾次來量延遲")
    ap.add_argument("--effort", choices=["minimal", "low", "medium", "high"],
                    help="Muse Spark 推理強度；越低越快（預設由模型自己決定）")
    args = ap.parse_args()
    if not args.target and not args.sam:
        ap.error("至少給一個 --target 或 --sam")

    key = load_api_key()
    img = load_image(args.image)
    stem = Path(args.image).stem
    (ROOT / "out").mkdir(exist_ok=True)
    print(f"圖片 {args.image} → 縮放後 {img.size[0]}x{img.size[1]}")

    if args.target:
        print(f"\n== Muse Spark grounding（{SPARK_MODEL}，effort={args.effort or '預設'}）==")
        times = []
        for run in range(args.runs):
            result, dt = spark_ground(key, img, args.target, args.effort)
            times.append(dt)
            print(f"  第 {run + 1} 次：{dt:.1f} 秒")
        print(f"  畫面：{result['scene']}")
        for i, obj in enumerate(result["objects"]):
            mark = "✓" if obj["found"] else "✗"
            print(f"  {mark} {i + 1}. {obj['target']}  box={obj['box']} point={obj['point']} "
                  f"[{obj['confidence']}] {obj['reason']}")
        out = ROOT / "out" / f"{stem}_spark.jpg"
        draw_spark(img, result, out)
        (ROOT / "out" / f"{stem}_spark.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"  延遲 中位數 {statistics.median(times):.1f} 秒；結果圖 {out}")

    for phrase in args.sam:
        print(f"\n== SAM 3.1：'{phrase}' ==")
        times = []
        for run in range(args.runs):
            boxes, raw, dt = sam_segment(key, img, phrase)
            times.append(dt)
            print(f"  第 {run + 1} 次：{dt:.1f} 秒，找到 {len(boxes)} 個")
        if not boxes and raw:
            print(f"  （沒解析到 box，原始輸出前 500 字）{raw[:500]}")
        safe = re.sub(r"\W+", "_", phrase)
        out = ROOT / "out" / f"{stem}_sam_{safe}.jpg"
        draw_sam(img, boxes, phrase, out)
        print(f"  延遲 中位數 {statistics.median(times):.1f} 秒；結果圖 {out}")


if __name__ == "__main__":
    main()
