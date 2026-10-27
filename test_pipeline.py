"""Muse Spark 框出零件 → SAM 3.1 用框描出精準輪廓 → 輸出發光效果圖與 SVG 輪廓。

用法：
  .venv/Scripts/python test_pipeline.py samples/mic_parts.jpg --target "防震架" --target "麥克風夾（塑膠）"

輸出：
  out/<圖名>_glow.jpg      發光輪廓效果圖
  out/<圖名>_glow.html     在瀏覽器（含手機）看的脈動發光預覽
  out/<圖名>_masks.json    每個零件的 SVG 輪廓路徑（之後給手機網頁用）
"""
import argparse
import asyncio
import json
import time
from pathlib import Path

import requests
from meta_sam_parser import (FramePrompt, PromptBox, PromptObject, build_object_prompt,
                             decode_mask_to_raster, decode_mask_to_svg_path,
                             image_segmentation_format, parse_responses_stream)
from PIL import Image, ImageChops, ImageDraw, ImageFilter

from test_grounding import API_BASE, COLORS, ROOT, SAM_MODEL, font, load_api_key, load_image, spark_ground, to_data_url

PAD = 0.03  # Muse Spark 的框常常切到零件邊緣，交給 SAM 前往外擴一點


def spark_boxes_to_pixels(objects: list[dict], size: tuple[int, int]) -> list[tuple[int, dict, PromptBox]]:
    W, H = size
    out = []
    for i, obj in enumerate(objects):
        if not obj["found"] or len(obj["box"]) != 4:
            continue
        x1, y1, x2, y2 = obj["box"]
        px, py = PAD * W, PAD * H
        box = PromptBox(left=max(0, int(x1 * W / 1000 - px)), top=max(0, int(y1 * H / 1000 - py)),
                        right=min(W, int(x2 * W / 1000 + px)), bottom=min(H, int(y2 * H / 1000 + py)))
        out.append((i, obj, box))
    return out


async def _sse_events(resp: requests.Response):
    for line in resp.iter_lines(decode_unicode=True):
        if line and line.startswith("data:"):
            payload = line[5:].strip()
            if payload and payload != "[DONE]":
                yield json.loads(payload)


def sam_with_boxes(key: str, img: Image.Image, boxes: list[tuple[int, dict, PromptBox]]):
    return sam_with_prompts(key, img, [
        PromptObject(id=i + 1, prompts=[FramePrompt(frame=0, box=box)]) for i, _, box in boxes])


def sam_with_prompts(key: str, img: Image.Image, objects: list[PromptObject]):
    prompt = build_object_prompt(size=img.size, objects=objects)
    body = {
        "model": SAM_MODEL,
        "stream": True,
        "input": [{"type": "message", "role": "user", "content": [
            {"type": "input_text", "text": prompt},
            {"type": "input_image", "image_url": to_data_url(img)},
        ]}],
    }
    t0 = time.perf_counter()
    with requests.post(f"{API_BASE}/responses", json=body, stream=True, timeout=120,
                       headers={"Authorization": f"Bearer {key}"}) as r:
        if not r.ok:
            raise SystemExit(f"SAM 呼叫失敗 HTTP {r.status_code}：{r.text[:800]}")

        async def run():
            parsed = parse_responses_stream(_sse_events(r), image_segmentation_format())
            return await parsed.final_result()

        result = asyncio.run(run())
    return result, time.perf_counter() - t0


def mask_image(rec, size: tuple[int, int]) -> Image.Image:
    """把 box-local 的遮罩貼回整張圖大小的 L 模式圖。"""
    m = rec.mask
    raster = decode_mask_to_raster(m)
    local = Image.frombytes("L", (m.width, m.height), bytes(v * 255 for v in raster))
    b = rec.bounds
    w, h = max(1, round(b.right - b.left)), max(1, round(b.bottom - b.top))
    if local.size != (w, h):
        local = local.resize((w, h), Image.NEAREST)
    full = Image.new("L", size, 0)
    full.paste(local, (round(b.left), round(b.top)))
    return full


def draw_glow(img: Image.Image, layers: list[tuple[str, str, Image.Image]], out_path: Path):
    base = img.convert("RGBA")
    dim = Image.new("RGBA", base.size, (0, 0, 0, 110))  # 壓暗背景讓發光更明顯
    canvas = Image.alpha_composite(base, dim)
    f = font(max(16, base.size[0] // 45))
    for label, color, m in layers:
        edge = ImageChops.subtract(m.filter(ImageFilter.MaxFilter(9)), m.filter(ImageFilter.MinFilter(3)))
        glow = edge.filter(ImageFilter.GaussianBlur(10))
        for alpha_mask, opacity in ((glow, 1.0), (edge, 1.0)):
            layer = Image.new("RGBA", base.size, color)
            layer.putalpha(alpha_mask.point(lambda v: int(v * opacity)))
            canvas = Image.alpha_composite(canvas, layer)
        # 零件本身恢復原亮度
        canvas.paste(base, (0, 0), m)
        bbox = m.getbbox()
        if bbox:
            ImageDraw.Draw(canvas).text((bbox[0], max(0, bbox[1] - f.size - 6)), label, fill=color, font=f)
    canvas.convert("RGB").save(out_path, quality=90)


def write_html(img_name: str, size: tuple[int, int], items: list[dict], out_path: Path):
    W, H = size
    paths = "\n".join(
        f'<path class="glow" d="{it["svg_path"]}" transform="translate({it["left"]} {it["top"]}) scale({it["scale"][0]} {it["scale"][1]})" '
        f'style="--c:{it["color"]}"><title>{it["label"]}</title></path>' for it in items)
    out_path.write_text(f"""<!doctype html><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Glow preview</title>
<style>
body{{margin:0;background:#111}} .wrap{{position:relative;max-width:100vw}}
img,svg{{width:100%;display:block}} svg{{position:absolute;inset:0}}
.glow{{fill:none;stroke:var(--c);stroke-width:4;filter:drop-shadow(0 0 6px var(--c)) drop-shadow(0 0 14px var(--c));
  animation:pulse 1.4s ease-in-out infinite}}
@keyframes pulse{{50%{{stroke-width:9;opacity:.55}}}}
</style>
<div class="wrap"><img src="../samples/{img_name}">
<svg viewBox="0 0 {W} {H}" preserveAspectRatio="none">{paths}</svg></div>
""", encoding="utf-8")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("image")
    ap.add_argument("--target", action="append", required=True)
    ap.add_argument("--effort", choices=["minimal", "low", "medium", "high"])
    args = ap.parse_args()

    key = load_api_key()
    img = load_image(args.image)
    stem = Path(args.image).stem
    (ROOT / "out").mkdir(exist_ok=True)

    spark, t_spark = spark_ground(key, img, args.target, args.effort)
    print(f"Muse Spark：{t_spark:.1f} 秒")
    boxes = spark_boxes_to_pixels(spark["objects"], img.size)
    for obj in spark["objects"]:
        if not obj["found"]:
            print(f"  ✗ {obj['target']}：{obj['reason']}")
    if not boxes:
        raise SystemExit("Muse Spark 一個零件都沒找到")

    result, t_sam = sam_with_boxes(key, img, boxes)
    print(f"SAM 3.1：{t_sam:.1f} 秒（{len(boxes)} 個零件一次送）")
    for d in result.diagnostics:
        print(f"  [SAM 診斷] {d}")

    masks = {rec.object_id: rec for rec in result.records if rec.kind == "mask"}
    layers, items = [], []
    for i, obj, _ in boxes:
        rec = masks.get(str(i + 1))
        color = COLORS[i % len(COLORS)]
        if rec is None:
            print(f"  ✗ {obj['target']}：SAM 沒有回傳遮罩")
            continue
        m = mask_image(rec, img.size)
        area = 1 - m.histogram()[0] / (img.size[0] * img.size[1])
        print(f"  ✓ {obj['target']}：輪廓佔畫面 {area:.1%}")
        layers.append((obj["target"], color, m))
        sx, sy = (rec.bounds.right - rec.bounds.left) / rec.mask.width, (rec.bounds.bottom - rec.bounds.top) / rec.mask.height
        items.append({"label": obj["target"], "color": color, "left": rec.bounds.left, "top": rec.bounds.top,
                      "scale": [sx, sy], "svg_path": decode_mask_to_svg_path(rec.mask)})

    draw_glow(img, layers, ROOT / "out" / f"{stem}_glow.jpg")
    (ROOT / "out" / f"{stem}_masks.json").write_text(
        json.dumps({"size": img.size, "items": items}, ensure_ascii=False), encoding="utf-8")
    write_html(Path(args.image).name, img.size, items, ROOT / "out" / f"{stem}_glow.html")
    print(f"總延遲 {t_spark + t_sam:.1f} 秒；效果圖 out/{stem}_glow.jpg，預覽 out/{stem}_glow.html")


if __name__ == "__main__":
    main()
