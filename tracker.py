"""即時追蹤：手機每 2–3 秒送一格畫面和上一次的零件位置，這裡用 SAM 3.1 重新描出輪廓。

手機端會先用畫面比對估算鏡頭平移（static/motion.js），送來的框已經是換算到這一格的位置。
零件被拿起來移動、或鏡頭移動太多時，框會對不上：輪廓不見、面積暴增暴減、或描到背景，
這些情況改用 Muse Spark 依零件名稱重新找，再交給 SAM。
"""
import time

import sam_fast
from meta_sam_parser import FramePrompt, PromptBox, PromptObject, decode_mask_to_raster, decode_mask_to_svg_path
from PIL import Image

from build_guide import LEAK_FILL, clean_path, fill_ratio, leak_retry_prompt
from test_grounding import spark_ground
from test_pipeline import sam_with_prompts, spark_boxes_to_pixels

TRACK_PAD = 0.1       # 鏡頭平移手機端已補償，只留一點餘裕給零件本身移動；太大會描到旁邊的線
AREA_GROW = 1.6       # 面積比上次大 1.6 倍以上，多半是連背景一起描進去（斜放的細長零件很常發生）
AREA_SHRINK = 3.0     # 小 3 倍以上視為跟丟（零件一部分出畫面時會變小，所以放寬）
RELOCATE_EFFORT = None  # 重新找零件用預設推理強度（約 10 秒）；low 比較快但框常偏掉


def padded(box: list[float], size: tuple[int, int]) -> PromptBox:
    W, H = size
    x1, y1, x2, y2 = box
    px, py = (x2 - x1) * TRACK_PAD, (y2 - y1) * TRACK_PAD
    return PromptBox(left=max(0, int(x1 - px)), top=max(0, int(y1 - py)),
                     right=min(W, int(x2 + px)), bottom=min(H, int(y2 + py)))


def mask_info(rec) -> tuple[list[float], float]:
    """遮罩在整張畫面上的外框（像素）和面積（像素數）。"""
    raster = sam_fast.raster(rec.mask)
    w, h = rec.mask.width, rec.mask.height
    m = Image.frombytes("L", (w, h), raster.translate(sam_fast._TO_255))
    bbox = m.getbbox()
    sx = (rec.bounds.right - rec.bounds.left) / w
    sy = (rec.bounds.bottom - rec.bounds.top) / h
    area = sum(raster) * sx * sy
    if not bbox:
        return [], 0.0
    return [rec.bounds.left + bbox[0] * sx, rec.bounds.top + bbox[1] * sy,
            rec.bounds.left + bbox[2] * sx, rec.bounds.top + bbox[3] * sy], area


def shape_of(rec) -> dict:
    sx = (rec.bounds.right - rec.bounds.left) / rec.mask.width
    sy = (rec.bounds.bottom - rec.bounds.top) / rec.mask.height
    return {"transform": f"translate({rec.bounds.left} {rec.bounds.top}) scale({sx} {sy})",
            "d": clean_path(decode_mask_to_svg_path(rec.mask))}


def segment(key: str, img, prompts: dict[str, PromptBox]) -> dict[str, object]:
    """一次 SAM 呼叫描出多個零件；描到背景的用中心點重描。回傳 {零件名稱: 遮罩紀錄}。"""
    if not prompts:
        return {}
    names = list(prompts)
    result, _ = sam_with_prompts(key, img, [
        PromptObject(id=i + 1, prompts=[FramePrompt(frame=0, box=prompts[n])]) for i, n in enumerate(names)])
    recs = {names[int(r.object_id) - 1]: r for r in result.records if r.kind == "mask"}
    leaky = [n for n, r in recs.items() if fill_ratio(r, prompts[n]) > LEAK_FILL]
    if leaky:
        retry, _ = sam_with_prompts(key, img, [leak_retry_prompt(names.index(n), prompts[n]) for n in leaky])
        for r in retry.records:
            if r.kind == "mask":
                recs[names[int(r.object_id) - 1]] = r
    return recs


def track(key: str, img, parts: list[dict]) -> dict:
    """parts: [{name, look?, box: [x1,y1,x2,y2] 像素或 null, area: 上次面積或 null}]

    描到背景（面積暴增）的零件放進 kept：手機繼續用上一次的輪廓，不用重新找。"""
    t0 = time.perf_counter()
    out = {"shapes": {}, "boxes": {}, "areas": {}, "relocated": [], "lost": [], "kept": []}

    def accept(name, rec):
        box, area = mask_info(rec)
        out["shapes"][name], out["boxes"][name], out["areas"][name] = shape_of(rec), box, area

    by_name = {p["name"]: p for p in parts}
    first = segment(key, img, {p["name"]: padded(p["box"], img.size) for p in parts if p.get("box")})
    redo = [p["name"] for p in parts if not p.get("box")]
    for name, p in by_name.items():
        if not p.get("box"):
            continue
        rec = first.get(name)
        if rec is None:
            redo.append(name)
            continue
        _, area = mask_info(rec)
        prev = p.get("area")
        if prev and area >= prev * AREA_GROW:
            out["kept"].append(name)
        elif area <= 0 or (prev and area <= prev / AREA_SHRINK):
            redo.append(name)
        else:
            accept(name, rec)

    if redo:
        # 跟丟的零件用名稱（加外觀描述）重新找
        targets = [f"{n}（{by_name[n].get('look', '')}）" if by_name[n].get("look") else n for n in redo]
        found, _ = spark_ground(key, img, targets, RELOCATE_EFFORT)
        located = [{"target": n, "found": o["found"], "box": o["box"]} for n, o in zip(redo, found["objects"])]
        boxes = {redo[i]: box for i, _, box in spark_boxes_to_pixels(located, img.size)}
        second = segment(key, img, boxes)
        for n in redo:
            if n in second:
                accept(n, second[n])
                out["relocated"].append(n)
            else:
                out["lost"].append(n)

    out["seconds"] = round(time.perf_counter() - t0, 2)
    return out
