"""一段影片錄全部零件 → 每個零件最清楚的一格，之後接 parts_guide.analyze。

零件放在桌上，鏡頭一個一個拍過去，每個停 2 秒左右（不用拿起來轉：手拿著轉，SAM 常把手或地板一起剪進去）。
Muse Spark 看整段影片，標出每個零件出現的時間段和最清楚、沒被手擋住的時間點；從整段影片已經抽好的格子
裡挑那一刻附近最清楚的一格當零件照片。不再切段、轉檔、上傳（14 個零件原本要多花 1–2 分鐘）。
推理步驟時整段影片還是會送給 Spark 看。

用法（測試）：
  .venv/Scripts/python video_split.py all.mp4
"""
import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from clip import FPS, Clip, sharpest, transcode, upload
from parts_guide import LANGUAGES, spark_json
from test_grounding import load_api_key

MAX_SECONDS = 150   # 整段影片最長取幾秒
SEG_MIN = 1.0       # 時間段至少幾秒
SEG_MAX = 10.0      # 時間段最長幾秒：同一個零件拍太久，只取最清楚那一刻前後
EDGE = 0.3          # 時間段前後各多留一點，Spark 標的時間可能差一點

SPLIT_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["parts"],
    "properties": {
        "parts": {"type": "array", "items": {
            "type": "object",
            "additionalProperties": False,
            "required": ["name", "look", "start", "end", "best"],
            "properties": {
                "name": {"type": "string", "description": "短名稱"},
                "look": {"type": "string", "description": "外觀（顏色、形狀、材質、明顯特徵）"},
                "start": {"type": "number", "description": "這個零件被拿起來展示的開始時間（秒，從影片開頭算）"},
                "end": {"type": "number", "description": "展示結束、放下的時間（秒）"},
                "best": {"type": "number", "description": "零件最清楚、最完整入鏡的時間點（秒）"},
            },
        }},
    },
}

SPLIT_PROMPT = (
    "This video ({seconds:.0f} seconds long) shows the parts of a product before assembly, one at a time (usually "
    "lying on a table while the camera pauses on each; sometimes held up). List every distinct part in the order it "
    "is shown, with the time range in seconds from the start of the video when it is being shown, and the best "
    "moment: the camera is steady, the whole part is in view, and no hand covers or holds it. If the same part is "
    "shown twice, list it once with its clearest showing. Each moment has one part in focus: list only that one, "
    "not other parts at the edges or in the background (they get their own turn). "
    "Do not list hands, the table or the floor. "
    "Write name and look in {language}."
)


def split_video(api_key: str, raw: bytes, log=print, lang: str = "zh") -> list[Clip]:
    en = lang == "en"
    data, frames = transcode(raw, MAX_SECONDS)
    seconds = len(frames) / FPS
    log(f"AI is finding the parts in the video ({seconds:.0f} s)" if en else f"AI 看整段影片找零件（{seconds:.0f} 秒）")
    fid = upload(data, api_key, "all_parts.mp4")
    content = [{"type": "text", "text": SPLIT_PROMPT.format(seconds=seconds, language=LANGUAGES[lang])},
               {"type": "file", "file": {"file_id": fid}}]
    found = spark_json(api_key, content, SPLIT_SCHEMA, "PartsInVideo")["parts"]
    segs = []
    for p in found:
        start, end = max(0.0, min(p["start"], p["end"]) - EDGE), min(seconds, max(p["start"], p["end"]) + EDGE)
        best = min(max(p["best"], start), end)
        if end - start > SEG_MAX:  # 拿太久：只取最清楚那一刻前後
            start, end = max(0.0, best - SEG_MAX / 2), min(seconds, best + SEG_MAX / 2)
        if end - start < SEG_MIN:
            start, end = max(0.0, best - SEG_MIN / 2), min(seconds, best + SEG_MIN / 2)
        segs.append((p, start, end, best))
    if en:
        log(f"Found {len(segs)} parts: " + ", ".join(f"{p['name']} ({s:.0f}–{e:.0f} s)" for p, s, e, _ in segs))
    else:
        log(f"找到 {len(segs)} 個零件：" + "、".join(f"{p['name']}（{s:.0f}–{e:.0f} 秒）" for p, s, e, _ in segs))

    def pick(seg) -> Clip:
        _, start, end, best = seg
        # Spark 標的時間點附近 ±0.5 秒、而且在時間段內，挑最清楚的一格
        b = round(best * FPS)
        lo, hi = max(round(start * FPS), b - FPS // 2), min(round(end * FPS), b + FPS // 2 + 1, len(frames))
        k = sharpest(frames, min(lo, len(frames) - 1), hi)
        return Clip(frames=frames[k:k + 1], key=0, file_id=None, source=(fid, start, end))

    with ThreadPoolExecutor(4) as ex:
        return list(ex.map(pick, segs))


def main():
    import sys
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video")
    ap.add_argument("--out", help="把每個零件最清楚那一格存到這個資料夾")
    args = ap.parse_args()
    clips = split_video(load_api_key(), Path(args.video).read_bytes(), log=lambda m: print(f"  {m}"))
    for i, c in enumerate(clips, 1):
        print(f"  零件 {i}：{c.source[1]:.1f}–{c.source[2]:.1f} 秒")
        if args.out:
            Path(args.out).mkdir(parents=True, exist_ok=True)
            c.photo.save(Path(args.out) / f"part{i}.jpg")


if __name__ == "__main__":
    main()
