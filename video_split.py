"""一段影片錄全部零件 → 每個零件一段短片（Clip），之後接 parts_guide.analyze。

使用者依序把每個零件拿起來轉一下再放下。Muse Spark 看整段影片，標出每個零件出現的時間段和
最清楚的時間點；照時間段把原始影片切成每個零件一段（各自轉檔、上傳），後面的流程和一個零件
一段影片完全一樣。比起 16 段各 5 秒，多的只有這一次看整段影片（100 秒約 2.6 萬 token，約 US$0.03）。

用法（測試）：
  .venv/Scripts/python video_split.py all.mp4
"""
import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from clip import FPS, Clip, sharpest, transcode, upload
from parts_guide import spark_json
from test_grounding import load_api_key

MAX_SECONDS = 150   # 整段影片最長取幾秒
SEG_MIN = 1.0       # 每段至少幾秒（SAM 每次請求至少計 50 格，太短也不省錢）
SEG_MAX = 10.0      # 每段最長幾秒：同一個零件拿太久，只取最清楚那一刻前後
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
    "This video ({seconds:.0f} seconds long) shows the parts of a product before assembly. The person picks up the "
    "parts one at a time and turns each one in front of the camera. List every distinct part in the order it is "
    "shown, with the time range in seconds from the start of the video when it is being shown, and the moment it "
    "is seen most clearly and completely. If the same part is shown twice, list it once with its clearest showing. "
    "Ignore hands and the table. Write name and look in Traditional Chinese."
)


def split_video(api_key: str, raw: bytes, log=print) -> list[Clip]:
    data, frames = transcode(raw, MAX_SECONDS)
    seconds = len(frames) / FPS
    log(f"AI 看整段影片找零件（{seconds:.0f} 秒）")
    fid = upload(data, api_key, "all_parts.mp4")
    content = [{"type": "text", "text": SPLIT_PROMPT.format(seconds=seconds)},
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
    log(f"找到 {len(segs)} 個零件：" + "、".join(f"{p['name']}（{s:.0f}–{e:.0f} 秒）" for p, s, e, _ in segs))

    def cut(seg) -> Clip:
        _, start, end, best = seg
        # 從已經轉好的影片切（固定格率、已縮小），每段各自上傳給 SAM 和 Spark
        sub, sub_frames = transcode(data, end - start, start)
        n = len(sub_frames)
        b = min(n - 1, max(0, round((best - start) * FPS)))
        # Spark 標的時間點附近 ±1 秒挑最清楚的一格
        key = sharpest(sub_frames, max(0, b - FPS), min(n, b + FPS + 1))
        return Clip(frames=sub_frames, key=key, file_id=upload(sub, api_key), source=(fid, start, end))

    with ThreadPoolExecutor(4) as ex:
        return list(ex.map(cut, segs))


def main():
    import sys
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video")
    ap.add_argument("--out", help="把每個零件最清楚那一格存到這個資料夾")
    args = ap.parse_args()
    clips = split_video(load_api_key(), Path(args.video).read_bytes(), log=lambda m: print(f"  {m}"))
    for i, c in enumerate(clips, 1):
        print(f"  零件 {i}：{len(c.frames)} 格，最清楚第 {c.key} 格")
        if args.out:
            Path(args.out).mkdir(parents=True, exist_ok=True)
            c.photo.save(Path(args.out) / f"part{i}.jpg")


if __name__ == "__main__":
    main()
