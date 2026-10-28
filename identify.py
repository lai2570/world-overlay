"""即時辨識：鏡頭畫面裡是哪個零件（DINOv2，在這台電腦上跑，不呼叫 API）。

分析完（parts_guide.analyze）每個零件都有去背圖。DINOv2 把圖切成 16×16 塊、每塊一個特徵向量；
零件去背圖只取有零件的那幾塊。比對時，零件的每一塊在鏡頭畫面裡找最像的一塊，平均起來就是分數。
這樣零件在畫面哪裡、背景是桌面還是手都不太影響。
有些零件（黑色線材這種）對什麼都像：先算每個零件和其他零件去背圖的分數分布，比對時換成標準分數。
這台筆電 CPU 上一張約 0.3–0.5 秒。

最高和第二高差很少時（例如幾條長得很像的白線）不確定，交給 Muse Spark 確認（server.py）。

用法（測試）：
  .venv/Scripts/python identify.py out/app/<job>_parts.json 照片1.jpg 照片2.jpg ...
"""
import base64
import io
import sys
import threading
import time

import numpy as np
from PIL import Image

MODEL = "facebook/dinov2-base"  # Meta 開源的影像特徵模型（ViT-B/14）；small 快一倍但實測認錯比較多
SIZE = 224                       # 224 / 14 = 16×16 塊
GRID = SIZE // 14
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
BG = (128, 128, 128)
MIN_FG = 0.3  # 一塊裡零件佔多少以上才算零件的一塊
SURE_MARGIN = 0.8  # 第一名比第二名高多少個標準差才算確定

_lock = threading.Lock()
_model = None


def model():
    global _model
    with _lock:
        if _model is None:
            import torch
            from transformers import AutoModel
            torch.set_num_threads(4)
            _model = AutoModel.from_pretrained(MODEL).eval()
        return _model


def square(img: Image.Image, fill=BG) -> Image.Image:
    """補成正方形（不裁掉零件）再縮到 SIZE。"""
    w, h = img.size
    s = max(w, h)
    out = Image.new(img.mode, (s, s), fill if img.mode == "RGB" else 0)
    out.paste(img, ((s - w) // 2, (s - h) // 2))
    return out.resize((SIZE, SIZE), Image.BILINEAR)


def features(images: list[Image.Image]) -> np.ndarray:
    """RGB 圖 → 每張 GRID*GRID 塊的特徵（已正規化），shape (n, GRID*GRID, 384)。"""
    import torch
    x = np.stack([(np.asarray(square(im.convert("RGB")), dtype=np.float32) / 255 - MEAN) / STD for im in images])
    with torch.inference_mode():
        out = model()(pixel_values=torch.from_numpy(x).permute(0, 3, 1, 2)).last_hidden_state[:, 1:]
    f = out.numpy()
    return f / np.linalg.norm(f, axis=-1, keepdims=True)


class PartIndex:
    """一次分析的所有零件：名字 → 零件那幾塊的特徵。"""

    def __init__(self, cutouts: dict[str, Image.Image]):
        self.names = list(cutouts)
        rgbs, fgs = [], []
        for im in cutouts.values():
            im = im.convert("RGBA")
            rgb = Image.new("RGB", im.size, BG)
            rgb.paste(im, mask=im.split()[-1])
            rgbs.append(rgb)
            a = np.asarray(square(im.split()[-1]).resize((GRID, GRID), Image.BOX), dtype=np.float32) / 255
            fgs.append(a.reshape(-1) >= MIN_FG)
        feats = features(rgbs) if rgbs else np.zeros((0, GRID * GRID, 384), np.float32)
        # 去背失敗（幾乎沒有零件）的就用整張
        self.parts = [f[m] if m.sum() >= 3 else f for f, m in zip(feats, fgs)]
        # 每個零件「對什麼都像」的程度：和其他零件整張去背圖的分數平均、標準差
        self.base = []
        for n, p in enumerate(self.parts):
            others = [self.raw(p, q) for k, q in enumerate(feats) if k != n]
            self.base.append((float(np.mean(others)), float(np.std(others)) + 0.02) if others else (0.0, 1.0))

    @staticmethod
    def raw(part: np.ndarray, frame: np.ndarray) -> float:
        return float((part @ frame.T).max(axis=1).mean())

    @classmethod
    def from_result(cls, result: dict) -> "PartIndex":
        cuts = {}
        for k, url in result["images"].items():
            if k.startswith("part:"):
                cuts[k[5:]] = Image.open(io.BytesIO(base64.b64decode(url.split(",", 1)[1])))
        return cls(cuts)

    def match(self, frame: Image.Image) -> list[tuple[str, float]]:
        """鏡頭畫面 → [(零件名, 標準分數)]，分數高的在前。"""
        q = features([frame])[0]
        scores = [(self.raw(p, q) - m) / s for p, (m, s) in zip(self.parts, self.base)]
        return sorted(zip(self.names, scores), key=lambda t: -t[1])


def main():
    import json
    sys.stdout.reconfigure(encoding="utf-8")
    result = json.loads(open(sys.argv[1], encoding="utf-8").read())
    t = time.perf_counter()
    idx = PartIndex.from_result(result)
    print(f"建索引 {len(idx.names)} 個零件：{time.perf_counter() - t:.1f} 秒（含載入模型）")
    for path in sys.argv[2:]:
        img = Image.open(path).convert("RGB")
        img.thumbnail((640, 640))
        t = time.perf_counter()
        top = idx.match(img)
        dt = (time.perf_counter() - t) * 1000
        print(f"{path}：{dt:.0f} ms　" + "　".join(f"{n} {s:.3f}" for n, s in top[:3]))


if __name__ == "__main__":
    main()
