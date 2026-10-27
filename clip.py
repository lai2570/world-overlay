"""零件短片：拿著零件轉一圈錄 3–8 秒 → 轉成 MP4、抽出每一格、上傳到 Files API。

同一個 file_id 給兩個模型用：
  Muse Spark 直接看影片判斷零件的每一面（接口、螺紋常在背面）；
  SAM 3.1 在一格上給框，整段影片一次追蹤，拿到每一格的輪廓（parts_guide.clip_masks）。
"""
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

import imageio_ffmpeg
import requests
from PIL import Image, ImageFilter, ImageStat

from test_grounding import API_BASE

FPS = 10          # 轉一圈 10 格/秒就夠順，SAM 依格數計價
MAX_SECONDS = 8
MAX_SIDE = 960


@dataclass
class Clip:
    frames: list          # 每一格的 PIL 圖（RGB）
    key: int              # 最清楚的那一格，零件照片、去背圖都用它
    file_id: str

    @property
    def photo(self) -> Image.Image:
        return self.frames[self.key]


def sharpness(img: Image.Image) -> float:
    small = img.convert("L")
    small.thumbnail((320, 320))
    return ImageStat.Stat(small.filter(ImageFilter.FIND_EDGES)).var[0]


def load_clip(raw: bytes, api_key: str) -> Clip:
    """手機錄的影片（MP4/MOV/WebM 都行）→ 轉成 SAM 收的 H.264 MP4，抽格，上傳。"""
    ff = imageio_ffmpeg.get_ffmpeg_exe()
    scale = f"scale='if(gt(iw,ih),min({MAX_SIDE},iw),-2)':'if(gt(iw,ih),-2,min({MAX_SIDE},ih))'"
    with tempfile.TemporaryDirectory() as d:
        src, mp4 = Path(d) / "in", Path(d) / "clip.mp4"
        src.write_bytes(raw)
        # 手機影片的旋轉資訊 ffmpeg 會自動套用；拿掉聲音、固定格率，SAM 的格數和抽出的格數才會一致
        subprocess.run([ff, "-y", "-loglevel", "error", "-i", str(src), "-t", str(MAX_SECONDS), "-an",
                        "-vf", f"fps={FPS},{scale}", "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
                        "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(mp4)], check=True)
        subprocess.run([ff, "-y", "-loglevel", "error", "-i", str(mp4), "-q:v", "3", str(Path(d) / "f%04d.jpg")],
                       check=True)
        frames = [Image.open(p).convert("RGB") for p in sorted(Path(d).glob("f*.jpg"))]
        data = mp4.read_bytes()
    if not frames:
        raise RuntimeError("影片裡沒有畫面")
    # 開頭結尾常在晃，從中間 80% 挑最清楚的一格
    lo, hi = len(frames) // 10, max(len(frames) // 10 + 1, len(frames) - len(frames) // 10)
    key = max(range(lo, hi), key=lambda i: sharpness(frames[i]))
    r = requests.post(f"{API_BASE}/files", headers={"Authorization": f"Bearer {api_key}"},
                      data={"purpose": "user_data"}, files={"file": ("part.mp4", data, "video/mp4")}, timeout=120)
    if not r.ok:
        raise RuntimeError(f"影片上傳失敗 HTTP {r.status_code}：{r.text[:300]}")
    return Clip(frames=frames, key=key, file_id=r.json()["id"])


def load_clip_file(path: str, api_key: str) -> Clip:
    return load_clip(Path(path).read_bytes(), api_key)
