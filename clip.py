"""零件短片：拿著零件轉一圈錄 3–8 秒 → 轉成 MP4、抽出每一格、上傳到 Files API。

同一個 file_id 給兩個模型用：
  Muse Spark 直接看影片判斷零件的每一面（接口、螺紋常在背面）；
  SAM 3.1 在一格上給框，整段影片一次追蹤，拿到每一格的輪廓（parts_guide.clip_masks）。
一段影片錄全部零件的流程（video_split.py）也用這裡轉檔、切段。
"""
import io
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


class Frames:
    """影片的每一格，存成 JPEG、用到才解開：100 秒的影片有 1000 格，全部解開要好幾 GB。"""

    def __init__(self, jpegs: list[bytes]):
        self.jpegs = jpegs

    def __len__(self):
        return len(self.jpegs)

    def __getitem__(self, i):
        if isinstance(i, slice):
            return Frames(self.jpegs[i])
        return Image.open(io.BytesIO(self.jpegs[i])).convert("RGB")


@dataclass
class Clip:
    frames: Frames
    key: int              # 最清楚的那一格，零件照片、去背圖都用它
    file_id: str | None   # None：只有一格（從整段影片挑出來的），當照片處理、不做影片追蹤
    source: tuple | None = None  # 從一段錄全部零件的影片切出來的：(整段的 file_id, 開始秒, 結束秒)

    @property
    def photo(self) -> Image.Image:
        return self.frames[self.key]


def sharpness(img: Image.Image) -> float:
    small = img.convert("L")
    small.thumbnail((320, 320))
    return ImageStat.Stat(small.filter(ImageFilter.FIND_EDGES)).var[0]


def sharpest(frames: Frames, lo: int, hi: int) -> int:
    return max(range(lo, max(lo + 1, hi)), key=lambda i: sharpness(frames[i]))


def transcode(raw: bytes, max_seconds: float = MAX_SECONDS, start: float = 0) -> tuple[bytes, Frames]:
    """手機錄的影片（MP4/MOV/WebM 都行）→ SAM 收的 H.264 MP4（固定 FPS、沒有聲音）和每一格。
    固定格率，SAM 回傳的格數才會和抽出來的格數一致。"""
    ff = imageio_ffmpeg.get_ffmpeg_exe()
    scale = f"scale='if(gt(iw,ih),min({MAX_SIDE},iw),-2)':'if(gt(iw,ih),-2,min({MAX_SIDE},ih))'"
    with tempfile.TemporaryDirectory() as d:
        src, mp4 = Path(d) / "in", Path(d) / "clip.mp4"
        src.write_bytes(raw)
        # 手機影片的旋轉資訊 ffmpeg 會自動套用
        subprocess.run([ff, "-y", "-loglevel", "error", "-ss", f"{start:.2f}", "-i", str(src), "-t", f"{max_seconds:.2f}",
                        "-an", "-vf", f"fps={FPS},{scale}", "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
                        "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(mp4)], check=True)
        subprocess.run([ff, "-y", "-loglevel", "error", "-i", str(mp4), "-q:v", "3", str(Path(d) / "f%05d.jpg")],
                       check=True)
        frames = Frames([p.read_bytes() for p in sorted(Path(d).glob("f*.jpg"))])
        data = mp4.read_bytes()
    if not len(frames):
        raise RuntimeError("影片裡沒有畫面")
    return data, frames


def upload(data: bytes, api_key: str, name: str = "part.mp4") -> str:
    r = requests.post(f"{API_BASE}/files", headers={"Authorization": f"Bearer {api_key}"},
                      data={"purpose": "user_data"}, files={"file": (name, data, "video/mp4")}, timeout=300)
    if not r.ok:
        raise RuntimeError(f"影片上傳失敗 HTTP {r.status_code}：{r.text[:300]}")
    return r.json()["id"]


def load_clip(raw: bytes, api_key: str) -> Clip:
    data, frames = transcode(raw)
    # 開頭結尾常在晃，從中間 80% 挑最清楚的一格
    n = len(frames)
    key = sharpest(frames, n // 10, n - n // 10)
    return Clip(frames=frames, key=key, file_id=upload(data, api_key))


def load_clip_file(path: str, api_key: str) -> Clip:
    return load_clip(Path(path).read_bytes(), api_key)
