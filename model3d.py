"""零件 3D 模型服務：去背 PNG → Hunyuan3D-2 生成的形狀（.glb，沒有材質）。在有 NVIDIA 顯示卡的電腦上跑。

server.py 是 CPU 版 PyTorch 的環境，這個要 CUDA 版，所以分開跑，server.py 用 http://127.0.0.1:8766 呼叫。
沒開這個服務時 App 照常運作，只是沒有「3D」按鈕。

安裝（另一個環境，Python 3.10；套件版本要固定，新版 transformers 會讀不了模型）：
  git clone --depth 1 https://github.com/Tencent-Hunyuan/Hunyuan3D-2.git
  py -3.10 -m venv .venv3d
  .venv3d/Scripts/python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
  .venv3d/Scripts/python -m pip install "transformers<5" "diffusers<0.36" einops opencv-python numpy omegaconf tqdm \
      trimesh pymeshlab pygltflib xatlas accelerate scikit-image pillow
  .venv3d/Scripts/python -m pip install --no-deps -e Hunyuan3D-2

用法：
  .venv3d/Scripts/python model3d.py     # 開服務（第一次會下載模型約 15 分鐘）

POST /shape  body 是去背 PNG → 回傳 glb（約 60 秒，顯示卡記憶體約 6 GB）；一次只做一個，其他的排隊。
Hunyuan3D-2 是 Tencent Hunyuan 3D 2.0 Community License（不適用於歐盟、英國、南韓）。
"""
import argparse
import io
import sys
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, HTTPServer

import torch
from PIL import Image

from hy3dgen.shapegen import DegenerateFaceRemover, FaceReducer, FloaterRemover, Hunyuan3DDiTFlowMatchingPipeline

PORT = 8766
MAX_FACES = 40000  # 手機上轉得動；檔案約 1–2 MB
SEEDS = (12345, 1, 2)  # 偶爾生成出空的體積（沒有表面，例如全黑的防噴綿），換種子再試

pipe = None


def load():
    global pipe
    t = time.time()
    pipe = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
        "tencent/Hunyuan3D-2", subfolder="hunyuan3d-dit-v2-0", use_safetensors=False, device="cuda")
    print(f"模型載入 {time.time() - t:.0f} 秒", flush=True)


def shape(png: bytes) -> bytes | None:
    image = Image.open(io.BytesIO(png)).convert("RGBA")
    for seed in SEEDS:
        mesh = pipe(image=image, num_inference_steps=50, octree_resolution=380, num_chunks=20000,
                    generator=torch.manual_seed(seed), output_type="trimesh")[0]
        if mesh is not None:
            break
    else:
        return None
    mesh = FloaterRemover()(mesh)
    mesh = DegenerateFaceRemover()(mesh)
    mesh = FaceReducer()(mesh, max_facenum=MAX_FACES)
    return mesh.export(file_type="glb")


class Handler(BaseHTTPRequestHandler):
    def do_POST(self):
        if self.path != "/shape":
            return self.send_error(HTTPStatus.NOT_FOUND)
        png = self.rfile.read(int(self.headers.get("Content-Length") or 0))
        t = time.time()
        try:
            glb = shape(png)
        except Exception as e:  # 圖片壞掉之類的：回報失敗，服務繼續開著
            print(f"失敗：{e}", flush=True)
            glb = None
        if glb is None:
            return self.send_error(HTTPStatus.UNPROCESSABLE_ENTITY, "no surface")
        print(f"完成 {time.time() - t:.0f} 秒，{len(glb) // 1024} KB", flush=True)
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "model/gltf-binary")
        self.send_header("Content-Length", str(len(glb)))
        self.end_headers()
        self.wfile.write(glb)

    def log_message(self, fmt, *args):
        pass


def main():
    argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter).parse_args()
    load()
    print(f"3D 模型服務：http://127.0.0.1:{PORT}（只接受這台電腦連線）", flush=True)
    HTTPServer(("127.0.0.1", PORT), Handler).serve_forever()  # 不開多執行緒：顯示卡一次做一個，其他請求自然排隊


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
