"""用 Hunyuan3D-2 把 parts/ 裡每張去背 PNG 變成 3D 形狀（.glb，不含材質）。

用法：
  .venv/Scripts/python gen.py mini   # Hunyuan3D-2mini-Turbo（0.6B，5 步，快）
  .venv/Scripts/python gen.py full   # Hunyuan3D-2 v2-0（1.1B，50 步，細）
結果放在 out/<模型>/，每個零件的秒數和顯示卡記憶體峰值寫在 out/<模型>/stats.json。
"""
import json
import sys
import time
from pathlib import Path

import torch
from PIL import Image

from hy3dgen.shapegen import DegenerateFaceRemover, FaceReducer, FloaterRemover, Hunyuan3DDiTFlowMatchingPipeline

ROOT = Path(__file__).resolve().parent
MODELS = {
    "mini": dict(repo="tencent/Hunyuan3D-2mini", subfolder="hunyuan3d-dit-v2-mini-turbo", steps=5, flashvdm=True),
    "full": dict(repo="tencent/Hunyuan3D-2", subfolder="hunyuan3d-dit-v2-0", steps=50, flashvdm=False),
}


def main():
    which = sys.argv[1] if len(sys.argv) > 1 else "mini"
    cfg = MODELS[which]
    out = ROOT / "out" / which
    out.mkdir(parents=True, exist_ok=True)

    t = time.time()
    pipe = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(cfg["repo"], subfolder=cfg["subfolder"], use_safetensors=False, device="cuda")
    if cfg["flashvdm"]:
        pipe.enable_flashvdm(topk_mode="merge")
    print(f"模型載入 {time.time() - t:.1f} 秒", flush=True)

    stats = {}
    for png in sorted((ROOT / "parts").glob("*.png")):
        image = Image.open(png).convert("RGBA")
        torch.cuda.reset_peak_memory_stats()
        t = time.time()
        for seed in (12345, 1, 2):  # 偶爾生成出空的體積（沒有表面），換種子再試
            mesh = pipe(image=image, num_inference_steps=cfg["steps"], octree_resolution=380, num_chunks=20000,
                        generator=torch.manual_seed(seed), output_type="trimesh")[0]
            if mesh is not None:
                break
            print(f"{png.stem}：種子 {seed} 沒有表面，重試", flush=True)
        gen = time.time() - t
        if mesh is None:
            stats[png.stem] = {"seconds": round(gen, 1), "failed": True}
            print(png.stem, "失敗", flush=True)
            continue
        mesh = FloaterRemover()(mesh)
        mesh = DegenerateFaceRemover()(mesh)
        mesh = FaceReducer()(mesh, max_facenum=40000)
        mesh.export(out / f"{png.stem}.glb")
        peak = torch.cuda.max_memory_allocated() / 2**30
        stats[png.stem] = {"seconds": round(gen, 1), "peak_vram_gb": round(peak, 2), "faces": len(mesh.faces)}
        print(png.stem, stats[png.stem], flush=True)
    (out / "stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
