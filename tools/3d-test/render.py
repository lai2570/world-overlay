"""把 out/<模型>/*.glb 從三個角度畫成灰階圖，存成 out/compare.png（每個零件一列：去背圖、mini 三視角、full 三視角）。"""
from pathlib import Path

import numpy as np
import trimesh
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent
SIZE = 220
VIEWS = [(0, 0), (90, 0), (35, 30)]  # (繞 y 軸轉幾度, 往下看幾度)


def render(mesh: trimesh.Trimesh, yaw: float, pitch: float) -> Image.Image:
    pts, idx = trimesh.sample.sample_surface(mesh, 300_000)
    nrm = mesh.face_normals[idx]
    a, b = np.radians(yaw), np.radians(pitch)
    ry = np.array([[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]])
    rx = np.array([[1, 0, 0], [0, np.cos(b), -np.sin(b)], [0, np.sin(b), np.cos(b)]])
    rot = rx @ ry
    p = (pts - mesh.bounds.mean(0)) @ rot.T
    n = nrm @ rot.T
    scale = (SIZE - 20) / (np.ptp(p[:, :2], axis=0).max() + 1e-9)
    x = (p[:, 0] * scale + SIZE / 2).astype(int)
    y = (-p[:, 1] * scale + SIZE / 2).astype(int)
    light = np.array([0.4, 0.6, 0.7]) / np.linalg.norm([0.4, 0.6, 0.7])
    shade = (60 + 190 * np.clip(n @ light, 0, 1)).astype(np.uint8)
    img = np.full((SIZE, SIZE), 255, np.uint8)
    depth = np.full((SIZE, SIZE), -np.inf)
    ok = (x >= 0) & (x < SIZE) & (y >= 0) & (y < SIZE)
    order = np.argsort(p[ok, 2])  # 由遠到近畫，近的蓋掉遠的
    xs, ys, zs, ss = x[ok][order], y[ok][order], p[ok, 2][order], shade[ok][order]
    for dx in (0, 1):
        for dy in (0, 1):
            xx, yy = np.clip(xs + dx, 0, SIZE - 1), np.clip(ys + dy, 0, SIZE - 1)
            img[yy, xx] = ss
            depth[yy, xx] = zs
    return Image.fromarray(img).convert("RGB")


def main():
    runs = [r for r in ("mini", "full") if (ROOT / "out" / r).exists()]
    parts = sorted((ROOT / "parts").glob("*.png"))
    cols = 1 + 3 * len(runs)
    sheet = Image.new("RGB", (cols * SIZE, len(parts) * SIZE + 24), "white")
    d = ImageDraw.Draw(sheet)
    d.text((10, 6), "input", fill="black")
    for i, r in enumerate(runs):
        d.text((SIZE * (1 + 3 * i) + 10, 6), r, fill="black")
    for row, png in enumerate(parts):
        top = 24 + row * SIZE
        src = Image.open(png).convert("RGBA")
        src.thumbnail((SIZE - 20, SIZE - 20))
        bg = Image.new("RGB", (SIZE, SIZE), "white")
        bg.paste(src, ((SIZE - src.width) // 2, (SIZE - src.height) // 2), src)
        sheet.paste(bg, (0, top))
        for i, r in enumerate(runs):
            glb = ROOT / "out" / r / f"{png.stem}.glb"
            if not glb.exists():
                continue
            mesh = trimesh.load(glb, force="mesh")
            for j, (yaw, pitch) in enumerate(VIEWS):
                sheet.paste(render(mesh, yaw, pitch), (SIZE * (1 + 3 * i + j), top))
        print(png.stem, flush=True)
    sheet.save(ROOT / "out" / "compare.png")


if __name__ == "__main__":
    main()
