"""從 world-overlay 的分析結果（out/app/*_parts.json）抽出每個零件的去背 PNG。"""
import base64, json, sys
from pathlib import Path

src = Path(sys.argv[1])
out = Path(sys.argv[2]); out.mkdir(parents=True, exist_ok=True)
d = json.loads(src.read_text(encoding="utf-8"))
for i, (k, v) in enumerate(d["images"].items()):
    if not k.startswith("part:"):
        continue
    name = k[5:]
    (out / f"{i}_{name}.png").write_bytes(base64.b64decode(v.split(",", 1)[1]))
    print(i, name)
