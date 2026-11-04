"""完成品的參考照片 → 推理組裝步驟時對照（parts_guide.plan）。

使用者有拍完成品就用使用者拍的。不管有沒有，Muse Spark 都會看零件照片認出是什麼產品，用內建的上網搜尋
（Responses API 的 web_search）讀同款產品的說明書、商品頁，寫下它是怎麼組起來的（文字）。
AI 回傳的圖片直連網址也試著下載（很多網站不給程式抓，抓不到就算了，不繞過），再請 Spark 挑出真的是
同一款、組裝好的。搜尋失敗就照舊推理。

用法（測試）：
  .venv/Scripts/python reference.py out/app/<job>_parts/[0-9]*.jpg --manual out/app/<job>_parts/manual1.jpg
"""
import argparse
import io
import ipaddress
import json
import re
import socket
import time
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import urljoin, urlparse

import requests
from PIL import Image

from parts_guide import image_part, spark_json
from test_grounding import API_BASE, SPARK_MODEL, load_api_key, load_image, to_data_url

SEARCH_CALLS = 8      # 最多搜尋／開網頁幾次：不限制的話會搜 20 幾次、90 秒
SEARCH_SIDE = 512     # 給搜尋看的零件照片最長邊：認產品夠用，請求小一點
MAX_CANDIDATES = 8    # 最多試下載幾張候選圖
MAX_REFS = 2          # 最多給推理幾張參考照
MAX_USER_REFS = 3     # 使用者拍的完成品照片最多用幾張
REF_SIDE = 1024
MIN_SIDE = 300        # 太小的多半是縮圖或圖示
MAX_BYTES = 8 << 20
FETCH_TIMEOUT = 10

SEARCH_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["product", "assembly", "image_urls"],
    "properties": {
        "product": {"type": "string", "description": "產品名稱（品牌、型號、套組名）；認不出來時空字串"},
        "assembly": {"type": "string", "description": "照找到的說明書／商品頁，這個產品組好時哪個零件接在哪個零件上、"
                                                      "怎麼接；沒找到就空字串"},
        "image_urls": {"type": "array", "items": {"type": "string"}},
    },
}

SEARCH_PROMPT = (
    "These photos show the parts of one product before assembly{manual_note}. Identify the product as exactly as you "
    "can (brand, model, kit name; it may be a white-label version of a common product). Then use web search to find "
    "its instruction manual or product pages (or those of the same kind of kit) and read how it is assembled. "
    "Return the product name; in assembly, how the finished product fits together according to those sources: "
    "which part attaches to which and how (screwed on, clamped, slid in, plugged in), and which parts are "
    "alternatives or accessories not attached to the main build; and any direct image URLs (.jpg/.png/.webp) of "
    "the fully assembled product you saw. If you cannot tell what the product is, return empty strings and an "
    "empty list."
)

PICK_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["same"],
    "properties": {"same": {"type": "array", "items": {"type": "integer"}, "description": "候選編號，最好的在前"}},
}

PICK_PROMPT = (
    "The first {n} images are parts of a product before assembly ({product}). The {m} candidate photos after them "
    "were found online. List the candidate numbers (1-{m}) that clearly show this kind of product fully assembled, "
    "best first. Leave out photos of single parts, the box, a different product, or scenes where the assembled "
    "product is too small to see how the parts connect."
)


def _public(url: str) -> bool:
    """只抓公開網路上的 http(s)：網址是 AI 給的，不能讓它連到本機或內網。"""
    u = urlparse(url)
    if u.scheme not in ("http", "https") or not u.hostname:
        return False
    try:
        infos = socket.getaddrinfo(u.hostname, u.port or (443 if u.scheme == "https" else 80))
    except OSError:
        return False
    return all(ipaddress.ip_address(i[4][0]).is_global for i in infos)


def _get(url: str, kind: str, limit: int) -> bytes | None:
    """自己跟轉址（每一跳都檢查是不是公開網址）；Content-Type 開頭要是 kind，超過 limit 就放棄。"""
    for _ in range(4):
        if not _public(url):
            return None
        try:
            r = requests.get(url, timeout=FETCH_TIMEOUT, stream=True, allow_redirects=False,
                             headers={"User-Agent": "WorldOverlay/1.0 (assembly guide app)", "Accept": f"{kind}*"})
        except requests.RequestException:
            return None
        with r:
            if r.is_redirect and "location" in r.headers:
                url = urljoin(url, r.headers["location"])
                continue
            if not r.ok or not r.headers.get("content-type", "").startswith(kind):
                return None
            data = b""
            for chunk in r.iter_content(64 << 10):
                data += chunk
                if len(data) > limit:
                    return None
            return data
    return None


def _image(url: str) -> Image.Image | None:
    data = _get(url, "image/", MAX_BYTES)
    try:
        img = Image.open(io.BytesIO(data)).convert("RGB") if data else None
    except Exception:
        return None
    if not img or min(img.size) < MIN_SIDE:
        return None
    img.thumbnail((REF_SIDE, REF_SIDE))
    return img


def _search(key: str, photos: list, manual: list) -> dict:
    note = " and a page of its instruction manual" if manual else ""
    content = [{"type": "input_text", "text": SEARCH_PROMPT.format(manual_note=note)}]
    for img in photos + manual[:1]:
        small = img.copy()
        small.thumbnail((SEARCH_SIDE, SEARCH_SIDE))
        content.append({"type": "input_image", "image_url": to_data_url(small)})
    body = {"model": SPARK_MODEL, "input": [{"type": "message", "role": "user", "content": content}],
            "tools": [{"type": "web_search"}], "max_tool_calls": SEARCH_CALLS,
            "text": {"format": {"type": "json_schema", "name": "Product", "strict": True, "schema": SEARCH_SCHEMA}}}
    for attempt in range(2):
        try:
            r = requests.post(f"{API_BASE}/responses", json=body, timeout=240,
                              headers={"Authorization": f"Bearer {key}"})
        except requests.RequestException:
            r = None
        if r is not None and r.ok:
            break
        time.sleep(3)
    else:
        raise RuntimeError(f"Muse Spark 搜尋失敗 HTTP {getattr(r, 'status_code', '連線中斷')}")
    # 回覆裡可能先有一句「我來搜尋…」，最後一段才是 JSON
    texts = [c.get("text", "") for o in r.json().get("output", []) if o.get("type") == "message"
             for c in o.get("content", [])]
    for text in reversed(texts):
        try:
            return json.loads(text)
        except ValueError:
            continue
    raise RuntimeError("Muse Spark 搜尋沒有回傳結果")


def find(key: str, photos: list, manual: list | None = None, finished: list | None = None, log=print,
         lang: str = "zh") -> dict:
    """回傳 {"product": 產品名稱, "assembly": 網路上查到的組裝方式, "images": [參考照], "source": "user" | "web" | None}。
    photos 是零件照片（PIL 圖），finished 是使用者拍的完成品照片（可省略；有的話參考照就用它）。"""
    en = lang == "en"
    log("AI is identifying the product and looking up how it is assembled online" if en
        else "AI 認出是什麼產品，上網查它怎麼組裝")
    t0 = time.perf_counter()
    try:
        found = _search(key, photos, manual or [])
    except RuntimeError as e:
        log(f"Web search skipped ({e})" if en else f"略過上網搜尋（{e}）")
        found = {"product": "", "assembly": "", "image_urls": []}
    product, assembly = found["product"].strip(" ."), found["assembly"].strip()
    refs = finished[:MAX_USER_REFS] if finished else []
    if not refs:
        with ThreadPoolExecutor(8) as ex:
            cands = [im for im in ex.map(_image, list(dict.fromkeys(found["image_urls"]))[:MAX_CANDIDATES]) if im]
        if cands:
            content = [{"type": "text", "text": PICK_PROMPT.format(n=len(photos), m=len(cands), product=product or "unknown")}]
            content += [image_part(p) for p in photos]
            for i, c in enumerate(cands, 1):
                content += [{"type": "text", "text": f"Candidate {i}:"}, image_part(c)]
            try:
                same = spark_json(key, content, PICK_SCHEMA, "SameProduct", "low")["same"]
            except RuntimeError:
                same = []
            refs = [cands[i - 1] for i in dict.fromkeys(same) if 1 <= i <= len(cands)][:MAX_REFS]
    dt = time.perf_counter() - t0
    if en:
        log(f"Product: {product or 'not recognized'}; assembly notes {'found' if assembly else 'not found'}; "
            f"{len(refs)} reference photo(s) ({dt:.0f} s)")
    else:
        log(f"產品：{product or '認不出來'}；{'查到' if assembly else '沒查到'}組裝方式；{len(refs)} 張參考照（{dt:.0f} 秒）")
    return {"product": product, "assembly": assembly, "images": refs,
            "source": "user" if finished else "web" if refs else None}


def main():
    import sys
    from pathlib import Path
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("photos", nargs="+")
    ap.add_argument("--manual", nargs="+", default=[])
    ap.add_argument("--out", help="把參考照存到這個資料夾")
    args = ap.parse_args()
    photos = [load_image(p, 1280) for p in args.photos]
    manual = [load_image(p, 1600) for p in args.manual]
    ref = find(load_api_key(), photos, manual, log=lambda m: print(f"  {m}"))
    print(f"  組裝方式：{ref['assembly']}")
    if args.out:
        Path(args.out).mkdir(parents=True, exist_ok=True)
        for i, im in enumerate(ref["images"], 1):
            im.save(Path(args.out) / f"ref{i}.jpg")


if __name__ == "__main__":
    main()
