"""SAM 遮罩解碼加速。

meta_sam_parser 收到每個遮罩都會用純 Python 完整解一次來驗證，解完再編碼回去比對（一個約 1 秒）。
SAM 本身一張圖約 30 ms、影片 60 格約 7 秒就回來，時間幾乎都花在這裡：影片 60 格 × 2 個提示要 1–2 分鐘。
這裡讓 parser 不再逐一驗證，改成用到哪個遮罩才解、不做編碼比對、解過的記住（同一個遮罩會被
fill_ratio、contrast、去背各用一次）。解不開時一樣會丟錯誤。
"""
from functools import lru_cache

import meta_sam_parser._segmentation as _segmentation
from meta_sam_parser import decode_mask_to_raster
from meta_sam_parser._mask_codec import _decode_raster

_segmentation.decode_mask_to_raster = lambda mask: None  # parser 收到時不驗證

_TO_255 = bytes([0] + [255] * 255)


@lru_cache(maxsize=512)
def raster(mask) -> bytes:
    """遮罩 → 每個像素 0/1。"""
    if mask.encoding == "one_bit":
        return _decode_raster(mask.payload, mask.width, mask.height)
    return decode_mask_to_raster(mask)


def raster_255(mask) -> bytes:
    """遮罩 → 每個像素 0/255，可以直接給 Image.frombytes("L", ...)。"""
    return raster(mask).translate(_TO_255)


def ones(mask) -> int:
    return raster(mask).count(1)
