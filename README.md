# WORLD OVERLAY

讓新手敢操作、快速學會的 AI 現場引導。手機鏡頭對著零件，AI 一次帶一個步驟：指出現在要拿哪個零件、裝在哪裡，做完再看畫面檢查對不對。

Meta Global AI Developer Hackathon 2026 參賽作品（開發期 10/26–11/4）。Demo 對象：IKEA 家具組裝。

核心流程：看懂現場 → 指出物件 → 引導操作 → 驗證結果 → 調整下一步

## 架構（規劃中）

```
手機網頁（相機畫面 + 疊加框線 + 語音）
   │ 按「下一步／檢查」時拍一張
   ▼
後端（保管 API key）
   ├─ Muse Spark：依目前步驟找出零件位置 → 座標 + 指示
   ├─ SAM 3.1：精準描出目標零件輪廓
   └─ Muse Spark：比對完成畫面 → 通過／錯誤原因／修正建議
   ▲
steps.json（由說明書轉出、人工確認過的 SOP）
```

## 目前進度

- [x] `test_grounding.py`：驗證 Muse Spark / SAM 3.1 能否在照片中找到指定零件，並量測延遲
- [x] `test_pipeline.py`：Muse Spark 框出零件 → SAM 3.1 描出發光輪廓
- [x] `test_assembly.py`：不給說明書，只看照片推理組裝步驟
- [x] `build_guide.py`：一張照片 → 互動組裝引導網頁（兩個零件發光 + 動畫箭頭，逐步切換）
- [ ] 說明書 → 步驟
- [x] `server.py` + `static/`：手機 App（拍照 → 分析進度 → 逐步引導），API key 只放在電腦
- [ ] 即時追蹤（鏡頭移動時發光跟著零件走）
- [ ] 驗證與修正流程
- [ ] 語音提問

## 手機試用

```bash
.venv/Scripts/python server.py
```

手機連同一個 Wi-Fi，打開終端機印出的網址（例如 `http://192.168.1.172:8000`）。
「看範例」不呼叫 API，可以先看引導畫面。區網 http 下會用手機內建相機拍照；要在 App 內開即時相機需要 HTTPS。

## 執行測試腳本

```bash
python -m venv .venv
.venv/Scripts/python -m pip install requests pillow meta-sam-parser   # macOS/Linux: .venv/bin/python
cp .env.example .env                                  # 填入 MODEL_API_KEY
.venv/Scripts/python test_grounding.py samples/parts.jpg \
  --target "最長的木板" --target "木釘" --sam "wooden dowel" --runs 3
```

結果圖與 JSON 會輸出到 `out/`。

產生組裝引導網頁（約 60 秒；`--reuse` 只重畫網頁不呼叫 API）：

```bash
.venv/Scripts/python build_guide.py samples/parts.jpg
```

API key 在 [dev.meta.ai](https://dev.meta.ai) 的 **API keys** 頁建立，請勿提交 `.env`。
