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
- [x] 即時追蹤：手機端畫面比對補償鏡頭平移（`static/motion.js`，每秒約 12 次）＋伺服器每 2–3 秒用 SAM 重描（`tracker.py`），跟丟時用 Muse Spark 依名稱重新找
- [x] 鏡頭掃描：用即時鏡頭的畫面分析，分析完直接在同一個鏡頭上引導，從掃描那一格的輪廓開始追蹤（不用重新找零件）；換步驟時新零件立刻從掃描畫面的位置顯示；SAM 描到背景（面積暴增）時沿用上一次的輪廓
- [ ] 驗證與修正流程
- [ ] 語音提問

## 主要流程：一個零件一張照片 → 組裝動畫（`parts_guide.py`）

1. **Muse Spark** 看所有零件照片（和說明書，可省略），排出組裝步驟；每一步把零件 A 裝到零件 B 或前面完成的組合件上
2. **Muse Spark** 找零件框和接口點 → **SAM 3.1** 剪成去背圖（框、框＋點兩種提示一起送，留下和背景差最多的那個；只送框附近、縮到 800 px 的裁切圖，SAM 一次呼叫的時間大多是上傳）
3. **Muse Image** 每一步把 A 裝到 B 上，生成裝好的照片；裝到組合件時照著前一步的生成圖修改，前後一致
4. 手機動畫（`static/assemble.js`）：A 的去背圖飛到 B 的接口 → 淡入生成的組裝結果 → 放大

**零件也可以錄影**：拿在手上轉一圈錄 3–8 秒（`clip.py`）。影片轉成 10 fps 的 H.264 MP4 上傳到 Files API 一次，同一個 file_id 給 **Muse Spark** 直接看影片（看得到背面的接口），也給 **SAM 3.1** 做影片追蹤：在最清楚的那一格給框，一次請求就拿到每一格的輪廓（60 格約 7 秒，每 1000 格 US$0.2）。動畫裡這個零件會先轉一圈，**Muse Image** 也會多拿到另一個角度的去背圖當參考。

**一段影片錄全部零件**（`video_split.py`）：零件放在桌上，鏡頭一個一個拍過去，每個停 2 秒（最長 150 秒）。
不要拿起來轉：實測手拿著轉時，SAM 常把手或地板剪進去。**Muse Spark** 看整段影片，標出每個零件的時間段和
最清楚、沒被手擋住的時間點，從整段影片已抽好的格子挑那一刻最清楚的一格當零件照片（不再切段、轉檔、上傳）。
推理步驟時送整段影片一次（Spark 一次請求最多 5 段影片）加上每個零件那一格。
Spark 看影片約每秒 260 token，100 秒的影片每次分析比 16 段各 5 秒的短片只多約 US$0.04。

實測 14 個零件、115 秒手機影片（透過通道、手機網路）：上傳約 4 分鐘、切段約 2.5 分鐘、推理 83 秒、去背 82 秒，
第一步要等約 9 分半。改成只挑格子後，6 個零件的測試影片切段 + 推理 + 去背約 2 分鐘。
Muse Image 每步約 20–25 秒、裝到組合件上的步驟要等前一步，邊看邊補上。

SAM 遮罩解碼（`sam_fast.py`）：meta_sam_parser 會把每個遮罩用純 Python 解一次驗證、再編碼回去比對，一個約 1 秒；
影片 60 格就要 1 分鐘以上。改成用到才解、不做比對、解過的記住，同一段影片從 60 多秒降到約 1 秒。

約 45 秒開始播第一步，之後每一步的結果圖陸續補上（6 個零件全部約 2 分 20 秒）。
命令列：`.venv/Scripts/python parts_guide.py samples/mic_each/*.jpg --manual samples/mic_manual.jpg`（零件可以混 .mp4/.mov）
舊版「一張照片標出零件」流程在 `/static/photo.html`。

## 手機試用

```bash
.venv/Scripts/python server.py
```

手機連同一個 Wi-Fi，打開終端機印出的網址（例如 `http://192.168.1.172:8000`）。
「看範例」不呼叫 API，可以先看引導畫面。

要用即時鏡頭（App 內開相機需要 HTTPS），改用：

```bash
.venv/Scripts/python server.py --https
```

會自動產生區網用的自簽憑證（`certs/`，不會進 git）。手機第一次打開會出現安全性警告，點「進階」→「繼續前往」。
首頁拍「📖 說明書」（可略過）和「🔩 零件」各一張，按開始分析，約 45 秒後逐步顯示發光輪廓和箭頭動畫。
即時鏡頭目前不夠準（手機移動時 AI 回應跟不上），入口預設隱藏；網址後面加 `#live` 才會出現。

命令列也可以帶說明書：`.venv/Scripts/python build_guide.py samples/mic_parts.jpg --manual samples/mic_manual.jpg`

**App 內錄影**：首頁「📹 在 App 裡錄全部零件」直接用瀏覽器錄（MediaRecorder，約 720p、2.5 Mbps），
60 秒約 19 MB，手機相機 App 錄的 1080p 約大 5 倍；需要 HTTPS（`--https` 或通道）。

## 給隊友試用

API key 只在這台電腦，隊友透過網址用，一人一組通行碼、每人每天限次數：

```bash
.venv/Scripts/python server.py --add-user 小明        # 印出小明的通行碼（存在 users.json，不進 git）
.venv/Scripts/python server.py --remove-user 小明     # 收回，伺服器不用重開
APP_PASSCODE=<自己的> APP_DAILY_RUNS=10 .venv/Scripts/python server.py --https
cloudflared tunnel --no-autoupdate --url https://localhost:8000 --no-tls-verify   # 印出 https://xxx.trycloudflare.com
```

每次分析約 US$0.3–0.5；同時最多 2 個分析在跑，用量記在 `out/usage.json`。
通道網址每次重開都會變；電腦要開著、不能睡眠。

## 執行測試腳本

```bash
python -m venv .venv
.venv/Scripts/python -m pip install requests pillow meta-sam-parser imageio-ffmpeg   # macOS/Linux: .venv/bin/python
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
