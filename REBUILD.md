# REBUILD.md — how to rebuild WORLD OVERLAY from this repo

This file is written for an AI coding agent (or a developer) who has only this repository and wants to
rebuild the project or something close to it. The working machines were wiped after the hackathon on
2026-11-10; **everything that still exists is in this repo and its GitHub Release `demo-video`**.
README.md has the user-facing description and measured numbers; this file is the map.

## 1. What it is

A phone web app that turns *a photo/video of loose parts + the (optional) instruction manual* into an
animated, step-by-step assembly guide. Built in 10 days for the Meta Global AI Developer Hackathon 2026
(Indie Developer track, "Push the Frontier"). It did not place (5,367 projects; judges never used the
judge passcode, so judging was most likely based on the demo video and README alone).

User flow (static/index.html):
1. Landing page → "Start" → upload the manual (photos or PDF pages) and the parts (one photo per part,
   one short spin video per part, or **one video filming every part in turn**, max 150 s).
2. The backend plans the steps, cuts out each part, and generates a picture of each finished step.
3. The phone plays an animation per step: part A's cutout flies to its connection point on B, then the
   AI-generated "assembled" image fades in. Next/Back buttons, a finish screen.
4. Extras: "Find a part" (point the camera at a part → which one it is and which step uses it),
   "3D" (spin a Hunyuan3D-2 mesh of the part), an English/Chinese UI switch.

## 2. Components

| File | Role |
|---|---|
| `server.py` | stdlib `ThreadingHTTPServer`; serves `static/`, runs analysis jobs in threads, passcodes, daily quota. The docstring at the top lists every endpoint. |
| `parts_guide.py` | Main pipeline (parts → steps → cutouts → step images). Read its docstring first. |
| `video_split.py` | One video of all parts → Muse Spark marks each part's time range and clearest frame → frames become part photos. |
| `clip.py` | Per-part spin clip → 10 fps H.264 → Files API upload; same `file_id` goes to Spark and to SAM video tracking. |
| `sam_fast.py` | Fast decoding of SAM masks (the official parser re-encodes every mask, ~1 s each). |
| `identify.py` | "Find a part": DINOv2 ViT-B/14 on CPU, patch-level matching against each cutout, z-scored; Spark breaks ties. |
| `tracker.py`, `static/motion.js` | Live-camera tracking (hidden behind `#live`, not accurate enough). |
| `model3d.py` | Separate CUDA service on :8766, cutout PNG → Hunyuan3D-2 `.glb`. Install steps are in its docstring. Optional; without it the 3D button just doesn't show. |
| `autodeploy.py` | Polls `origin/main` every 30 s, pulls, restarts `server.py` when `.py` files change. Local commits do **not** trigger a restart. |
| `build_guide.py`, `static/photo.html` | Older "all parts in one photo" flow. |
| `test_*.py` | The experiments that proved each model step works (grounding, SAM outline, planning without a manual). |
| `static/` | Frontend: `index.html` (whole app), `assemble.js` (step animation), `models3d.js` (3D viewer), `guide.js/.css`, `i18n.js` (all UI strings, en + zh). |
| `tools/3d-test/` | The first Hunyuan3D-2 benchmark scripts (mini vs full model, timing, VRAM, a glb viewer). |
| `video/` | Everything used to make the 3-minute demo video (section 6). |

Sample data in git (so "See an example" works with no API key): `samples/` (mic kit photos + manual),
`out/parts_sample.json` (a real saved analysis result), `sample_en.json` (its English translation),
`out/parts_sample_3d/`, `out/parts_sample_en_3d/` (its 3D models), `out/mic_parts_guide.json` (old flow).

## 3. Models and API

- Meta Model API, base `https://api.meta.ai/v1`, key from dev.meta.ai → `.env` `MODEL_API_KEY=...`
  - `muse-spark-1.3`: multimodal reasoning (reads manual images, watches videos, plans steps, finds boxes/points, reviews cutout candidates, confirms identification).
  - `sam-3.1`: segmentation from box / box+points / text prompts; also video tracking from one box.
  - `muse-image-1.0`: image generation/editing for each step's "assembled" picture (multipart `image[i]` uploads).
  - Files API (`/files`, `purpose=user_data`) for videos.
  - Exact request/response shapes: `test_grounding.py` (simplest), then `parts_guide.py`, `clip.py`.
- Local: `facebook/dinov2-base` (transformers, CPU torch), `tencent/Hunyuan3D-2` (+ `-2mini`) via the
  Hunyuan3D-2 GitHub repo in a separate Python 3.10 CUDA venv (≈6 GB VRAM, ~60 s per part).
- Cost: about US$0.3–0.5 per analysis; 6 parts ≈ 2 min end to end, first step visible after ≈45 s.

## 4. Run it

```bash
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements.txt     # CPU torch
echo MODEL_API_KEY=... > .env
.venv/Scripts/python server.py --https                      # self-signed cert in certs/, camera needs HTTPS
# optional public URL:
cloudflared tunnel --no-autoupdate --url https://localhost:8000 --no-tls-verify
# passcodes for other people (stored in users.json, never committed):
.venv/Scripts/python server.py --add-user judge
```

Env vars: `MODEL_API_KEY`, `APP_PASSCODE` (owner, unlimited), `APP_DAILY_RUNS` (default 10).
Usage counts per user per day go to `out/usage.json`. Sample jobs (`sample`, `sample_en`) need no passcode,
including their 3D models and "Find a part" (guest confirms capped at 200/day).

Not in the repo (recreate): `.env`, `users.json`, `certs/`, `.venv`, the Hunyuan3D-2 clone and its venv,
downloaded model weights, `out/app/*` (users' uploads).

## 5. Lessons learned (worth keeping)

- **Film parts lying on the table, one at a time, ~2 s each.** Turning parts in the hand made SAM cut out the hand or floor.
- Box-only SAM prompts fail on low-contrast parts (white cable on beige table). Sending three prompt
  kinds in parallel (box+point, shrunken box + negative corner points, text) and letting Spark pick the
  cleanest mask on a white background fixed most cases.
- Send SAM only a crop around the box scaled to 800 px — most of each call is upload time.
- Muse Image consistency: when attaching to an earlier assembly, *edit the previous step's generated image* instead of generating fresh.
- Stream results: show the first step as soon as cutouts exist, fill in step images as they finish.
- Live camera tracking with cloud models is too laggy; the scan-then-animate flow is what worked.
- **Hackathon judging:** with ~5,000 entries, judges watch the video and skim the README. The live app
  was never opened with the judge passcode. Put the effort into a sharp, human-narrated video and a
  README whose first screen shows the result (GIF), and make the demo usable with zero friction.

## 6. The demo video (`video/`)

Built without a video editor: an HTML compositor rendered frame by frame in headless Chrome.

- `outline_zh.txt` — the brief (6 sections: audience → pain → why text-only AI falls short → solution → real product demo → vision + slogan).
- `vo/script.json` + `vo/*.wav` — narration lines (Windows SAPI "Microsoft Zira" via `vo/tts.ps1`). `narration-script.txt` has timestamps.
- `timeline.json` — 11 scenes, 167.5 s, scene and narration start times.
- `video.html` — the compositor (`init(timeline)`, `setT(seconds)` draws that exact moment; subtitles, dip transitions).
- `cdp.py` — tiny stdlib Chrome DevTools client + screencast recorder. `record_app.py` records the real app. `render.py` renders frames. `audio.py` mixes narration (and optionally a numpy-synthesized music bed).
- `assets/` — real photos, the SAM cutouts and Muse Image results of real job `2f01fa90` (13-part USB microphone kit, Spanish manual, 11 steps).
- `demoen_parts.json` — that job's result with step text translated to English (the run was made in the Chinese UI).
- `build.sh` — the full rebuild sequence, including the ffmpeg commands.
- GitHub Release `demo-video`: `brag.mp4` (submitted version, narration only), `brag_with_music.mp4`,
  `footage.mp4` (raw phone footage), `rec_*.mp4` (app screen recordings), `demoen_3d.zip`.
