# WORLD OVERLAY

AI on-site guidance that gives beginners the confidence to start and helps them learn fast. Point your phone camera at the parts and the AI walks you through one step at a time: it shows which part to pick up and where it goes, then looks at the camera view to check your work.

Entry for the Meta Global AI Developer Hackathon 2026 (build period Oct 26 – Nov 4). Demo use case: IKEA furniture assembly.

**Live demo: https://fast-bobby-fewer-isle.trycloudflare.com** (open it on a phone; "See an example" works without a passcode)

### For judges

- Running your own analysis needs a passcode; the judge passcode is in our hackathon submission.
- It allows **10 analyses per day**, because each run uses paid API calls (about US$0.3–0.5). Before every run, the app asks you to confirm and shows how many runs are left today.
- Need more runs? Contact me at lai2570@gmail.com and I'll raise the limit.

Core loop: understand the scene → point out the object → guide the action → verify the result → adjust the next step

## Architecture (planned)

```
Phone web page (camera view + overlay outlines + voice)
   │ takes one photo when you tap "Next / Check"
   ▼
Backend (holds the API key)
   ├─ Muse Spark: finds the part for the current step → coordinates + instruction
   ├─ SAM 3.1: traces a precise outline of the target part
   └─ Muse Spark: compares the finished view → pass / what went wrong / how to fix it
   ▲
steps.json (SOP converted from the manual, checked by a person)
```

## Progress

- [x] `test_grounding.py`: checks whether Muse Spark / SAM 3.1 can find a given part in a photo, and measures latency
- [x] `test_pipeline.py`: Muse Spark draws a box around the part → SAM 3.1 traces a glowing outline
- [x] `test_assembly.py`: works out the assembly steps from a photo alone, without the manual
- [x] `build_guide.py`: one photo → interactive assembly guide web page (two parts glow + animated arrow, step by step)
- [ ] Manual → steps
- [x] `server.py` + `static/`: phone app (take a photo → analyze progress → step-by-step guidance); the API key stays on the computer
- [x] Live tracking: the phone matches frames to compensate for camera panning (`static/motion.js`, about 12 times a second) + the server re-traces with SAM every 2–3 seconds (`tracker.py`); when tracking is lost, Muse Spark finds the part again by name
- [x] Camera scan: analysis runs on frames from the live camera, and guidance starts on the same camera right after, tracking from the outline in the scanned frame (no need to find the parts again); when the step changes, the new part shows up at once at its position in the scanned frame; if SAM traces the background (the area suddenly balloons), the previous outline is kept
- [ ] Verify-and-correct loop
- [ ] Voice questions

## Main flow: one photo per part → assembly animation (`parts_guide.py`)

1. **Muse Spark** looks at all the part photos (and the manual, which is optional) and plans the assembly steps; each step attaches part A to part B or to an assembly built in an earlier step
2. **Muse Spark** finds each part's box and connection points → **SAM 3.1** cuts it out (box and box + point prompts are sent together, and the one that differs most from the background is kept; only a crop around the box, scaled to 800 px, is sent, because most of each SAM call is upload time)
3. **Muse Image** generates a photo of A attached to B for each step; when attaching to an assembly, it edits the image generated in the previous step so the steps stay consistent
4. Phone animation (`static/assemble.js`): A's cutout flies to B's connection point → the generated assembly result fades in → zoom in

**Parts can also be filmed**: hold the part and turn it around for a 3–8 second clip (`clip.py`). The video is converted to a 10 fps H.264 MP4 and uploaded to the Files API once. The same file_id goes to **Muse Spark**, which watches the video directly (so it can see connectors on the back), and to **SAM 3.1** for video tracking: give it a box on the clearest frame and one request returns the outline for every frame (60 frames in about 7 seconds, US$0.2 per 1000 frames). In the animation, that part first spins around once, and **Muse Image** also gets a cutout from another angle as reference.

**One video for all parts** (`video_split.py`): lay the parts on the table and film them one at a time, holding 2 seconds on each (150 seconds max).
Don't pick them up and turn them: in testing, when the part was turned in the hand, SAM often cut the hand or the floor into the mask. **Muse Spark** watches the whole video and marks each part's time range and
the moment it is clearest and not blocked by a hand. The clearest frame at that moment is picked from the frames already extracted from the whole video and used as the part photo (no more cutting into clips, converting, or uploading).
For step planning, the whole video is sent once (Spark takes up to 5 videos per request), plus each part's frame.
Spark uses about 260 tokens per second of video, so analyzing one 100-second video costs only about US$0.04 more per run than 16 clips of 5 seconds each.

Measured with 14 parts and a 115-second phone video (through the tunnel, on mobile data): upload about 4 minutes, splitting about 2.5 minutes, planning 83 seconds, cutouts 82 seconds,
so the first step took about 9.5 minutes. After switching to picking frames only, splitting + planning + cutouts for a 6-part test video take about 2 minutes.
Muse Image takes about 20–25 seconds per step, and a step that attaches to an assembly has to wait for the previous step; results fill in while you watch.

Cutout candidates + review: each part is sent to SAM with three kinds of prompts at once (box / box + point, a shrunken box + negative points at the corners, and a text prompt with the English category name).
The resulting masks are drawn on a white background and given to **Muse Spark** together, which picks the one that is "complete, with no table or hands"; if none is right, it falls back to the one that didn't trace the floor.
With only a box, a white cable on a beige table is often traced as the table; with the text prompt and Spark's review, the right one gets picked. This adds about 5–10 seconds per part (parts run in parallel).
The part images used in each analysis are saved in `out/app/<job>_parts/`, so they can be re-run when something goes wrong.

SAM mask decoding (`sam_fast.py`): meta_sam_parser decodes every mask once in pure Python to validate it and then re-encodes it to compare, about 1 second per mask;
a 60-frame video takes over a minute. Decoding only when needed, skipping the comparison, and caching decoded masks brings the same video down from 60+ seconds to about 1 second.

The first step starts playing after about 45 seconds, and the result images for later steps fill in as they finish (about 2 minutes 20 seconds for all 6 parts).
Command line: `.venv/Scripts/python parts_guide.py samples/mic_each/*.jpg --manual samples/mic_manual.jpg` (parts can mix in .mp4/.mov)
The older "mark the parts in one photo" flow is at `/static/photo.html`.

## Try it on a phone

```bash
.venv/Scripts/python server.py
```

Connect the phone to the same Wi-Fi and open the URL printed in the terminal (for example `http://192.168.1.172:8000`).
"See an example" doesn't call the API, so you can look at the guidance screens first.

To use the live camera (opening the camera inside the app requires HTTPS), run this instead:

```bash
.venv/Scripts/python server.py --https
```

A self-signed certificate for the local network is generated automatically (`certs/`, not committed to git). The first time you open it on the phone you'll see a security warning; tap "Advanced" → "Proceed".
On the home page, take one photo each of the "📖 Manual" (optional) and the "🔩 Parts", then tap Start analysis. After about 45 seconds the glowing outlines and arrow animation appear step by step.
The live camera isn't accurate enough yet (the AI can't keep up while the phone moves), so its entry is hidden by default; add `#live` to the end of the URL to show it.

You can also pass the manual on the command line: `.venv/Scripts/python build_guide.py samples/mic_parts.jpg --manual samples/mic_manual.jpg`

**In-app recording**: "Record all parts in the app" on the home page records directly in the browser (MediaRecorder, about 720p, 2.5 Mbps).
60 seconds is about 19 MB; 1080p from the phone's camera app is about 5 times larger. Requires HTTPS (`--https` or the tunnel).

## Live identification: point the camera at a part → which one it is and where it connects (`identify.py`)

After the analysis, tap "📷 Find a part" on the animation screen. Every 0.3 seconds the phone sends the view inside the frame back to the computer, where Meta's open-source **DINOv2** (ViT-B/14,
running on this computer's CPU, about 0.4 seconds per image, no API credits used) compares it with each part's cutout and shows "This is X → connects to Y in step k"
along with what it looks like assembled. How matching works: for each patch (14×14 pixels) of the part cutout, find the most similar patch in the view and take the average;
then convert to standard scores to remove the advantage of parts that "look like everything", such as black cables. When the top two scores are close, **Muse Spark** confirms (about 3–5 seconds).
In testing (photos taken at a different time, matched against cutouts from the phone video), 4 of 6 parts were identified correctly; the one it got wrong was a part whose cutout had failed.

Installation: `requirements.txt` includes the CPU build of PyTorch and transformers.

## Sharing with teammates

The API key stays only on this computer; teammates use the app through a URL, each with their own passcode and a daily usage limit:

```bash
.venv/Scripts/python server.py --add-user alice        # prints alice's passcode (stored in users.json, not committed to git)
.venv/Scripts/python server.py --remove-user alice     # revokes it; no server restart needed
APP_PASSCODE=<your-own> APP_DAILY_RUNS=10 .venv/Scripts/python autodeploy.py --https   # arguments are passed on to server.py
cloudflared tunnel --no-autoupdate --url https://localhost:8000 --no-tls-verify   # prints https://xxx.trycloudflare.com
```

Each analysis costs about US$0.3–0.5; at most 2 analyses run at the same time, and usage is recorded in `out/usage.json`.
The home page shows the daily limit, and before every analysis the app asks the user to confirm using one run (with how many are left today).
The tunnel URL changes every time it restarts; the computer must stay on and must not sleep.

## Developing together

Teammates have Write access to the repo and run it on their own computers (with their own API key). Everyone creates a branch → pushes it → opens a Pull Request → merges into `main`.
`autodeploy.py` checks `main` every 30 seconds and automatically pulls any new commit onto this computer:

- Only `static/` changed: takes effect as soon as it's pulled; no backend restart
- A `.py` file changed: waits until no analysis is running, then restarts `server.py` (unreachable for about 5 seconds)
- `requirements.txt` changed: runs `pip install` first, then restarts

Code merged into `main` runs on this computer with this computer's API key, so review it before merging.

## Running the test scripts

```bash
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements.txt   # macOS/Linux: .venv/bin/python
cp .env.example .env                                  # fill in MODEL_API_KEY
.venv/Scripts/python test_grounding.py samples/parts.jpg \
  --target "longest wooden board" --target "wooden dowel" --sam "wooden dowel" --runs 3
```

Result images and JSON are written to `out/`.

Generate the assembly guide web page (about 60 seconds; `--reuse` only redraws the page without calling the API):

```bash
.venv/Scripts/python build_guide.py samples/parts.jpg
```

Create an API key on the **API keys** page at [dev.meta.ai](https://dev.meta.ai). Do not commit `.env`.
