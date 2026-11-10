#!/usr/bin/env bash
# Rebuild the 3-minute demo video (brag.mp4) from scratch. Run from this video/ folder in Git Bash.
# The Python tools still have the original absolute path hard-coded:
#   W = C:\Users\lai25\Desktop\worldoverlay\build\work
# Change W in render.py / audio.py (and the profile paths in record_app.py) to point at ./work before running.
#
# Inputs:
#   assets/        photos, SAM cutouts (cut/), Muse Image step results, logo   (in git)
#   demoen_parts.json   the real analysis job 2f01fa90 with step text translated to English (in git)
#   footage.mp4    raw phone footage of the parts (GitHub Release "demo-video", 132 MB)
#   demoen_3d/     3D models for that job (GitHub Release zip)
set -e
FF=${FF:-ffmpeg}            # imageio-ffmpeg's binary works: .venv/Lib/site-packages/imageio_ffmpeg/binaries/ffmpeg-*.exe
PY=${PY:-python}            # needs numpy (the app's .venv has it)
mkdir -p work

# 1) Record the real app in headless Chrome (needs the app running on :8010 with out/app/demoen_parts.json,
#    out/app/demoen_3d/ and static/_demo.json = demoen_parts.json; MODEL_API_KEY can be a dummy, no API calls are made)
#    -> work/rec_A, rec_B, rec_C frame folders + frames.txt (ffmpeg concat lists)
$PY record_app.py
for r in rec_A rec_B rec_C; do
  $FF -y -f concat -safe 0 -i work/$r/frames.txt -vf fps=30 -pix_fmt yuv420p work/$r.mp4
done

# 2) Cut frame sequences the compositor plays (assets/seq/<name>/0000.jpg ...)
S=assets/seq
x(){ mkdir -p $S/$1; $FF -hide_banner -loglevel error -y -ss $3 -t $4 -i $2 -vf "$5" -q:v 3 -start_number 0 $S/$1/%04d.jpg; }
x foot1 footage.mp4 0  9 "setpts=0.5*PTS,fps=30,scale=720:1280"    # real footage, shown at 2x speed
x foot2 footage.mp4 26 9 "setpts=0.5*PTS,fps=30,scale=720:1280"
x foot3 footage.mp4 72 9 "setpts=0.5*PTS,fps=30,scale=720:1280"
x recA  work/rec_A.mp4 0 9 "fps=30"
x recB1 work/rec_B.mp4 0 22.5 "fps=30"
for k in 4 5 6 7 8 9 10; do st=$(awk "BEGIN{print 7.5*($k-1)+0.4}"); x mont$k work/rec_B.mp4 $st 3.75 "fps=30"; done
x montdone work/rec_B.mp4 83.2 3.75 "fps=30"
x recC  work/rec_C.mp4 3.5 12.5 "fps=30"

# 3) Narration: Windows SAPI voice (Microsoft Zira), one wav per scene from vo/script.json
#    (the wavs used in the final cut are already in vo/)
# powershell -File vo/tts.ps1 -text "..." -out vo/s1.wav

# 4) Render every frame of video.html deterministically (setT(t) + screenshot), 167.5 s * 30 fps = 5025 frames.
#    Split across processes for speed: render.py frames START END PORT
for i in 0 1 2 3; do $PY render.py frames $((i*1257)) $(((i+1)*1257)) $((9400+i)) & done; wait

# 5) Audio: narration only (what was submitted). Drop --vo-only to get the synthesized music bed back.
$PY audio.py --vo-only          # -> work/mix.wav

# 6) Encode. Frame 2232 (13 cutouts + "11 assembly steps planned") doubles as poster / frame 0.
cp work/frames/02232.jpg brag.jpg
$FF -y -framerate 30 -i work/frames/%05d.jpg -i work/mix.wav -map 0:v -map 1:a \
  -vf "scale=in_range=full:out_range=tv" -c:v libx264 -preset slow -crf 18 -pix_fmt yuv420p \
  -movflags +faststart -c:a aac -b:a 192k -shortest brag.mp4
