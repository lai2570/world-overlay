"""Synthesize the soundtrack on the timeline's beat grid and mix in the narration -> work/mix.wav"""
import json, wave, sys
VO_ONLY = "--vo-only" in sys.argv
import numpy as np

W = r"C:\Users\lai25\Desktop\worldoverlay\build\work"
SR = 48000
tl = json.load(open(W + r"\timeline.json"))
T = tl["total"]
N = int(T * SR)
BPM = 96
BEAT = 60 / BPM            # 0.625 s; every scene starts on a beat
t = np.arange(N) / SR
rng = np.random.default_rng(7)
start = {s["id"]: s["start"] for s in tl["scenes"]}
REVEAL = start["s4"]       # arps + drums come in with the product
END = start["s7"]

def midi(n): return 440 * 2 ** ((n - 69) / 12)

# Am - F - C - G, 8 beats each
CHORDS = [(57, [57, 60, 64]), (53, [53, 57, 60]), (48, [55, 60, 64]), (55, [55, 59, 62])]
BAR8 = 8 * BEAT

def chord_at(x):
    return CHORDS[int(x // BAR8) % 4]

music = np.zeros((N, 2))

# --- pad: soft detuned triangles per chord block, slow attack/release
def tri(f, tt):
    return 2 * np.abs(2 * ((f * tt) % 1) - 1) - 1

nblocks = int(np.ceil(T / BAR8))
for b in range(nblocks):
    s0 = int(b * BAR8 * SR); s1 = min(N, int((b + 1) * BAR8 * SR) + int(.4 * SR))
    if s0 >= N: break
    tt = t[s0:s1] - b * BAR8
    root, notes = CHORDS[b % 4]
    env = np.minimum(1, tt / 1.2) * np.clip((BAR8 + .4 - tt) / .8, 0, 1)
    sig_l = sum(tri(midi(n) * 1.003, tt) + tri(midi(n + 12) * .997, tt) * .3 for n in notes)
    sig_r = sum(tri(midi(n) * .997, tt) + tri(midi(n + 12) * 1.003, tt) * .3 for n in notes)
    music[s0:s1, 0] += sig_l * env * .045
    music[s0:s1, 1] += sig_r * env * .045

# --- bass: root on beats 1 and 3 of every bar (from s2), sine + a little 2nd harmonic
for k in range(int(T / BEAT)):
    bt = k * BEAT
    if bt < start["s2"] or bt >= T - BEAT: continue
    if k % 2: continue
    root, _ = chord_at(bt)
    s0 = int(bt * SR); s1 = min(N, s0 + int(BEAT * 1.9 * SR))
    tt = t[s0:s1] - bt
    f = midi(root - 12)
    env = np.exp(-tt * 1.6) * np.minimum(1, tt / .01)
    sig = (np.sin(2 * np.pi * f * tt) + .25 * np.sin(4 * np.pi * f * tt)) * env * .16
    music[s0:s1] += sig[:, None]

# --- pluck arpeggio, 8th notes, from the reveal until the outro
EIGHTH = BEAT / 2
pattern = [0, 1, 2, 1, 0, 2, 1, 2]
for k in range(int(REVEAL / EIGHTH), int((END + 4 * BEAT) / EIGHTH)):
    et = k * EIGHTH
    _, notes = chord_at(et)
    n = notes[pattern[k % 8]] + 12
    s0 = int(et * SR); s1 = min(N, s0 + int(.5 * SR))
    tt = t[s0:s1] - et
    env = np.exp(-tt * 9) * np.minimum(1, tt / .004)
    sig = (tri(midi(n), tt) * .6 + np.sin(2 * np.pi * midi(n) * 2 * tt) * .2) * env * .05
    pan = .35 if k % 2 else -.35
    music[s0:s1, 0] += sig * (1 - pan)
    music[s0:s1, 1] += sig * (1 + pan)

# --- soft kick on every beat + quiet hats on off-beats, from s5a to the outro
for k in range(int(start["s5a"] / BEAT), int(END / BEAT)):
    bt = k * BEAT
    s0 = int(bt * SR); s1 = min(N, s0 + int(.35 * SR))
    tt = t[s0:s1] - bt
    f = 48 + 70 * np.exp(-tt * 30)
    ph = 2 * np.pi * np.cumsum(f) / SR
    kick = np.sin(ph) * np.exp(-tt * 9) * .22
    music[s0:s1] += kick[:, None]
    h0 = int((bt + BEAT / 2) * SR); h1 = min(N, h0 + int(.06 * SR))
    if h0 < N:
        th = t[h0:h1] - (bt + BEAT / 2)
        noise = rng.standard_normal(h1 - h0)
        noise = noise - np.concatenate([[0], noise[:-1]])          # crude high-pass
        music[h0:h1] += (noise * np.exp(-th * 70) * .018)[:, None]

# --- gentle swell into each scene cut (soft riser of filtered noise, 1 beat)
for s in tl["scenes"][1:]:
    c = s["start"]; s0 = int((c - BEAT) * SR); s1 = int(c * SR)
    tt = np.linspace(0, 1, s1 - s0)
    noise = rng.standard_normal(s1 - s0)
    noise = np.convolve(noise, np.ones(24) / 24, mode="same")
    music[s0:s1] += (noise * tt ** 2 * .05)[:, None]

# --- narration
vo = np.zeros(N)
for v in tl["vo"]:
    w = wave.open(rf"{W}\vo\{v['id']}.wav")
    x = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float64) / 32768
    if w.getframerate() != SR:
        x = np.interp(np.arange(0, len(x), w.getframerate() / SR), np.arange(len(x)), x)
    s0 = int(v["start"] * SR); s1 = min(N, s0 + len(x))
    vo[s0:s1] += x[:s1 - s0]
vo = vo / (np.abs(vo).max() + 1e-9) * .89

# --- ducking: music down while narration plays (smooth 0.25 s ramps)
duck = np.ones(N)
for v in tl["vo"]:
    s0 = int((v["start"] - .25) * SR); s1 = int((v["start"] + v["dur"] + .3) * SR)
    duck[max(0, s0):min(N, s1)] = .42
k = int(.25 * SR)
c = np.concatenate([[0], np.cumsum(np.pad(duck, (k // 2, k - k // 2), mode="edge"))])
duck = (c[k:] - c[:-k])[:N] / k

# fades
fade = np.ones(N)
fi = int(1.0 * SR); fade[:fi] = np.linspace(0, 1, fi)
fo = int(3.5 * SR); fade[-fo:] = np.linspace(1, 0, fo) ** 1.5

if VO_ONLY: music[:] = 0
music *= (duck * fade)[:, None]
music = music / (np.abs(music).max() + 1e-9) * .5
mix = music + vo[:, None]
peak = np.abs(mix).max()
if peak > .97: mix *= .97 / peak
out = (mix * 32767).astype(np.int16)
w = wave.open(W + r"\mix.wav", "wb"); w.setnchannels(2); w.setsampwidth(2); w.setframerate(SR)
w.writeframes(out.tobytes()); w.close()
print("ok", round(T, 2), "s, music peak", round(float(np.abs(music).max()), 2), "mix peak", round(float(np.abs(mix).max()), 2))
