"""render.py stills t1 t2 ...   |   render.py frames START END PORT  (frame indices, 30 fps)"""
import sys, json, base64, os, time
sys.path.insert(0, 'tools')
from cdp import Browser
W = r'C:\Users\lai25\Desktop\worldoverlay\build\work'
tl = json.load(open(W + r'\timeline.json'))
tl['names'] = json.load(open(W + r'\assets\cut\names.json'))
mode = sys.argv[1]
port = int(sys.argv[4]) if mode == 'frames' else 9400
b = Browser(width=1920, height=1080, scale=1, port=port, profile=W + rf'\chrome_render_{port}')
b.goto('file:///' + W.replace(chr(92), '/') + '/video.html', 1.5)
b.js(f"init({json.dumps(tl)})")
def grab(t, path, fmt='jpeg'):
    b.js(f"setT({t})")
    d = b.call('Page.captureScreenshot', format=fmt, quality=92) if fmt == 'jpeg' else b.call('Page.captureScreenshot', format='png')
    open(path, 'wb').write(base64.b64decode(d['data']))
if mode == 'stills':
    os.makedirs(W + r'\stills', exist_ok=True)
    for t in sys.argv[2:]:
        grab(float(t), W + rf'\stills\t{float(t):07.2f}.jpg')
else:
    s, e = int(sys.argv[2]), int(sys.argv[3])
    os.makedirs(W + r'\frames', exist_ok=True)
    t0 = time.time()
    for f in range(s, e):
        grab(f / 30, W + rf'\frames\{f:05d}.jpg')
        if (f - s) % 300 == 0: print(port, f, round(time.time() - t0), flush=True)
b.close()
