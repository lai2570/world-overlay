import sys, time
sys.path.insert(0, 'tools')
from cdp import Browser, record
b = Browser(profile=r'C:\Users\lai25\Desktop\worldoverlay\build\work\chrome_rec', extra=['--enable-unsafe-swiftshader', '--use-angle=swiftshader'])
U = 'http://localhost:8010/'
b.goto(U, 2); b.js("localStorage.setItem('lang','en')"); b.goto(U, 4)
ZOOM = "document.body.style.zoom='0.9'; 1"
# A: landing, then Start -> upload screen
print('A', record(b, [(None, 5), ("document.getElementById('startFlow').click()", 4)], 'work/rec_A'))
b.goto(U, 3)
PLAY = ZOOM + "; currentJob='demoen'; fetch('/static/_demo.json').then(r=>r.json()).then(r=>{player=null; play(r);}); 1"
steps = [(PLAY, 7.5)] + [("document.getElementById('next').click()", 7.5)] * 10 + [("document.getElementById('next').click()", 5)]
print('B', record(b, steps, 'work/rec_B'))
b.goto(U, 3)
SHOW5 = PLAY + "; setTimeout(()=>player.goto(4), 1500); 1"
print('C', record(b, [(SHOW5, 4), ("document.getElementById('btn3d').click()", 14)], 'work/rec_C'))
b.shot('work/rec_C_last.png'); b.close()
