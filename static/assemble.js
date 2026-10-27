// 組裝動畫：零件 A 的去背圖飛到 B（零件去背圖或前一步的組合件）的接口，接著淡入 Muse Image 生成的組裝結果。
// parts_guide.py 的結果格式：
//   D = {goal, steps: [{part_a, part_b, instruction, why, confidence, a_scale, onto_step,
//                       a_img, a_pt, b_img, b_pt, result}], images: {key: data URL}, pending,
//        spins: {零件名稱: [轉一圈的去背圖 data URL...]}}（零件是用影片拍的才有 spins）
//   a_pt / b_pt 是接口在圖片裡的相對位置（0–1），可能是 null（AI 沒找到，改用圖片中心）。
// 頁面上需要這些 id：stage count pa pb inst why conf prev next replay done genTip
function startAssembly(D, onStep) {
  const $ = id => document.getElementById(id);
  const stage = $("stage");
  const CONF = {high: "把握高", medium: "把握中", low: "把握低"};
  const T = {look: 900, spin: 1800, fly: 1500, fade: 700, hold: 1800};  // 各段時間（毫秒）；spin：零件是影片拍的就先轉一圈
  const LEFT = 0.36;  // 左邊放零件 A 的區域寬度比例；B 和結果圖放右邊
  let i = 0, raf = 0, t0 = 0, els = null;

  const clamp = (v, a, b) => Math.min(b, Math.max(a, v));
  const ease = t => t < .5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2;
  const lerp = (a, b, t) => a + (b - a) * t;

  function img(src, cls) {
    const e = document.createElement("img");
    e.src = src; e.className = cls; e.alt = "";
    stage.appendChild(e);
    return e;
  }

  // 等比例放進 (x, y, w, h) 區域，置中
  function fit(nw, nh, x, y, w, h) {
    const k = Math.min(w / nw, h / nh);
    return {x: x + (w - nw * k) / 2, y: y + (h - nh * k) / 2, w: nw * k, h: nh * k};
  }

  function pt(rel, r) {
    const p = rel && rel.length === 2 ? rel : [.5, .5];
    return [r.x + clamp(p[0], 0, 1) * r.w, r.y + clamp(p[1], 0, 1) * r.h];
  }

  async function build() {
    cancelAnimationFrame(raf);
    stage.replaceChildren();
    const s = D.steps[i];
    const srcA = D.images[s.a_img], srcB = D.images[s.b_img], srcR = s.result && D.images[s.result];
    const W = stage.clientWidth, H = stage.clientHeight, pad = 12;
    const eB = srcB && img(srcB, "asm b" + (s.b_img.startsWith("part:") ? " cut" : ""));
    const eA = srcA && img(srcA, "asm a cut");
    const eR = srcR && img(srcR, "asm r");
    const spins = D.spins || {};
    const spinA = eA && spins[s.part_a] || null;
    const spinB = eB && s.b_img.startsWith("part:") && spins[s.part_b] || null;
    const spinEls = [...(spinA || []), ...(spinB || [])].map(src => Object.assign(new Image(), {src}));  // 先載入，轉的時候才不會閃
    await Promise.all([eA, eB, eR, ...spinEls].filter(Boolean).map(e => e.decode().catch(() => {})));
    if (stage.dataset.step !== String(i)) return;  // 等圖片時已經換步驟

    const right = {x: W * LEFT, y: pad, w: W * (1 - LEFT) - pad, h: H - 2 * pad};
    const rB = eB ? fit(eB.naturalWidth, eB.naturalHeight, right.x, right.y, right.w, right.h) : right;
    const rR = eR ? fit(eR.naturalWidth, eR.naturalHeight, right.x, right.y, right.w, right.h) : null;
    // 結果圖淡入後放大到整個畫面（下一步再從右邊開始，和這一步的 B 位置一致）
    const rFull = eR ? fit(eR.naturalWidth, eR.naturalHeight, pad, pad, W - 2 * pad, H - 2 * pad) : null;
    // A 開始時放在左邊區域；裝上去時依 a_scale 換算成相對 B 的大小
    const aw = eA ? eA.naturalWidth : 1, ah = eA ? eA.naturalHeight : 1;
    const start = fit(aw, ah, pad, H * .2, W * LEFT - 2 * pad, H * .6);
    const len = clamp(s.a_scale || .4, .12, 1) * Math.max(rB.w, rB.h);
    const k = len / Math.max(aw, ah);
    const bp = pt(s.b_pt, rB);
    const ap = s.a_pt && s.a_pt.length === 2 ? [clamp(s.a_pt[0], 0, 1), clamp(s.a_pt[1], 0, 1)] : [.5, .5];
    const end = {w: aw * k, h: ah * k};
    end.x = bp[0] - ap[0] * end.w; end.y = bp[1] - ap[1] * end.h;

    // 虛線箭頭（從 A 的接口到 B 的接口）和 B 接口上的光圈
    const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
    svg.setAttribute("class", "asm-ov");
    svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
    const a0 = [start.x + ap[0] * start.w, start.y + ap[1] * start.h];
    const mx = (a0[0] + bp[0]) / 2, my = Math.min(a0[1], bp[1]) - 60;
    const path = `M${a0[0]} ${a0[1]} Q${mx} ${my} ${bp[0]} ${bp[1]}`;
    svg.innerHTML = `<path d="${path}" class="flow"/><circle cx="${bp[0]}" cy="${bp[1]}" r="16" class="ring"/>`;
    stage.appendChild(svg);

    const place = (e, r) => { e.style.left = r.x + "px"; e.style.top = r.y + "px"; e.style.width = r.w + "px"; e.style.height = r.h + "px"; };
    if (eB) place(eB, rB);
    if (eR) { place(eR, rR); eR.style.opacity = 0; }
    els = {eA, eB, eR, svg, start, end, rR, rFull, place, spinA, spinB, srcA, srcB, look: spinA || spinB ? T.spin : T.look};
    t0 = performance.now();
    raf = requestAnimationFrame(tick);
  }

  function tick(now) {
    const {eA, eB, eR, svg, start, end, rR, rFull, place, spinA, spinB, srcA, srcB, look} = els;
    const total = look + T.fly + (eR ? T.fade + T.hold : T.hold);
    const t = (now - t0) % total;
    const fly = ease(clamp((t - look) / T.fly, 0, 1));
    const fade = eR ? clamp((t - look - T.fly) / T.fade, 0, 1) : 0;
    // 影片拍的零件先轉一圈給你看每一面，再停在最清楚的那一面（接口位置是在那一面上找的）
    const turn = (e, frames, still) => {
      const src = t < look ? frames[Math.floor(t / look * frames.length) % frames.length] : still;
      if (e.getAttribute("src") !== src) e.src = src;
    };
    if (spinA) turn(eA, spinA, srcA);
    if (spinB) turn(eB, spinB, srcB);
    if (eA) {
      // 位置走弧線：中間往上抬一點，看起來像拿起來放過去
      const lift = Math.sin(fly * Math.PI) * 60;
      eA.style.left = lerp(start.x, end.x, fly) + "px";
      eA.style.top = lerp(start.y, end.y, fly) - lift + "px";
      eA.style.width = lerp(start.w, end.w, fly) + "px";
      eA.style.height = lerp(start.h, end.h, fly) + "px";
      eA.style.opacity = 1 - fade;
    }
    if (eB) eB.style.opacity = 1 - fade;
    if (eR) {
      eR.style.opacity = fade;
      const grow = ease(clamp((t - look - T.fly - T.fade) / 600, 0, 1));
      place(eR, {x: lerp(rR.x, rFull.x, grow), y: lerp(rR.y, rFull.y, grow), w: lerp(rR.w, rFull.w, grow), h: lerp(rR.h, rFull.h, grow)});
    }
    svg.style.opacity = fly < 1 ? 1 : 0;
    raf = requestAnimationFrame(tick);
  }

  function draw() {
    const s = D.steps[i];
    stage.dataset.step = String(i);
    $("count").textContent = `步驟 ${i + 1} / ${D.steps.length}`;
    $("pa").textContent = s.part_a;
    $("pb").textContent = s.onto_step ? `組合件（${s.part_b}）` : s.part_b;
    $("inst").textContent = s.instruction;
    $("why").textContent = s.why;
    $("conf").textContent = CONF[s.confidence] || "";
    $("prev").disabled = i === 0;
    $("next").textContent = i === D.steps.length - 1 ? "完成" : "下一步";
    $("done").style.display = "none";
    const waiting = !s.result && D.pending > 0;
    $("genTip").hidden = !waiting;
    build();
  }

  function go(k) { i = k; draw(); onStep && onStep(i); }

  $("prev").onclick = () => { if (i > 0) go(i - 1); };
  $("next").onclick = () => {
    if (i < D.steps.length - 1) go(i + 1);
    else { cancelAnimationFrame(raf); $("done").style.display = "block"; }
  };
  $("replay").onclick = () => { t0 = performance.now(); };
  window.addEventListener("resize", () => build());
  draw();

  return {
    // 生成中的結果圖陸續送到：只有目前這一步的圖變了才重畫
    update(next) {
      const before = JSON.stringify(D.steps[i]);
      D = next;
      if (JSON.stringify(D.steps[i]) !== before || (!D.pending && !$("genTip").hidden)) draw();
    },
    stop() { cancelAnimationFrame(raf); },
    seek(ms) { t0 = performance.now() - ms; },  // 跳到動畫的第 ms 毫秒（測試和截圖用）
  };
}
