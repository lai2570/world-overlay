// 組裝引導畫面：在照片（或即時鏡頭）上畫出這一步的兩個零件發光輪廓、壓暗其他區域、A → B 動畫箭頭。
// build_guide.py 產生的單檔網頁和手機 App（index.html）共用這支程式。
//
// D = {w, h, goal, steps: [{part_a, part_b, instruction, why, confidence, a, b, a_rel, b_rel}],
//      shapes: {零件名稱: {d: SVG 路徑, transform}}, boxes: {零件名稱: [x1,y1,x2,y2]}}，座標都是畫面像素。
// a_rel/b_rel 是接口在零件框裡的相對位置；有框時用它算箭頭端點，零件移動時箭頭會跟著走。
// 頁面上需要這些 id：ov(svg) count pa pb inst why conf prev next done
//
// 回傳控制器：
//   setOffset(dx, dy)  整個疊加層平移（即時鏡頭在等 SAM 時用畫面比對補償手機移動）
//   update({w, h, shapes, boxes})  換成新的一格畫面的輪廓（會重畫）
//   step()  目前這一步；onStep 在換步驟時呼叫
function startGuide(D, startStep, onStep) {
  const NS = "http://www.w3.org/2000/svg";
  const $ = id => document.getElementById(id);
  const ov = $("ov");
  const CONF = {high: "把握高", medium: "把握中", low: "把握低"};
  let i = Math.min(Math.max((startStep || 1) - 1, 0), D.steps.length - 1);
  let offset = [0, 0];
  let moving = [];  // 需要跟著 offset 平移的 <g>

  function el(tag, attrs, parent) {
    const e = document.createElementNS(NS, tag);
    for (const [k, v] of Object.entries(attrs)) e.setAttribute(k, v);
    (parent || ov).appendChild(e);
    return e;
  }

  function curve(a, b) {
    // 二次貝茲曲線，往垂直方向彎一點，看起來像「拿起來放過去」
    const mx = (a[0] + b[0]) / 2, my = (a[1] + b[1]) / 2;
    const dx = b[0] - a[0], dy = b[1] - a[1], len = Math.hypot(dx, dy) || 1;
    const k = Math.min(120, len * 0.25);
    return `M${a[0]} ${a[1]} Q${mx - dy / len * k} ${my + dx / len * k} ${b[0]} ${b[1]}`;
  }

  function endpoint(part, rel, fallback) {
    const box = D.boxes && D.boxes[part];
    if (rel && box && box.length === 4) return [box[0] + rel[0] * (box[2] - box[0]), box[1] + rel[1] * (box[3] - box[1])];
    return fallback;
  }

  function shiftAttr() { return `translate(${offset[0]} ${offset[1]})`; }

  function draw() {
    const s = D.steps[i];
    ov.setAttribute("viewBox", `0 0 ${D.w} ${D.h}`);
    ov.replaceChildren();
    const parts = [s.part_a, s.part_b].filter(n => D.shapes[n]);
    // 壓暗其他區域：遮罩挖空 A、B 兩個零件（還沒有輪廓時不壓暗，免得整個畫面變黑）
    if (parts.length) {
      const defs = el("defs", {});
      const mask = el("mask", {id: "hole"}, defs);
      el("rect", {x: -D.w, y: -D.h, width: 3 * D.w, height: 3 * D.h, fill: "#fff"}, mask);
      const holes = el("g", {transform: shiftAttr()}, mask);
      for (const n of parts) el("path", {d: D.shapes[n].d, transform: D.shapes[n].transform, fill: "#000", "fill-rule": "evenodd"}, holes);
      el("rect", {width: D.w, height: D.h, fill: "rgba(0,0,0,.55)", mask: "url(#hole)"});
      moving = [holes];
    } else moving = [];
    const g = el("g", {transform: shiftAttr()});
    moving.push(g);
    for (const [name, cls] of [[s.part_a, "a"], [s.part_b, "b"]]) {
      const sh = D.shapes[name];
      if (sh) el("path", {d: sh.d, transform: sh.transform, class: "part " + cls, "vector-effect": "non-scaling-stroke"}, g);
    }
    const a = endpoint(s.part_a, s.a_rel, s.a), b = endpoint(s.part_b, s.b_rel, s.b);
    if (a && b && (parts.length || !D.live)) {
      const d = curve(a, b);
      el("path", {d, class: "flow"}, g);
      el("circle", {cx: b[0], cy: b[1], r: 22, class: "ring"}, g);
      const dot = el("circle", {r: 11, fill: "var(--a)", stroke: "#fff", "stroke-width": 3}, g);
      const mv = el("animateMotion", {dur: "1.8s", repeatCount: "indefinite", path: d,
        keyPoints: "0;1", keyTimes: "0;1", calcMode: "spline", keySplines: ".45 0 .25 1"}, dot);
      mv.beginElement && mv.beginElement();
    }
    $("count").textContent = `步驟 ${i + 1} / ${D.steps.length}`;
    $("pa").textContent = s.part_a;
    $("pb").textContent = s.part_b;
    $("inst").textContent = s.instruction;
    $("why").textContent = s.why;
    $("conf").textContent = CONF[s.confidence] || "";
    $("prev").disabled = i === 0;
    $("next").textContent = i === D.steps.length - 1 ? "完成" : "下一步";
    $("done").style.display = "none";
  }

  function go(k) {
    i = k;
    draw();
    onStep && onStep(D.steps[i]);
  }

  $("prev").onclick = () => { if (i > 0) go(i - 1); };
  $("next").onclick = () => {
    if (i < D.steps.length - 1) go(i + 1);
    else { ov.replaceChildren(); $("done").style.display = "block"; }
  };
  draw();

  return {
    step: () => D.steps[i],
    setOffset(dx, dy) {
      offset = [dx, dy];
      for (const g of moving) g.setAttribute("transform", shiftAttr());
    },
    update(patch) {
      Object.assign(D, patch);
      offset = [0, 0];
      draw();
    },
  };
}
