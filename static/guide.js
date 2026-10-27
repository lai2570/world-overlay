// 組裝引導畫面：在照片上畫出這一步的兩個零件發光輪廓、壓暗其他區域、A → B 動畫箭頭。
// build_guide.py 產生的單檔網頁和手機 App（index.html）共用這支程式。
//
// D = {w, h, goal, steps: [{part_a, part_b, instruction, why, confidence, a: [x,y], b: [x,y]}],
//      shapes: {零件名稱: {d: SVG 路徑, transform}}}，座標都是照片的像素。
// 頁面上需要這些 id：ov(svg) count pa pb inst why conf prev next done
function startGuide(D, startStep) {
  const NS = "http://www.w3.org/2000/svg";
  const $ = id => document.getElementById(id);
  const ov = $("ov");
  ov.setAttribute("viewBox", `0 0 ${D.w} ${D.h}`);
  const CONF = {high: "把握高", medium: "把握中", low: "把握低"};
  let i = Math.min(Math.max((startStep || 1) - 1, 0), D.steps.length - 1);

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

  function draw() {
    const s = D.steps[i];
    ov.replaceChildren();
    // 壓暗其他零件：遮罩挖空 A、B 兩個零件
    const defs = el("defs", {});
    const mask = el("mask", {id: "hole"}, defs);
    el("rect", {width: D.w, height: D.h, fill: "#fff"}, mask);
    for (const name of [s.part_a, s.part_b]) {
      const sh = D.shapes[name];
      if (sh) el("path", {d: sh.d, transform: sh.transform, fill: "#000", "fill-rule": "evenodd"}, mask);
    }
    el("rect", {width: D.w, height: D.h, fill: "rgba(0,0,0,.55)", mask: "url(#hole)"});
    for (const [name, cls] of [[s.part_a, "a"], [s.part_b, "b"]]) {
      const sh = D.shapes[name];
      if (sh) el("path", {d: sh.d, transform: sh.transform, class: "part " + cls, "vector-effect": "non-scaling-stroke"});
    }
    if (s.a && s.b) {
      const d = curve(s.a, s.b);
      el("path", {d, class: "flow"});
      el("circle", {cx: s.b[0], cy: s.b[1], r: 22, class: "ring"});
      const dot = el("circle", {r: 11, fill: "var(--a)", stroke: "#fff", "stroke-width": 3});
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

  $("prev").onclick = () => { if (i > 0) { i--; draw(); } };
  $("next").onclick = () => {
    if (i < D.steps.length - 1) { i++; draw(); }
    else { ov.replaceChildren(); $("done").style.display = "block"; }
  };
  draw();
}
