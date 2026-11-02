// 3D 檢視：這一步的零件 A、B 換成 Hunyuan3D 生成的形狀（model3d.py → /api/models），手指拖曳旋轉，支援的手機可以按 AR 放到桌上。
// 模型在背景一個一個做（每個約 60 秒），還沒好的顯示「生成中」，好了自動換上。
// 頁面上需要這些 id：view3d btn3d；api() 和 passcodeHeaders() 由 index.html 提供。
const MODEL_VIEWER = "https://unpkg.com/@google/model-viewer@3.5.0/dist/model-viewer.min.js";

function startModels(job, getStep, onToggle) {
  const $ = id => document.getElementById(id);
  const view = $("view3d"), btn = $("btn3d");
  const COLOR = {a: [1, .69, .13, 1], b: [.13, .83, .93, 1]};  // 和 guide.css 的 --a、--b 一樣
  let parts = {}, on = false, timer = 0, shown = "", stopped = false;
  const urls = {};  // 零件 → Promise<blob URL>（glb 要帶通行碼標頭，model-viewer 自己抓不行）

  function glb(name) {
    urls[name] ??= fetch(`/api/model?job=${encodeURIComponent(job)}&part=${encodeURIComponent(name)}`, {headers: passcodeHeaders()})
      .then(r => r.ok ? r.blob() : Promise.reject(new Error(r.statusText)))
      .then(b => URL.createObjectURL(b));
    return urls[name];
  }

  async function refresh() {
    let r;
    try { r = await api(`/api/models?job=${encodeURIComponent(job)}`); } catch (e) { r = null; }
    if (stopped) return;
    if (r) {
      btn.hidden = !r.on;
      parts = r.parts;
      render();
    }
    // 還有在做的（或剛開始做、清單還沒出來）就繼續問
    if (!r || Object.values(parts).includes("pending") || (r.on && !Object.keys(parts).length)) timer = setTimeout(refresh, 4000);
  }

  function pane(name, cls, label) {
    const div = document.createElement("div");
    div.className = "pane";
    div.innerHTML = `<span class="chip ${cls}"></span>`;
    div.firstChild.textContent = label;
    const st = parts[name];
    if (st !== "ready") {
      const p = document.createElement("p");
      p.className = "wait";
      p.textContent = st === "failed" ? "這個零件做不出 3D 形狀" : "3D 模型生成中…（一個零件約 1 分鐘）";
      div.appendChild(p);
      return div;
    }
    const mv = document.createElement("model-viewer");
    for (const [k, v] of Object.entries({"camera-controls": "", "auto-rotate": "", "auto-rotate-delay": "0", ar: "",
      "ar-modes": "webxr quick-look", "shadow-intensity": "1", exposure: "1.1", "interaction-prompt": "none", alt: name}))
      mv.setAttribute(k, v);
    mv.addEventListener("load", () => {  // 生成的形狀沒有材質，塗上 A／B 的顏色
      for (const m of mv.model?.materials || []) {
        m.pbrMetallicRoughness.setBaseColorFactor(COLOR[cls]);
        m.pbrMetallicRoughness.setRoughnessFactor(.55);
      }
    });
    glb(name).then(u => { mv.src = u; }, () => { mv.replaceWith(Object.assign(document.createElement("p"), {className: "wait", textContent: "模型載入失敗"})); });
    div.appendChild(mv);
    return div;
  }

  function render() {
    if (!on) return;
    const s = getStep();
    const key = JSON.stringify([s.part_a, s.part_b, parts[s.part_a], parts[s.part_b]]);
    if (key === shown) return;  // 輪詢時沒變就不要重建（不然轉到一半會跳回去）
    shown = key;
    view.replaceChildren(pane(s.part_a, "a", s.part_a),
                         pane(s.part_b, "b", s.onto_step ? `裝到組合件（${s.part_b}）` : `裝到 ${s.part_b}`));
  }

  function toggle(v) {
    on = v;
    view.hidden = !on;
    btn.textContent = on ? "▶ 動畫" : "🧊 3D";
    if (on && !customElements.get("model-viewer") && !document.querySelector(`script[src="${MODEL_VIEWER}"]`))
      document.head.appendChild(Object.assign(document.createElement("script"), {type: "module", src: MODEL_VIEWER}));
    shown = "";
    render();
    onToggle && onToggle(on);
  }

  btn.onclick = () => toggle(!on);
  btn.hidden = true;
  refresh();

  return {
    step() { render(); },  // 換步驟時呼叫
    get on() { return on; },
    stop() {
      stopped = true;
      clearTimeout(timer);
      if (on) toggle(false);
      btn.hidden = true;
      for (const p of Object.values(urls)) p.then(URL.revokeObjectURL, () => {});
    },
  };
}
