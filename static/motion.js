// 鏡頭平移估算：把畫面縮成很小的灰階邊緣圖，和關鍵畫面（上次送去 SAM 的那一格）比對，
// 找出內容整體平移了多少。每秒可以跑十幾次，讓發光輪廓在等 SAM 的空檔也跟著畫面走。
// 只估平移，不估縮放和旋轉；手機大幅移動時交給伺服器重新描。
class MotionTracker {
  constructor(width = 96, range = 14) {
    this.w = width;      // 縮圖寬度（像素）
    this.r = range;      // 最大搜尋位移（縮圖像素），約為畫面寬度的 15%
    this.canvas = document.createElement("canvas");
    this.ctx = this.canvas.getContext("2d", {willReadFrequently: true});
    this.key = null;
  }

  // 取得縮圖的邊緣強度圖（對光線變化比原始亮度穩定）
  snapshot(source) {
    const sw = source.videoWidth || source.width, sh = source.videoHeight || source.height;
    const w = this.w, h = Math.round(w * sh / sw);
    this.canvas.width = w; this.canvas.height = h;
    this.ctx.drawImage(source, 0, 0, w, h);
    const px = this.ctx.getImageData(0, 0, w, h).data;
    const gray = new Float32Array(w * h);
    for (let i = 0; i < w * h; i++) gray[i] = px[4 * i] * .299 + px[4 * i + 1] * .587 + px[4 * i + 2] * .114;
    const edge = new Float32Array(w * h);
    for (let y = 0; y < h - 1; y++) for (let x = 0; x < w - 1; x++) {
      const i = y * w + x;
      edge[i] = Math.abs(gray[i + 1] - gray[i]) + Math.abs(gray[i + w] - gray[i]);
    }
    return {edge, w, h, scale: sw / w};
  }

  setKeyframe(snap) { this.key = snap; }

  // 平均絕對差：關鍵畫面中央區域 vs 目前畫面位移 (dx, dy) 後的同一區域
  cost(cur, dx, dy) {
    const {edge: a, w, h} = this.key, b = cur.edge, r = this.r;
    let sum = 0, n = 0;
    for (let y = r; y < h - r; y += 1) {
      const row = y * w, row2 = (y + dy) * w + dx;
      for (let x = r; x < w - r; x += 1) { sum += Math.abs(a[row + x] - b[row2 + x]); n++; }
    }
    return sum / n;
  }

  // 回傳目前畫面相對於關鍵畫面的平移（原始畫面像素）；沒有關鍵畫面或尺寸不同時回傳 null
  estimate(source) {
    if (!this.key) return null;
    const cur = this.snapshot(source);
    if (cur.w !== this.key.w || cur.h !== this.key.h) return null;
    let best = {dx: 0, dy: 0, c: Infinity};
    const r = this.r;
    for (let dy = -r; dy <= r; dy += 2) for (let dx = -r; dx <= r; dx += 2) {  // 粗搜
      const c = this.cost(cur, dx, dy);
      if (c < best.c) best = {dx, dy, c};
    }
    const coarse = best;
    for (let dy = coarse.dy - 1; dy <= coarse.dy + 1; dy++) for (let dx = coarse.dx - 1; dx <= coarse.dx + 1; dx++) {  // 細修
      if (Math.abs(dx) > r || Math.abs(dy) > r) continue;
      const c = this.cost(cur, dx, dy);
      if (c < best.c) best = {dx, dy, c};
    }
    const atEdge = Math.abs(best.dx) >= r - 1 || Math.abs(best.dy) >= r - 1;  // 移太多，估算不可信
    return {dx: best.dx * cur.scale, dy: best.dy * cur.scale, reliable: !atEdge, snap: cur};
  }
}
