/**
 * plot.js — minimal dependency-free canvas line/scatter plotter.
 *
 * Deliberately hand-rolled (like the rest of the site): no CDN, no build
 * step, view-source friendly.  Supports linear axes, "nice" 1-2-5 ticks,
 * line series, scatter series, and a movable crosshair marker.
 *
 * Usage:
 *   const p = makePlot(canvas, {xLabel, yLabel, xMin, xMax, yMin, yMax});
 *   p.line(xs, ys, {color: "#c33", width: 2});
 *   p.points(xs, ys, {color: "#333", radius: 3});
 *   p.marker(x, y);
 *   p.redraw();   // after changing limits
 */

/** Pick a "nice" tick step (1/2/5 × 10^n) for the given span. */
function niceStep(span, targetCount) {
  const raw = span / Math.max(targetCount, 1);
  const mag = Math.pow(10, Math.floor(Math.log10(raw)));
  for (const m of [1, 2, 5, 10]) {
    if (m * mag >= raw) return m * mag;
  }
  return 10 * mag;
}

/** Format a tick value, trimming float noise ("0.50" -> "0.5", "10" stays "10"). */
function fmtTick(v, step) {
  const decimals = Math.max(0, -Math.floor(Math.log10(step) + 1e-9));
  let s = v.toFixed(Math.min(decimals, 6));
  if (s.includes(".")) s = s.replace(/0+$/, "").replace(/\.$/, "");
  return s === "-0" ? "0" : s;
}

export function makePlot(canvas, opts) {
  const o = Object.assign(
    {
      xLabel: "",
      yLabel: "",
      xMin: 0,
      xMax: 1,
      yMin: 0,
      yMax: 1,
      y2Max: null, // optional: unused placeholder for future dual-axis
    },
    opts,
  );

  const margin = { left: 56, right: 14, top: 12, bottom: 40 };
  const state = { series: [], marker: null };
  const ctx = canvas.getContext("2d");

  function setupCanvas() {
    const dpr = window.devicePixelRatio || 1;
    const w = canvas.clientWidth;
    const h = canvas.clientHeight;
    if (canvas.width !== w * dpr || canvas.height !== h * dpr) {
      canvas.width = w * dpr;
      canvas.height = h * dpr;
    }
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    return { w, h };
  }

  function xToPx(x, w) {
    return margin.left + ((x - o.xMin) / (o.xMax - o.xMin)) * (w - margin.left - margin.right);
  }
  function yToPx(y, h) {
    return h - margin.bottom - ((y - o.yMin) / (o.yMax - o.yMin)) * (h - margin.top - margin.bottom);
  }

  function drawAxes(w, h) {
    ctx.clearRect(0, 0, w, h);
    ctx.font = "12px system-ui, sans-serif";
    ctx.lineWidth = 1;

    // grid + ticks
    const xStep = niceStep(o.xMax - o.xMin, 8);
    const yStep = niceStep(o.yMax - o.yMin, 6);

    ctx.strokeStyle = "#e2e2e2";
    ctx.fillStyle = "#555";
    ctx.textAlign = "center";
    for (let x = Math.ceil(o.xMin / xStep) * xStep; x <= o.xMax + 1e-12; x += xStep) {
      const px = xToPx(x, w);
      ctx.beginPath();
      ctx.moveTo(px, margin.top);
      ctx.lineTo(px, h - margin.bottom);
      ctx.stroke();
      ctx.fillText(fmtTick(x, xStep), px, h - margin.bottom + 16);
    }
    ctx.textAlign = "right";
    for (let y = Math.ceil(o.yMin / yStep) * yStep; y <= o.yMax + 1e-12; y += yStep) {
      const py = yToPx(y, h);
      ctx.beginPath();
      ctx.moveTo(margin.left, py);
      ctx.lineTo(w - margin.right, py);
      ctx.stroke();
      ctx.fillText(fmtTick(y, yStep), margin.left - 6, py + 4);
    }

    // axis box
    ctx.strokeStyle = "#333";
    ctx.strokeRect(margin.left, margin.top, w - margin.left - margin.right, h - margin.top - margin.bottom);

    // labels
    ctx.fillStyle = "#222";
    ctx.textAlign = "center";
    ctx.fillText(o.xLabel, margin.left + (w - margin.left - margin.right) / 2, h - 6);
    ctx.save();
    ctx.translate(14, margin.top + (h - margin.top - margin.bottom) / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.fillText(o.yLabel, 0, 0);
    ctx.restore();
  }

  function draw(w, h) {
    drawAxes(w, h);
    // clip to plot area
    ctx.save();
    ctx.beginPath();
    ctx.rect(margin.left, margin.top, w - margin.left - margin.right, h - margin.top - margin.bottom);
    ctx.clip();

    for (const s of state.series) {
      ctx.strokeStyle = s.color;
      ctx.fillStyle = s.color;
      ctx.lineWidth = s.width;
      if (s.kind === "line") {
        ctx.beginPath();
        let started = false;
        for (let i = 0; i < s.xs.length; i++) {
          const px = xToPx(s.xs[i], w);
          const py = yToPx(s.ys[i], h);
          if (!started) { ctx.moveTo(px, py); started = true; }
          else ctx.lineTo(px, py);
        }
        ctx.stroke();
      } else if (s.kind === "points") {
        for (let i = 0; i < s.xs.length; i++) {
          const px = xToPx(s.xs[i], w);
          const py = yToPx(s.ys[i], h);
          ctx.beginPath();
          ctx.arc(px, py, s.radius, 0, 2 * Math.PI);
          if (s.hollow) { ctx.lineWidth = 1.2; ctx.stroke(); }
          else ctx.fill();
        }
      }
    }

    if (state.marker) {
      const { x, y } = state.marker;
      const px = xToPx(x, w);
      const py = yToPx(y, h);
      ctx.strokeStyle = "#000";
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      ctx.arc(px, py, 6, 0, 2 * Math.PI);
      ctx.stroke();
      ctx.beginPath();
      ctx.moveTo(px - 10, py); ctx.lineTo(px + 10, py);
      ctx.moveTo(px, py - 10); ctx.lineTo(px, py + 10);
      ctx.stroke();
    }
    ctx.restore();
  }

  const api = {
    /** Replace all series and redraw. Series: {kind, xs, ys, color, width|radius, hollow}. */
    setSeries(series) { state.series = series; },
    setMarker(x, y) { state.marker = x == null ? null : { x, y }; },
    setLimits(xMin, xMax, yMin, yMax) {
      Object.assign(o, { xMin, xMax, yMin, yMax });
    },
    redraw() {
      const { w, h } = setupCanvas();
      if (w > 0 && h > 0) draw(w, h);
    },
  };

  window.addEventListener("resize", () => api.redraw());
  return api;
}
