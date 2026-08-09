/**
 * smoketest.mjs — headless boot test for the applet. node docs/smoketest.mjs
 *
 * There is no browser in this dev environment, so this stubs just enough DOM,
 * canvas 2D context and window to import docs/js/app.js and run its real boot
 * path — control wiring, the first simulation frames, the fire-map ImageData
 * write and the Fig 8 panel draw — failing on any exception.
 *
 * It cannot catch visual bugs. It does catch the things that silently break a
 * page and never show up in a unit test: a mistyped element id, a missing
 * data file, a canvas call that does not exist, schema drift between the
 * export scripts and the frontend.
 *
 * Harness adapted from scripts/web_export/smoke_site.mjs by Timothy LaPlaca,
 * which did the same job for the earlier docs/ applet.
 */
import { readFileSync } from 'node:fs';
import { join, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';

const DOCS = dirname(fileURLToPath(import.meta.url));

// Mirror the page's real initial state, so readControls() sees what a fresh
// browser would. Slider defaults are PARSED OUT OF index.html rather than
// duplicated here -- a stub that invents its own values is how you get a
// green smoke test for a page that renders nothing (a `value` defaulting to
// "0" reads as a 0 m/s wind, and a fire that cannot spread looks like a bug
// in the simulation rather than in the harness).
const INDEX_HTML = readFileSync(join(DOCS, 'index.html'), 'utf8');

const SLIDER_DEFAULTS = (() => {
  const out = {};
  for (const tag of INDEX_HTML.match(/<input[^>]*>/g) || []) {
    if (!/type=["']range["']/.test(tag)) continue;
    const id = /id=["']([^"']+)["']/.exec(tag);
    const value = /\svalue=["']([^"']+)["']/.exec(tag);
    if (id && value) out[id[1]] = value[1];
  }
  return out;
})();

const RADIO_DEFAULTS = (() => {
  const out = {};
  for (const tag of INDEX_HTML.match(/<input[^>]*>/g) || []) {
    if (!/type=["']radio["']/.test(tag) || !/\bchecked\b/.test(tag)) continue;
    const name = /name=["']([^"']+)["']/.exec(tag);
    const value = /\svalue=["']([^"']+)["']/.exec(tag);
    if (name && value) out[name[1]] = value[1];
  }
  return out;
})();

if (!Object.keys(SLIDER_DEFAULTS).length || !Object.keys(RADIO_DEFAULTS).length) {
  console.error('FAIL: could not parse control defaults out of index.html');
  process.exit(1);
}

let putImageDataCalls = 0;
let fillTextCalls = 0;

/** Canvas 2D context: real enough for ImageData, no-op for everything drawn. */
function makeCtx() {
  const real = {
    createImageData: (w, h) => ({
      width: w, height: h, data: new Uint8ClampedArray(w * h * 4),
    }),
    putImageData: () => { putImageDataCalls++; },
    measureText: (t) => ({ width: String(t).length * 6 }),
    fillText: () => { fillTextCalls++; },
    strokeText: () => {},
    setTransform: () => {},
  };
  return new Proxy(real, {
    get: (o, k) => (k in o ? o[k] : () => {}),
    set: (o, k, v) => ((o[k] = v), true),
  });
}

function makeEl(tag = 'div') {
  const base = {
    tagName: tag.toUpperCase(),
    style: {}, children: [], value: '0', checked: false, hidden: false,
    textContent: '', innerHTML: '', className: '',
    width: 0, height: 0, clientWidth: 800, clientHeight: 500,
    classList: { add() {}, remove() {}, contains: () => false, toggle() {} },
  };
  return new Proxy(base, {
    get(o, k) {
      if (k in o) return o[k];
      switch (k) {
        case 'getContext': return () => (o.__ctx ||= makeCtx());
        case 'getBoundingClientRect': return () => ({ width: 600, height: 450, top: 0, left: 0 });
        case 'addEventListener': return () => {};
        case 'appendChild': return (c) => { o.children.push(c); return c; };
        case 'querySelectorAll': return () => [];
        case 'querySelector': return () => makeEl();
        case 'getAttribute': return () => null;
        default: return () => {};
      }
    },
    set(o, k, v) { o[k] = v; return true; },
  });
}

const elById = new Map();

/** Resolve `input[name=X]:checked` to the page's actual default. */
function radioFor(selector) {
  const m = /input\[name=(\w+)\]/.exec(selector);
  const el = makeEl('input');
  if (m && RADIO_DEFAULTS[m[1]]) el.value = RADIO_DEFAULTS[m[1]];
  return el;
}

globalThis.document = {
  documentElement: { getAttribute: () => null },
  getElementById(id) {
    if (!elById.has(id)) {
      const el = makeEl(id in SLIDER_DEFAULTS ? 'input' : 'div');
      if (id in SLIDER_DEFAULTS) el.value = SLIDER_DEFAULTS[id];
      elById.set(id, el);
    }
    return elById.get(id);
  },
  querySelector: (sel) => radioFor(sel),
  querySelectorAll: () => [],
  createElement: (tag) => makeEl(tag),
};

globalThis.window = {
  devicePixelRatio: 1,
  addEventListener: () => {},
  matchMedia: () => ({ matches: false, addEventListener() {} }),
};

// Drive a bounded number of animation frames at a fixed 60 fps cadence, then
// stop — the real loop is infinite.
let frames = 0;
const MAX_FRAMES = 40;
globalThis.requestAnimationFrame = (cb) => {
  if (frames++ >= MAX_FRAMES) return 0;
  setTimeout(() => cb(frames * (1000 / 60)), 0);
  return frames;
};

globalThis.fetch = async (url) => {
  try {
    const body = readFileSync(join(DOCS, url), 'utf8');
    return { ok: true, status: 200, json: async () => JSON.parse(body) };
  } catch {
    return { ok: false, status: 404, json: async () => ({}) };
  }
};

// ── boot the real app ──────────────────────────────────────────────────────
await import(join(DOCS, 'js', 'app.js'));
await new Promise((r) => setTimeout(r, 400));

// ── assertions ─────────────────────────────────────────────────────────────
const fails = [];
const get = (id) => String(elById.get(id)?.textContent ?? '');

if (frames < 5) fails.push(`animation loop ran only ${frames} frames`);
if (putImageDataCalls < 5) fails.push(`fire map painted only ${putImageDataCalls} times`);
if (fillTextCalls < 5) fails.push(`Fig 8 panel drew only ${fillTextCalls} labels`);

for (const [id, want] of [
  ['out-ros', /\d/], ['out-ros-ms', /\d/], ['out-intensity', /\d/],
  ['out-flame', /\d/], ['out-load', /\d/], ['out-residence', /\d/],
  ['out-area', /\d/], ['out-perimeter', /\d/], ['out-time', /\d:\d\d/],
  ['windval', /m\/s/], ['moistureval', /%/], ['winddirval', /°/],
  ['speedval', /×/], ['speedval2', /×/], ['domain', /m/],
]) {
  const v = get(id);
  if (!want.test(v)) fails.push(`#${id} not populated (got ${JSON.stringify(v)})`);
}

// The burn must actually have progressed, not merely rendered.
const area = parseFloat(get('out-area'));
if (!(area > 0)) fails.push(`burnt area never grew (got ${JSON.stringify(get('out-area'))})`);

if (fails.length) {
  console.error('FAIL — applet boot path broken:');
  for (const f of fails) console.error(`  · ${f}`);
  process.exit(1);
}

console.log('OK — applet boot path ran clean against real exported data');
console.log(`  ${frames} frames, ${putImageDataCalls} map paints, ${fillTextCalls} panel labels`);
console.log(`  ROS ${get('out-ros')} m/min · ${get('out-intensity')} kW/m · ` +
            `flame ${get('out-flame')} m · burnt ${get('out-area')} ha at t=${get('out-time')}`);
