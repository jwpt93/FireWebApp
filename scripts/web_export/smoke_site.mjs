#!/usr/bin/env node
/**
 * smoke_site.mjs — headless smoke test for the static site.
 *
 * No browser is available in the dev environment, so this stubs just enough
 * DOM (elements, canvas 2D context, window) to import docs/js/app.js and run
 * its full boot path — presets, Tier-1 redraw, Tier-2 run loading (against
 * the REAL exported JSON in docs/data/) and the gallery — and fails on any
 * exception.  It cannot catch visual bugs, but it catches syntax errors,
 * bad element ids, and data-schema drift between the export scripts and
 * the frontend.
 *
 * Run from the repo root:  node scripts/web_export/smoke_site.mjs
 */

import { readFileSync } from "node:fs";
import { join, dirname } from "node:path";
import { fileURLToPath } from "node:url";

const DOCS = join(dirname(fileURLToPath(import.meta.url)), "..", "..", "docs");

// ── Minimal canvas 2D context: absorb everything ────────────────────────────
const ctxStub = new Proxy({}, {
  get: (o, k) => (k in o ? o[k] : () => {}),
  set: (o, k, v) => ((o[k] = v), true),
});

// ── Minimal element stub ────────────────────────────────────────────────────
function makeEl(tag = "div") {
  const base = {
    tagName: tag.toUpperCase(),
    style: {},
    children: [],
    value: "0",
    checked: false,
    hidden: false,
    textContent: "",
    innerHTML: "",
    className: "",
    clientWidth: 800,
    clientHeight: 360,
    width: 0,
    height: 0,
    classList: { add() {}, remove() {}, contains: () => false, toggle() {} },
  };
  return new Proxy(base, {
    get(o, k) {
      if (k in o) return o[k];
      switch (k) {
        case "getContext": return () => ctxStub;
        case "addEventListener": return () => {};
        case "appendChild": return (c) => { o.children.push(c); return c; };
        case "querySelectorAll": return () => [];
        case "querySelector": return () => makeEl();
        default: return () => {};
      }
    },
    set(o, k, v) { o[k] = v; return true; },
  });
}

const elById = new Map();
globalThis.document = {
  getElementById(id) {
    if (!elById.has(id)) elById.set(id, makeEl());
    return elById.get(id);
  },
  querySelectorAll: () => [],
  querySelector: () => makeEl(),
  createElement: (tag) => makeEl(tag),
};
globalThis.window = {
  devicePixelRatio: 1,
  addEventListener: () => {},
};

// fetch: serve JSON straight from docs/ on disk (mirrors http.server behaviour)
globalThis.fetch = async (url) => {
  try {
    const body = readFileSync(join(DOCS, url), "utf8");
    return { ok: true, status: 200, json: async () => JSON.parse(body) };
  } catch {
    return { ok: false, status: 404, json: async () => ({}) };
  }
};

// ── Run the real app ────────────────────────────────────────────────────────
await import(join(DOCS, "js", "app.js"));
await new Promise((r) => setTimeout(r, 200)); // let async redraws settle

// Spot-check that the boot actually populated the DOM stubs
const readout = elById.get("tier1-readout").innerHTML;
if (!readout.includes("m/s")) {
  console.error("FAIL: tier1 readout never populated:", JSON.stringify(readout));
  process.exit(1);
}
const t2 = elById.get("tier2-readout").innerHTML;
if (!t2.includes("m/s") && !t2.includes("no spread")) {
  console.error("FAIL: tier2 readout never populated:", JSON.stringify(t2));
  process.exit(1);
}
const nPresetChips = elById.get("preset-buttons").children.length;
if (nPresetChips !== 5) {
  console.error(`FAIL: expected 5 preset chips, got ${nPresetChips}`);
  process.exit(1);
}
console.log("OK — site boot path ran clean against real exported data");
console.log(`  tier1 readout: ${readout.replace(/<[^>]+>/g, " ").slice(0, 80)}…`);
console.log(`  tier2 readout: ${t2.replace(/<[^>]+>/g, " ").slice(0, 80)}…`);
