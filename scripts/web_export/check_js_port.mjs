#!/usr/bin/env node
/**
 * check_js_port.mjs — verify docs/js/empirical.js matches the Python models.
 *
 * Reads scripts/web_export/tier1_reference.json (written by
 * export_tier1_reference.py) and asserts the JS port reproduces every
 * value to within float round-off.  Run from the repo root:
 *
 *     .venv/bin/python scripts/web_export/export_tier1_reference.py
 *     node scripts/web_export/check_js_port.mjs
 */

import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";

import {
  cheneyEq6Ros,
  marsdenSmedleyRos,
  marsdenSmedleyPSustain,
} from "../../docs/js/empirical.js";

const here = dirname(fileURLToPath(import.meta.url));
const ref = JSON.parse(readFileSync(join(here, "tier1_reference.json"), "utf8"));

const TOL = 1e-12;
let failures = 0;
let checked = 0;

function check(label, got, want) {
  checked += 1;
  const err = Math.abs(got - want);
  if (err > TOL * Math.max(1, Math.abs(want))) {
    failures += 1;
    console.error(`FAIL ${label}: js=${got} python=${want} |err|=${err}`);
  }
}

for (const c of ref.cheney_eq6) {
  check(
    `cheney(U=${c.U_m_s}, M=${c.moisture_frac}, a=${c.a_ch})`,
    cheneyEq6Ros(c.U_m_s, c.moisture_frac, c.a_ch),
    c.ros_m_s,
  );
}

for (const c of ref.marsden_smedley) {
  check(
    `ms(U=${c.U_m_s}, M=${c.moisture_frac}, age=${c.age_yr})`,
    marsdenSmedleyRos(c.U_m_s, c.moisture_frac, c.age_yr),
    c.ros_m_s,
  );
  check(
    `ms_p_sustain(U=${c.U_m_s}, M=${c.moisture_frac})`,
    marsdenSmedleyPSustain(c.U_m_s, c.moisture_frac),
    c.p_sustain,
  );
}

// README anchor: U=4 m/s, M=4%, natural pasture → 52.37 m/min.
const anchor = cheneyEq6Ros(4.0, 0.04, 0.406) * 60.0;
if (Math.abs(anchor - 52.37) > 0.01) {
  failures += 1;
  console.error(`FAIL README anchor: got ${anchor.toFixed(2)} m/min, want 52.37`);
}

if (failures > 0) {
  console.error(`\n${failures}/${checked + 1} checks FAILED`);
  process.exit(1);
}
console.log(`OK — ${checked} checks passed, JS port matches Python (anchor ${anchor.toFixed(2)} m/min)`);
