/**
 * CLI cross-check runner — node docs/test.mjs
 *
 * Same checks as docs/test.html (both call runChecks in js/selftest.js).
 * Exits non-zero on any failure so it can gate a commit or CI job.
 */
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';

import { runChecks } from './js/selftest.js';

const here = dirname(fileURLToPath(import.meta.url));
const golden = JSON.parse(readFileSync(join(here, 'data', 'golden.json'), 'utf8'));

const groups = runChecks(golden);
const width = Math.max(...groups.map(g => g.name.length));

for (const g of groups) {
  console.log(`  ${g.ok ? 'PASS' : 'FAIL'}  ${g.name.padEnd(width)}  ${g.detail}`);
  if (g.extra) console.log(`        ${g.extra}`);
}

const nPass = groups.filter(g => g.ok).length;
console.log(`\n${nPass}/${groups.length} checks passed`);
process.exit(nPass === groups.length ? 0 : 1);
