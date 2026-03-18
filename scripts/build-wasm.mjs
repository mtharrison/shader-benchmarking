import { copyFileSync, existsSync, mkdirSync } from 'node:fs';
import { dirname, join, resolve } from 'node:path';
import { execFileSync } from 'node:child_process';

const rootDir = resolve(dirname(new URL(import.meta.url).pathname), '..');
const wasmDir = join(rootDir, 'wasm');
const pkgDir = join(wasmDir, 'pkg');
const outputFile = join(pkgDir, 'matrix_wasm.wasm');

function ensureWasmTarget() {
  const installedTargets = execFileSync('rustup', ['target', 'list', '--installed'], {
    encoding: 'utf8',
  });

  if (installedTargets.includes('wasm32-unknown-unknown')) {
    return;
  }

  execFileSync('rustup', ['target', 'add', 'wasm32-unknown-unknown'], {
    stdio: 'inherit',
  });
}

ensureWasmTarget();

execFileSync(
  'cargo',
  ['build', '--target', 'wasm32-unknown-unknown', '--release'],
  {
    cwd: wasmDir,
    stdio: 'inherit',
  },
);

mkdirSync(pkgDir, { recursive: true });

const wasmArtifact = join(
  wasmDir,
  'target',
  'wasm32-unknown-unknown',
  'release',
  'matrix_wasm.wasm',
);

if (!existsSync(wasmArtifact)) {
  throw new Error(`Expected wasm artifact at ${wasmArtifact}`);
}

copyFileSync(wasmArtifact, outputFile);

console.log(`WASM module written to ${outputFile}`);
